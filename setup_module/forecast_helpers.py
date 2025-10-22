# setup_module/forecast_helpers.py
"""
Zentrale Forecast-Helper-Funktionen
Vermeidet Code-Duplikation zwischen forecast*.py Tabs
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional
from setup_module.exceptions import ModelTrainingError, InsufficientDataError, DataValidationError
from setup_module.error_handler import ErrorHandler, UserFeedback
from setup_module.logging_config import log_data_operation, log_model_training
from setup_module.evaluation import measure_performance
from setup_module.data_validator import DataValidator


# ============================================================================
# MODEL TRAINING
# ============================================================================

@st.cache_resource(ttl=3600, max_entries=10)
@ErrorHandler.handle_errors(show_details=True)
def train_forecast_model(model_class, train_data, **params):
    """
    Universelle Model-Training-Funktion für alle Forecast-Tabs.
    Cached mit TTL und Error Handling.
    
    Args:
        model_class: Modell-Klasse (z.B. ARIMAModel, LSTMModel)
        train_data: Trainingsdaten als pandas Series
        **params: Modell-spezifische Parameter
        
    Returns:
        Trainiertes Modell
        
    Raises:
        ModelTrainingError: Bei Fehlern im Training
        InsufficientDataError: Bei zu wenig Trainingsdaten
    """
    # ✨ VALIDIERUNG: Trainingsdaten validieren (nutzt MIN_FORECAST_DATA_POINTS = 10)
    # Nutze forecast_horizon aus params falls vorhanden, sonst None
    forecast_horizon = params.get('forecast_horizon', None)
    
    validation = DataValidator.validate_forecast_data(
        train_data, 
        forecast_horizon=forecast_horizon if forecast_horizon else 1  # Mindestens 1 für Validierung
    )
    
    # Zeige Validierungsergebnis nur wenn Warnings oder Info vorhanden
    if validation.warnings or validation.info:
        validation.display_in_streamlit()
    
    # Bei Fehlern: raise (validate_forecast_data wirft InsufficientDataError wenn < 10 Datenpunkte)
    validation.raise_if_invalid()
    
    try:
        # Handle model initialization with parameters
        model = _initialize_model(model_class, params)
        
        # Measure performance during training
        @measure_performance
        def fit_model(model, data, **fit_params):
            model.fit(data, **fit_params)
            return model
        
        trained_model, train_time, memory_used = fit_model(model, train_data, **params)
        trained_model.train_time = train_time
        trained_model.memory_used = memory_used
        
        log_model_training(
            model_name=model_class.__name__,
            data_shape=(len(train_data), 1),
            duration=train_time,
            memory_mb=round(memory_used, 2)
        )
        
        return trained_model
        
    except Exception as e:
        # Log error without calling log_model_training (to avoid signature issues)
        from setup_module.logging_config import log_error
        log_error(
            error_type="ModelTraining",
            message=f"Failed to train {model_class.__name__}",
            data_points=len(train_data),
            error=str(e)
        )
        raise ModelTrainingError(
            message=f"Fehler beim Trainieren des {model_class.__name__} Modells",
            details=str(e),
            help_text="Versuchen Sie ein anderes Modell oder passen Sie die Parameter an.",
            original_exception=e
        )


def _initialize_model(model_class, params: dict):
    """
    Initialisiert Modell mit spezifischen Parametern.
    
    Args:
        model_class: Modell-Klasse
        params: Parameter-Dictionary (wird modifiziert!)
        
    Returns:
        Initialisiertes Modell
    """
    model_name = model_class.__name__
    
    if model_name == "ARIMAModel" and "order" in params:
        return model_class(order=params.pop("order"))
    
    elif model_name == "SARIMAModel" and "seasonal_order" in params:
        return model_class(seasonal_order=params.pop("seasonal_order"))
    
    elif model_name in ["LSTMModel", "GRUModel"]:
        if "epochs" in params:
            epochs = params.pop("epochs")
            batch_size = params.pop("batch_size", 1)
            return model_class(epochs=epochs, batch_size=batch_size)
        return model_class()
    
    else:
        return model_class()


# ============================================================================
# MODEL PARAMETERS UI
# ============================================================================

def get_model_params_ui(selected_models: List[str]) -> Dict[str, dict]:
    """
    Erstellt UI für Modell-Parameter und gibt Parameter-Dictionary zurück.
    
    Args:
        selected_models: Liste der ausgewählten Modellnamen
        
    Returns:
        Dictionary: {model_name: {param: value}}
    """
    params = {}
    
    with st.expander("⚙️ Modell Parameter"):
        if "ARIMA" in selected_models:
            st.subheader("ARIMA Parameter")
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.slider("AR (p)", 0, 7, 5, help="Autoregressive Ordnung")
            with col2:
                d = st.slider("Differenzierung (d)", 0, 2, 1, help="Differenzierungsgrad")
            with col3:
                q = st.slider("MA (q)", 0, 7, 0, help="Moving Average Ordnung")
            params["ARIMA"] = {"order": (p, d, q)}
            
        if "SARIMA" in selected_models:
            st.subheader("SARIMA Parameter")
            col1, col2, col3 = st.columns(3)
            with col1:
                P = st.slider("Saisonale AR (P)", 0, 2, 1)
            with col2:
                D = st.slider("Saisonale Diff (D)", 0, 1, 1)
            with col3:
                Q = st.slider("Saisonale MA (Q)", 0, 2, 1)
            seasonal_period = st.number_input("Saisonale Periode", min_value=2, max_value=365, value=12)
            params["SARIMA"] = {"seasonal_order": (P, D, Q, seasonal_period)}
            
        if "Prophet" in selected_models:
            st.subheader("Prophet Parameter")
            col1, col2 = st.columns(2)
            with col1:
                yearly = st.selectbox("Yearly Seasonality", [True, False, "auto"], index=2)
            with col2:
                weekly = st.selectbox("Weekly Seasonality", [True, False, "auto"], index=2)
            params["Prophet"] = {
                "yearly_seasonality": yearly,
                "weekly_seasonality": weekly
            }
            
        if any(x in selected_models for x in ["LSTM", "GRU"]):
            st.subheader("Neural Network Parameter")
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Epochs", 10, 200, 50, help="Trainingsiterationen")
            with col2:
                batch_size = st.slider("Batch Size", 1, 32, 1, help="Batch-Größe")
            for model in ["LSTM", "GRU"]:
                if model in selected_models:
                    params[model] = {"epochs": epochs, "batch_size": batch_size}
    
    return params


def get_available_models() -> List[str]:
    """
    Gibt Liste aller verfügbaren Forecast-Modelle zurück.
    
    Returns:
        Liste der Modellnamen
    """
    return [
        "ARIMA", "Prophet", "LSTM", "GRU", "SES", 
        "SARIMA", "Holt-Winters", "XGBoost", 
        "Random Forest", "Moving Average", "Seasonal Naive", 
        "Transformer", "Ensemble"
    ]


def create_model_selection_ui(
    default_models: Optional[List[str]] = None,
    key_suffix: str = ""
) -> List[str]:
    """
    Erstellt UI für Modell-Auswahl.
    
    Args:
        default_models: Standard-Modelle (None = ["ARIMA"])
        key_suffix: Suffix für st.multiselect key (wichtig für Uniqueness)
        
    Returns:
        Liste der ausgewählten Modelle
    """
    available = get_available_models()
    default = default_models if default_models else ["ARIMA"]
    
    return st.multiselect(
        "Wählen Sie die Modelle für die Prognose",
        available,
        default=default,
        key=f'model_selection_{key_suffix}',
        help="Mehrere Modelle können für Vergleich ausgewählt werden"
    )


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_forecast_data(
    df: pd.DataFrame,
    date_column: str,
    product_column: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    resample_freq: str = 'M'
) -> pd.Series:
    """
    Bereitet Daten für Forecast vor: Datums-Parsing, Filterung, Resampling.
    
    Args:
        df: Input DataFrame
        date_column: Name der Datumsspalte
        product_column: Name der Produktspalte
        start_date: Optional Startdatum für Filterung
        end_date: Optional Enddatum für Filterung
        resample_freq: Resample-Frequenz ('D', 'W', 'M', 'Q', 'Y')
        
    Returns:
        Vorbereitete pandas Series mit Datums-Index
    """
    # Copy to avoid modifying original
    data = df[[date_column, product_column]].copy()
    
    # Parse dates
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    data = data.dropna(subset=[date_column])
    
    # Filter by date range if specified
    if start_date is not None or end_date is not None:
        if start_date is not None:
            data = data[data[date_column] >= start_date]
        if end_date is not None:
            data = data[data[date_column] <= end_date]
    
    # Set index and sort
    data.set_index(date_column, inplace=True)
    data = data.sort_index()
    
    # Resample
    result = data[product_column].resample(resample_freq).sum()
    
    log_data_operation(
        "Forecast-Daten vorbereitet",
        data_shape=(len(result), 1),
        product=product_column,
        freq=resample_freq
    )
    
    return result


def split_train_test(
    data: pd.Series,
    forecast_horizon: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Teilt Daten in Training und Test basierend auf Prognosehorizont.
    
    Args:
        data: Zeitreihen-Daten
        forecast_horizon: Anzahl der Perioden für Prognose
        
    Returns:
        Tuple (train_data, test_data)
        
    Raises:
        DataValidationError: Wenn forecast_horizon ungültig
        InsufficientDataError: Wenn zu wenig Daten
    """
    # ✨ VALIDIERUNG: Nutze DataValidator statt manuelle Checks
    validation = DataValidator.validate_forecast_data(data, forecast_horizon)
    validation.raise_if_invalid()  # Wirft Exception bei Fehlern
    
    train = data.iloc[:-forecast_horizon]
    test = data.iloc[-forecast_horizon:]
    
    return train, test


# ============================================================================
# VISUALIZATION
# ============================================================================

@ErrorHandler.handle_errors(show_details=False)
def create_forecast_chart(
    historical_data: pd.Series,
    forecasts: Dict[str, pd.Series],
    test_data: Optional[pd.Series] = None,
    title: str = "Forecast Vergleich",
    product_name: str = ""
) -> go.Figure:
    """
    Erstellt standardisierte Forecast-Visualisierung.
    
    Args:
        historical_data: Historische Daten
        forecasts: Dictionary {model_name: forecast_series}
        test_data: Optional Test-Daten zum Vergleich
        title: Chart-Titel
        product_name: Produktname für Untertitel
        
    Returns:
        Plotly Figure
        
    Raises:
        VisualizationException: Bei Fehlern in der Visualisierung
    """
    from setup_module.exceptions import VisualizationException
    
    try:
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode='lines',
            name='Historisch',
            line=dict(color='blue', width=2)
        ))
        
        # Test data (actual values) if available
        if test_data is not None:
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=test_data.values,
                mode='lines',
                name='Tatsächlich',
                line=dict(color='green', width=2, dash='dot')
            ))
        
        # Forecasts
        colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        for idx, (model_name, forecast) in enumerate(forecasts.items()):
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines',
                name=f'{model_name} Prognose',
                line=dict(color=color, width=2, dash='dash')
            ))
        
        # Layout
        fig.update_layout(
            title=f"{title}<br><sub>{product_name}</sub>",
            xaxis_title="Datum",
            yaxis_title="Wert",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        return fig
        
    except Exception as e:
        raise VisualizationException(
            message="Fehler bei der Erstellung des Forecast-Charts",
            details=str(e),
            help_text="Überprüfen Sie die Forecast-Daten.",
            original_exception=e
        )


def calculate_forecast_metrics(
    actual: pd.Series,
    predicted: pd.Series
) -> Dict[str, float]:
    """
    Berechnet Standard-Metriken für Forecast-Qualität.
    
    Args:
        actual: Tatsächliche Werte
        predicted: Vorhergesagte Werte
        
    Returns:
        Dictionary mit Metriken (MAE, RMSE, sMAPE, etc.)
    """
    import numpy as np
    
    # Align series
    actual_aligned = actual.reindex(predicted.index)
    
    # Remove NaN
    mask = ~(actual_aligned.isna() | predicted.isna())
    actual_clean = actual_aligned[mask].values
    predicted_clean = predicted[mask].values
    
    if len(actual_clean) == 0:
        return {"error": "No overlapping data points"}
    
    # Calculate metrics
    mae = np.mean(np.abs(actual_clean - predicted_clean))
    rmse = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
    
    # sMAPE (symmetric Mean Absolute Percentage Error)
    denominator = (np.abs(actual_clean) + np.abs(predicted_clean)) / 2
    smape = np.mean(np.abs(actual_clean - predicted_clean) / denominator) * 100
    
    # MAPE
    mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "sMAPE": round(smape, 2),
        "MAPE": round(mape, 2),
        "n_points": len(actual_clean)
    }


def display_forecast_metrics(metrics_dict: Dict[str, Dict[str, float]]):
    """
    Zeigt Forecast-Metriken in formatierter Tabelle an.
    
    Args:
        metrics_dict: Dictionary {model_name: metrics_dict}
    """
    if not metrics_dict:
        return
    
    # Convert to DataFrame for nice display
    df_metrics = pd.DataFrame(metrics_dict).T
    
    # Sort by sMAPE (lower is better)
    if 'sMAPE' in df_metrics.columns:
        df_metrics = df_metrics.sort_values('sMAPE')
    
    st.subheader("📊 Modell-Metriken")
    st.dataframe(
        df_metrics.style.highlight_min(
            subset=['MAE', 'RMSE', 'sMAPE', 'MAPE'],
            color='lightgreen'
        ),
        use_container_width=True
    )
    
    # Best model
    if 'sMAPE' in df_metrics.columns and len(df_metrics) > 0:
        best_model = df_metrics['sMAPE'].idxmin()
        best_smape = df_metrics.loc[best_model, 'sMAPE']
        st.success(f"✅ Bestes Modell: **{best_model}** (sMAPE: {best_smape:.2f}%)")
