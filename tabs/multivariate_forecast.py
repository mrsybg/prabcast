# tabs/multivariate_forecast.py
import streamlit as st
from setup_module.helpers import *
from setup_module.session_state import get_app_state
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objects as go
from setup_module.evaluation import calculate_metrics, measure_performance
from setup_module.error_handler import ErrorHandler, UserFeedback
from setup_module.exceptions import DataValidationError, ModelTrainingError
from setup_module.logging_config import log_data_operation, log_model_training
from app.models_multi import build_lstm_model, build_xgboost_model
import pandas as pd
import time
import tracemalloc
# ✨ UX/UI Components
from setup_module.design_system import UIComponents
from setup_module.ui_helpers import (
    safe_execute,
    run_with_progress,
    export_dialog
)

UI = UIComponents()

def create_sequences(X, y, seq_length=12):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

@st.cache_resource(ttl=3600, max_entries=5)  # Cache für 1h, max 5 Modelle
def train_multivariate_model(model_type, X_train, y_train, input_shape=None):
    """Cache model training with performance measurement"""
    def fit_model():
        start_time = time.time()
        tracemalloc.start()
        
        try:
            if model_type == "LSTM":
                model = build_lstm_model(input_shape)
                model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
            else:  # XGBoost
                model = build_xgboost_model()
                model.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return model, train_time, peak / 10**6  # Convert to MB
            
        except Exception as e:
            tracemalloc.stop()
            raise e
    
    return fit_model()

def display_tab():
    """Tab mit strukturiertem Session State Management."""
    state = get_app_state()
    
    # ✨ PROFESSIONELL: Kompakte Info-Box mit Collapsible
    UI.info_message("""
    **Multivariate Absatzprognose:** Nutzen Sie externe Einflussfaktoren (Indizes) zusammen mit historischen 
    Verkaufsdaten für präzisere Prognosen. Die Datenbasis stammt aus der Datenanreicherung.
    
    **Wichtig:** Sie müssen zuerst im Tab 'Datenanreicherung' eine Analyse durchführen.
    """, collapsible=True)
        
    with st.expander("Informationen zu den Metriken und dem Performance Vergleich"):
        st.write("""
            **Metriken**:
            1. **MAE (Mean Absolute Error)**:

            Der **MAE** misst den durchschnittlichen Betrag der Fehler zwischen den vorhergesagten 
            und den tatsächlichen Verkaufszahlen. Er wird berechnet, indem man die absoluten Differenzen zwischen den 
            vorhergesagten und den tatsächlichen Werten nimmt und dann den Durchschnitt dieser Werte bildet.

            - **Vorteil**: MAE ist intuitiv und gibt die durchschnittliche Fehlergröße direkt in den Einheiten der 
            Verkaufszahlen an, ohne dass überproportional große Fehler stärker gewichtet werden.

            - **Nachteil**: MAE ignoriert die Richtung der Fehler (ob die Vorhersagen zu hoch oder zu niedrig sind). 

            2. **RMSE (Root Mean Squared Error)**:

            Der **RMSE** ist ebenfalls eine Maßzahl für die Differenzen zwischen den vorhergesagten und den 
            tatsächlichen Werten. Er berechnet den Durchschnitt der quadrierten Fehler, bevor dann die Quadratwurzel gezogen wird.

            - **Vorteil**: RMSE betont größere Fehler stärker, da sie quadriert werden. Das ist hilfreich, wenn man große Fehler vermeiden will.

            - **Nachteil**: Wegen der Quadrate kann RMSE von Ausreißern stark beeinflusst werden.

            3. **sMAPE (Symmetric Mean Absolute Percentage Error)**:

            Der **sMAPE** ist ein relatives Maß für die Genauigkeit der Vorhersage, das in Prozent ausgedrückt wird. 
            Es handelt sich um eine verbesserte Version des MAPE (Mean Absolute Percentage Error), um Symmetrie 
            zwischen Über- und Unterschätzungen zu gewährleisten.

            - **Vorteil**: sMAPE ist nützlich, um den Vorhersagefehler im Verhältnis zur Größe der tatsächlichen Werte zu bewerten.

            - **Nachteil**: sMAPE kann instabil werden, wenn die tatsächlichen Werte nahe null liegen.

            4. **Bias**:

            Der **Bias** misst die systematische Verzerrung des Modells, also ob das Modell im Durchschnitt zu hohe oder zu 
            niedrige Vorhersagen macht. Ein positiver Bias deutet darauf hin, dass die Vorhersagen tendenziell höher 
            als die tatsächlichen Werte sind, während ein negativer Bias auf zu niedrige Vorhersagen hinweist.

            - **Vorteil**: Der Bias zeigt auf, ob eine systematische Abweichung in den Vorhersagen besteht.

            - **Nachteil**: Bias alleine gibt keine Informationen über die Genauigkeit der einzelnen Vorhersagen, sondern nur über die generelle Tendenz.

            5. **Theil’s U (Theil's Inequality Coefficient)**:

            **Theil’s U** ist ein Maß für die Vorhersagegenauigkeit. Es vergleicht das Verhältnis zwischen dem RMSE des 
            Modells und dem RMSE einer "naiven" Vorhersage, die keine intelligenten Vorhersagemechanismen anwendet 
            (z. B. einfach den letzten bekannten Wert für die nächste Periode verwendet).

            - **Vorteil**: Theil’s U zeigt an, wie gut das Modell im Vergleich zu einer einfachen, naiven Methode ist. 
            Ein Wert unter 1 bedeutet, dass das Modell besser ist als die naive Methode.

            - **Nachteil**:  Es ist komplexer zu interpretieren als andere Fehlermaße.

             6. **Trainingszeit**:

             Die **Trainingszeit** gibt die Zeit an, die das Modell benötigt, um aus den Trainingsdaten zu lernen. 
             Dies umfasst das Anpassen der Modellparameter und die Ausführung von Optimierungsprozessen.
             Die Trainingszeit ist ein wichtiger Faktor in der Modellwahl, besonders wenn man mit großen Datensätzen 
             oder in Echtzeitanwendungen arbeitet. Längere Trainingszeiten können in ressourcenbegrenzten Umgebungen problematisch sein.

             7. **Memory Usage**:

             Die **Memory Usage** beschreibt den Speicherverbrauch des Modells während des Trainings und der Vorhersage. 
             Dieser Aspekt ist besonders relevant, wenn große Modelle oder Datenmengen verarbeitet werden müssen.

             **Performance Vergleich**

             Das Diagramm visualisiert die 3 Metriken **sMAPE**, **Trainingszeit in Sekunden** und den **Speicherverbrauch**.
             Dabei liegt der sMAPE auf der Y-Achse und die Trainingszeit auf der X-Achse. Das optimale Modell liegt also im Nullpunkt.
             Wenn die Trainingszeit aber kein limitierender Faktor ist, sollte diese nicht überbewertet werden. 
             Der Speicherverbrauch wird über das Volumen des Punktes beschrieben. Je kleiner dabei der Punkt, desto 
             kleiner der Speicherverbrauch.
        """)

    st.header("Multivariate Absatzprognose")

    if 'multivariate_data' not in st.session_state or st.session_state.multivariate_data is None:
        UserFeedback.warning("Bitte führen Sie zuerst die erweiterte Analyse durch, um Indexdaten zu generieren")
        return

    # Get data from session state
    data = state.forecast.multivariate_data
    if data is None or data.empty:
        UserFeedback.error("Keine Daten für die Prognose verfügbar")
        return
    
    # Copy data to avoid modifying original
    data = data.copy()

    # Add expander to display the dataset
    with st.expander("Verwendeter Datensatz für die Prognose"):
        st.write("Diese Daten werden für die multivariate Prognose verwendet. Die erste Spalte enthält die Verkaufszahlen des Produkts, die weiteren Spalten enthalten die relevanten Indizes aus der Datenanreicherung.")
        st.dataframe(data)

    #Produkt aus der Datenanreicherung anzeigen
    st.write(f"In der Datenanreicherung gewähltes Produkt: **{data.columns[0]}**")

    # Forecast horizon selection
    forecast_horizon = st.slider(
        "Wählen Sie den Prognosehorizont (Monate)", 
        min_value=1, 
        max_value=12, 
        value=12,
        key='mv_forecast_horizon',
        help="Anzahl der Monate für die Prognose. Je kürzer der Horizont, desto genauer die Vorhersage."
    )
    
    # ✨ NEU: Professioneller Button
    if UI.primary_button("Modellvergleich durchführen", key="mv_forecast_btn", help="Trainiert LSTM und XGBoost Modelle"):
        with st.spinner("Trainiere Modelle und erstelle Prognosen..."):
            try:
                # Prepare data
                target_col = data.columns[0]  # First column is sales
                features = data.drop(columns=[target_col])
                
                # Normalize data
                scaler_y = MinMaxScaler()
                scaler_X = MinMaxScaler()
                
                y = scaler_y.fit_transform(data[[target_col]])
                X = scaler_X.fit_transform(features)
                
                # Create sequences
                seq_length = 12
                X_seq, y_seq = create_sequences(X, y, seq_length)
                
                # Split data
                train_size = len(X_seq) - forecast_horizon
                X_train, X_test = X_seq[:train_size], X_seq[train_size:]
                y_train, y_test = y_seq[:train_size], y_seq[train_size:]
                
                # Prepare XGBoost data
                X_train_xgb = X_train.reshape(X_train.shape[0], -1)
                X_test_xgb = X_test.reshape(X_test.shape[0], -1)
                
                # Train models with performance measurement
                lstm_model, lstm_time, lstm_memory = train_multivariate_model(
                    "LSTM", X_train, y_train, input_shape=(seq_length, X.shape[1])
                )
                
                xgb_model, xgb_time, xgb_memory = train_multivariate_model(
                    "XGBoost", X_train_xgb, y_train
                )
                
                # Store models and scalers in session state for future use
                st.session_state.trained_models = {
                    'LSTM': lstm_model,
                    'XGBoost': xgb_model
                }
                
                st.session_state.scalers = {
                    'X': scaler_X,
                    'y': scaler_y
                }
                
                st.session_state.model_info = {
                    'seq_length': seq_length,
                    'feature_shape': X.shape[1],
                    'target_col': target_col,
                    'last_training_date': data.index[-1]
                }
                
                # Generate forecasts
                lstm_forecast = lstm_model.predict(X_test)
                xgb_forecast = xgb_model.predict(X_test_xgb)
                
                # Inverse transform predictions
                lstm_forecast = scaler_y.inverse_transform(lstm_forecast)
                xgb_forecast = scaler_y.inverse_transform(xgb_forecast.reshape(-1, 1))
                y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                
                # Calculate metrics with performance data
                lstm_metrics = calculate_metrics(
                    y_test_orig, lstm_forecast, 
                    execution_time=lstm_time, 
                    memory_used=lstm_memory
                )
                xgb_metrics = calculate_metrics(
                    y_test_orig, xgb_forecast, 
                    execution_time=xgb_time, 
                    memory_used=xgb_memory
                )
                
                # Create forecast DataFrame
                forecast_dates = data.index[-forecast_horizon:]
                forecast_df = pd.DataFrame({
                    'Actual': y_test_orig.flatten(),
                    'LSTM': lstm_forecast.flatten(),
                    'XGBoost': xgb_forecast.flatten()
                }, index=forecast_dates)
                
                # Display results
                st.subheader("Prognoseergebnisse")
                
                # Plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data.index[:-forecast_horizon],
                    y=data[target_col][:-forecast_horizon],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Actual test data
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_df['Actual'],
                    mode='lines',
                    name='Actual',
                    line=dict(color='black')
                ))
                
                # Forecasts
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_df['LSTM'],
                    mode='lines',
                    name='LSTM Forecast',
                    line=dict(color='red')
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_df['XGBoost'],
                    mode='lines',
                    name='XGBoost Forecast',
                    line=dict(color='green')
                ))
                
                fig.update_layout(
                    title="Modellvergleich der Prognosen",
                    xaxis_title="Datum",
                    yaxis_title="Wert",
                    height=600
                )
                
                st.plotly_chart(fig)
                
                # Display metrics
                st.subheader("Metriken")
                metrics_df = pd.DataFrame({
                    'LSTM': lstm_metrics,
                    'XGBoost': xgb_metrics
                })

                metrics_df = metrics_df.T

                # format table
                metrics_df_styled = format_summary_table(metrics_df, metrics_df.columns[0:2].tolist())

                st.dataframe(metrics_df_styled)

                if 'metrics_df_styled_easy' in st.session_state and st.session_state.metrics_df_styled_easy is not None:
                    st.subheader("Metriken der einfachen Prognose zum Vergleichen")
                    st.write("Zum Vergleichen bitte den selben Prognosehorizont/Produkt wählen.")
                    st.dataframe(st.session_state.metrics_df_styled_easy)
                
                # Add performance visualization
                st.subheader("Performance Vergleich")
                
                # Create performance comparison plot
                perf_fig = go.Figure()
                
                model_names = ["LSTM", "XGBoost"]
                time_values = [lstm_time, xgb_time]
                smape_values = [
                    lstm_metrics['sMAPE'],
                    xgb_metrics['sMAPE']
                ]
                memory_values = [lstm_memory, xgb_memory]
                
                # Create bubble chart
                perf_fig.add_trace(go.Scatter(
                    x=time_values,
                    y=smape_values,
                    mode='markers',
                    marker=dict(
                        size=[m/5 for m in memory_values],
                        sizemode='area',
                        sizeref=2.*max(memory_values)/(40.**2),
                        sizemin=4
                    ),
                    text=[f"{name}<br>Memory: {memory:.1f}MB" 
                         for name, memory in zip(model_names, memory_values)],
                    hoverinfo='text'
                ))
                
                perf_fig.update_layout(
                    title="Modell Performance Vergleich",
                    xaxis_title="Trainingszeit (Sekunden)",
                    yaxis_title="sMAPE (%)",
                    height=400
                )
                
                st.plotly_chart(perf_fig)
                
                # Add a button to save models for complex forecasting
                save_col1, save_col2 = st.columns([1, 2])
                with save_col1:
                    # ✨ NEU: Professioneller Button
                    if UI.secondary_button("Modelle speichern", key='save_models_btn', help="Speichert trainierte Modelle für erweiterte Prognose"):
                        # Verwende tiefe Kopien um Referenzprobleme zu vermeiden
                        st.session_state['saved_models_for_complex'] = {
                            'models': {
                                'LSTM': lstm_model,
                                'XGBoost': xgb_model
                            },
                            'scalers': {
                                'X': scaler_X,
                                'y': scaler_y
                            },
                            'info': {
                                'seq_length': seq_length,
                                'feature_shape': X.shape[1],
                                'target_col': target_col,
                                'last_training_date': data.index[-1].strftime('%Y-%m-%d')
                            }
                        }
                        # Verwende state.forecast.saved_model_status für die Statusmeldung
                        st.session_state['saved_model_status'] = f"Modelle für {target_col} erfolgreich gespeichert!"
                        # Rerun um die Statusmeldung anzuzeigen ohne Reset
                        st.rerun()
                
                with save_col2:
                    # Zeige Statusmeldung wenn vorhanden
                    if 'saved_model_status' in st.session_state:
                        st.success(st.session_state['saved_model_status'])
                        # Optional: Entferne die Statusmeldung nach dem Anzeigen
                        # del st.session_state['saved_model_status']
                
                # Download buttons replaced with export_dialog
                export_dialog(
                    data=forecast_df,
                    filename_base="multivariate_prognosen",
                    metadata={
                        'Modelle': 'LSTM, XGBoost',
                        'Prognosehorizont': f"{forecast_horizon} Monate",
                        'Produkt': target_col,
                        'Training bis': data.index[-1].strftime('%Y-%m-%d')
                    }
                )
                
                export_dialog(
                    data=metrics_df,
                    filename_base="multivariate_metriken",
                    metadata={
                        'Modelle': 'LSTM, XGBoost',
                        'Produkt': target_col
                    }
                )
                
            except Exception as e:
                UserFeedback.error(f"Error in multivariate forecasting: {str(e)}")
                st.exception(e)