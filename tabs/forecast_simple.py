# forecast_simple.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from setup_module.helpers import *
from app.models import (
    ARIMAModel, ProphetModel, LSTMModel, GRUModel, SESModel, 
    SARIMAModel, HoltWintersModel, XGBoostModel, RandomForestModel, 
    MovingAverageModel, SeasonalNaiveModel, EnsembleModel
)

@st.cache_resource
def train_model(model_class, train_data, **params):
    """Cache model training"""
    # Handle model initialization with parameters
    if model_class.__name__ == "ARIMAModel" and "order" in params:
        model = model_class(order=params.pop("order"))
    elif model_class.__name__ == "SARIMAModel" and "seasonal_order" in params:
        model = model_class(seasonal_order=params.pop("seasonal_order"))
    elif model_class.__name__ in ["LSTMModel", "GRUModel"] and "epochs" in params:
        epochs = params.pop("epochs")
        batch_size = params.pop("batch_size", 1)
        model = model_class(epochs=epochs, batch_size=batch_size)
    else:
        model = model_class()
    
    model.fit(train_data, **params)
    return model

def display_tab():
    with st.expander("Informationen zu diesem Tab"):
        st.write("""
       In diesem Tab können Sie eine **univariate Absatzprognose** für ein ausgewähltes Produkt erstellen. 
       Das bedeutet, dass zukünftige Verkaufszahlen einzig auf Grundlage der vergangenen Verkaufsdaten 
       dieses einen Produkts geschätzt werden.
       
        **Ablauf der Prognoseerstellung:**
        
        1. **Produkt auswählen:** Sie bestimmen zunächst, für welches Produkt die Prognose durchgeführt werden soll.
        2. **Zeitraum festlegen:** Anschließend wählen Sie einen Prognosehorizont (z. B. in Monaten), für den 
        Vorhersagen erstellt werden.
        3. **Modelle auswählen und konfigurieren:** Wählen Sie aus verschiedenen Prognosemodellen – von klassischen 
        statistischen Verfahren bis hin zu KI-basierten Ansätzen – und passen Sie bei Bedarf Modellparameter an.
        4. **Prognose berechnen:** Nach dem Start werden die Modelle mit den historischen Daten trainiert und 
        liefern daraufhin Vorhersagen für den gewählten Zeitraum. Wenn verfügbar, werden auch Konfidenzintervalle 
        für die Prognosen angezeigt, um die Unsicherheit der Schätzungen besser einschätzen zu können.
        5. **Ergebnisse:** Die Prognose kann visualisiert und als CSV heruntergeladen werden.

        
        """)
        
    if not st.session_state.get('df') is not None:
        st.error("Bitte laden Sie zuerst Daten hoch.")
        return
        
    st.header("Einfache Absatzprognose")

    # Produkt auswählen
    selected_product = st.selectbox(
        "Wähle ein Produkt für die Prognose",
        st.session_state.selected_products_in_data,
        key='forecast_simple_product'
    )
    
    # Prognosehorizont auswählen
    forecast_horizon = st.slider(
        "Prognosehorizont (Monate)",
        min_value=1,
        max_value=24,
        value=12,
        key='forecast_simple_horizon'
    )

    # Modelle auswählen
    available_models = [
        "ARIMA", "Prophet", "LSTM", "GRU", "SES",
        "SARIMA", "Holt-Winters", "XGBoost",
        "Random Forest", "Moving Average", "Seasonal Naive"
    ]
    
    selected_models = st.multiselect(
        "Wähle die Modelle",
        available_models,
        default=["ARIMA"],
        key='forecast_simple_models'
    )

    if st.button("Prognose durchführen", key='forecast_simple_run'):
        with st.spinner("Berechne Prognose..."):
            try:
                # Daten vorbereiten
                df = st.session_state.df.copy()
                df[st.session_state.date_column] = pd.to_datetime(
                    df[st.session_state.date_column]
                )
                df.set_index(st.session_state.date_column, inplace=True)
                
                # Produktdaten extrahieren
                product_data = df[selected_product].resample('M').sum()
                
                # Modelle initialisieren
                models_dict = {
                    "ARIMA": ARIMAModel,
                    "Prophet": ProphetModel,
                    "LSTM": LSTMModel,
                    "GRU": GRUModel,
                    "SES": SESModel,
                    "SARIMA": SARIMAModel,
                    "Holt-Winters": HoltWintersModel,
                    "XGBoost": XGBoostModel,
                    "Random Forest": RandomForestModel,
                    "Moving Average": MovingAverageModel,
                    "Seasonal Naive": SeasonalNaiveModel
                }

                forecasts = {}
                
                for model_name in selected_models:
                    model_class = models_dict[model_name]
                    model = train_model(model_class, product_data)
                    forecast = model.predict(steps=forecast_horizon)
                    forecasts[model_name] = forecast

                # Visualisierung
                fig = go.Figure()

                # Historische Daten
                fig.add_trace(go.Scatter(
                    x=product_data.index,
                    y=product_data.values,
                    mode='lines',
                    name='Historische Daten'
                ))

                # Prognosen
                for name, forecast in forecasts.items():
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast.values,
                        mode='lines',
                        name=f'Prognose ({name})'
                    ))

                fig.update_layout(
                    title=f"Prognose für {selected_product}",
                    xaxis_title="Datum",
                    yaxis_title="Wert",
                    height=600
                )

                st.plotly_chart(fig)

                # Download der Prognosen
                forecasts_df = pd.DataFrame(forecasts)
                st.download_button(
                    "Download Prognosen (CSV)",
                    forecasts_df.to_csv().encode('utf-8'),
                    "prognosen.csv",
                    "text/csv",
                    key='forecast_simple_download'
                )

            except Exception as e:
                st.error(f"Fehler bei der Prognose: {str(e)}")