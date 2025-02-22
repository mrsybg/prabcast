import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from setup_module.helpers import *
from sklearn.preprocessing import MinMaxScaler
from app.models_multi import *  # Use absolute imports

def display_tab():
    with st.expander("Informationen zu diesem Tab"):
        st.write("""
       In diesem Tab können Sie eine **multivariate Absatzprognose** erstellen, die neben den historischen 
       Verkaufsdaten auch externe Einflussfaktoren (Indizes) berücksichtigt. Die Datenbasis wird aus der 
       Datenanreicherung übernommen und muss daher zuerst erstellt werden. Die Vorhersage wird für das Produkt aus 
       der Datenanreicherung erstellt.

        - **Datenbasis:** Verwendet werden die angereicherten Daten aus der Datenanreicherung.
        - **Prognosehorizont:** Sie können den Prognosezeitraum in Monaten festlegen.
        - **Modelle:** Auswahl zwischen verschiedenen multivariaten Modellen (z.B. LSTM, XGBoost). Hier können nur 
        Modelle verwendet werden, welche sich für die multivariate Prognose eignen.
        - **Ergebnisse:** Die Prognose kann visualisiert und als CSV heruntergeladen werden.
        """)



    if 'multivariate_data' not in st.session_state:
        st.warning("Bitte führen Sie zuerst die Datenanreicherung durch, um die benötigten Daten zu generieren.")
        return

    st.header("Komplexe Prognose")
    # Daten laden
    data = st.session_state.multivariate_data.copy()

    if data.empty:
        st.error("Keine Daten für die Prognose verfügbar.")
        return

    # Zielvariable auswählen
    target_col = data.columns[0]

    st.write(f"In der Datenanreicherung gewähltes Produkt: **{data.columns[0]}**")

    # Prognosehorizont auswählen
    forecast_horizon = st.slider(
        "Wähle den Prognosehorizont (in Monaten)",
        min_value=1,
        max_value=24,
        value=12,
        step=1,
        key='forecast_complex_forecast_horizon'
    )

    # Modelle auswählen
    available_models = ["LSTM", "XGBoost"]
    selected_models = st.multiselect(
        "Wähle die Modelle für die Prognose",
        available_models,
        default=["LSTM"],
        key='forecast_complex_selected_models'
    )

    if st.button("Prognose durchführen", key='forecast_complex_prognose_button'):
        with st.spinner("Führe multivariate Prognose durch..."):
            try:
                # Daten vorbereiten
                features = data  # Zielvariable bleibt enthalten
                target = data[target_col]

                # Normalisierung
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()

                X_scaled = scaler_X.fit_transform(features)
                y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

                # Sequenzen erstellen
                seq_length = 12
                X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

                # Gesamte Daten zum Training verwenden
                X_train = X_seq
                y_train = y_seq

                # Zukünftige Daten für die Prognose vorbereiten
                last_sequence = X_scaled[-seq_length:]

                forecast_results = {}

                for model_name in selected_models:
                    
                    if model_name == "LSTM":
                        model = build_lstm_model(input_shape=(seq_length, X_scaled.shape[1]))
                        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                        forecast_input = last_sequence.reshape(1, seq_length, X_scaled.shape[1])
                        forecast_scaled = []
                        for _ in range(forecast_horizon):
                            pred = model.predict(forecast_input)
                            forecast_scaled.append(pred[0, 0])
                            forecast_input = np.roll(forecast_input, -1, axis=1)
                            forecast_input[0, -1, :] = np.append(pred, forecast_input[0, -1, 1:])
                        forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
                    elif model_name == "XGBoost":
                        X_train_flat = X_train.reshape(X_train.shape[0], -1)
                        model = build_xgboost_model()
                        model.fit(X_train_flat, y_train)
                        forecast_input = last_sequence.reshape(1, -1)
                        forecast_scaled = []
                        for _ in range(forecast_horizon):
                            pred = model.predict(forecast_input)
                            forecast_scaled.append(pred[0])
                            forecast_input = np.roll(forecast_input, -1)
                            forecast_input[0, -1] = pred
                        forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
                    else:
                        st.error(f"Unbekanntes Modell: {model_name}")
                        continue

                    # Inverse Transformation
                    forecast = scaler_y.inverse_transform(forecast_scaled)

                    # Erstelle Datumsindex für Prognose
                    last_date = data.index[-1]
                    forecast_dates = pd.date_range(last_date, periods=forecast_horizon+1, freq='M')[1:]

                    forecast_series = pd.Series(forecast.flatten(), index=forecast_dates)
                    forecast_results[model_name] = forecast_series

                # Plotten der Ergebnisse mit Plotly
                fig = go.Figure()

                # Historische Daten
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[target_col],
                    mode='lines',
                    name='Historische Daten'
                ))

                # Prognosen
                for name, forecast in forecast_results.items():
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast.values,
                        mode='lines',
                        name=f'Prognose ({name})'
                    ))

                fig.update_layout(
                    title=f"Prognose für {target_col}",
                    xaxis_title="Datum",
                    yaxis_title="Wert",
                    height=600
                )

                st.plotly_chart(fig)

                # Prognosen anzeigen und herunterladen
                if forecast_results:
                    all_forecasts = pd.DataFrame(forecast_results)
                    st.subheader("Prognoseergebnisse")

                    # format table
                    all_forecasts_styled = format_summary_table(all_forecasts, all_forecasts.columns[0:2].tolist())

                    st.dataframe(all_forecasts_styled)

                    csv = all_forecasts.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Prognose als CSV",
                        data=csv,
                        file_name='prognose_komplex.csv',
                        mime='text/csv',
                        key='forecast_complex_download_button'
                    )
                else:
                    st.warning("Keine Prognoseergebnisse verfügbar.")

            except Exception as e:
                st.error(f"Fehler bei der Durchführung der Prognose: {str(e)}")

def create_sequences(X, y, seq_length=12):
    """Erstellt Sequenzen für die Zeitreihenvorhersage."""
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)