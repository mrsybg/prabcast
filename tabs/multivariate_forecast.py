# tabs/multivariate_forecast.py
import streamlit as st
from setup_module.helpers import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objects as go
from setup_module.evaluation import calculate_metrics, measure_performance
from app.models_multi import build_lstm_model, build_xgboost_model
import pandas as pd
import time
import tracemalloc

def create_sequences(X, y, seq_length=12):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

@st.cache_resource
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
    with st.expander("Informationen zu diesem Tab:"):
        st.write("""
            In diesem Tab können Sie eine **multivariate Absatzprognose** erstellen, die neben den historischen 
            Verkaufsdaten auch externe Einflussfaktoren (Indizes) berücksichtigt. Dabei wird keine echte Prognose für 
            die Zukunft generiert. Vielmehr wird der Datensatz beschnitten, um die "Prognose" mit tatsächlich 
            eingetroffenen Absätzen vergleichen zu können. Die Datenbasis wird aus der Datenanreicherung übernommen 
            und muss daher zuerst erstellt werden.

            **Voraussetzung:**
            - Sie müssen zuerst im Tab 'Datenanreicherung' eine Analyse durchführen, um relevante Indexdaten zu generieren

            **Ablauf der Prognoseerstellung:**
            1. **Daten:** Es werden die Zeitreihen aus der Datenanreicherung benutzt (Dort wird auch das Produkt gewählt)
            2. **Prognosehorizont:** Wählen Sie, für wie viele Monate Sie eine Vorhersage erstellen möchten
            3. **Modellauswahl:** Verschiedene Algorithmen werden benutzt:
               - LSTM (Deep Learning)
               - XGBoost (Gradient Boosting)
            4. **Prognoseberechnung:** 
               - Training mit historischen Daten und Indizes
               - Erstellung der Vorhersagen
            5. **Ergebnisdarstellung:**
               - Visualisierung der Prognosen
               - Vergleich verschiedener Modelle
               - Download der Ergebnisse

            Multivariate Prognosen können genauer sein als univariate, da sie zusätzliche relevante 
            Informationen aus den Indexdaten berücksichtigen.
        """)
        
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

    if 'multivariate_data' not in st.session_state:
        st.warning("Bitte führen Sie zuerst die erweiterte Analyse durch, um Indexdaten zu generieren")
        return

    # Get data from session state
    data = st.session_state.multivariate_data.copy()
    
    if data.empty:
        st.error("Keine Daten für die Prognose verfügbar")
        return

    #Produkt aus der Datenanreicherung anzeigen
    st.write(f"In der Datenanreicherung gewähltes Produkt: **{data.columns[0]}**")

    # Forecast horizon selection
    forecast_horizon = st.slider(
        "Wählen Sie den Prognosehorizont (Monate)", 
        min_value=1, 
        max_value=12, 
        value=12,
        key='mv_forecast_horizon'
    )
    
    if st.button("Modellvergleich durchführen", key='mv_forecast_button'):
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
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Prognosen herunterladen",
                        forecast_df.to_csv(),
                        "modellvergleich_prognosen.csv",
                        "text/csv"
                    )
                with col2:
                    st.download_button(
                        "Metriken herunterladen",
                        metrics_df.to_csv(),
                        "modellvergleich_metriken.csv",
                        "text/csv"
                    )
                
            except Exception as e:
                st.error(f"Error in multivariate forecasting: {str(e)}")
                st.exception(e)