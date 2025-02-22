# tabs/forecast.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from setup_module.helpers import *
from setup_module.evaluation import calculate_metrics, measure_performance
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
    
    # Measure performance during training
    @measure_performance
    def fit_model(model, data, **params):
        model.fit(data, **params)
        return model
    
    trained_model, train_time, memory_used = fit_model(model, train_data, **params)
    trained_model.train_time = train_time
    trained_model.memory_used = memory_used
    return trained_model

def get_model_params(selected_models):
    """Get model-specific parameters"""
    params = {}
    
    with st.expander("Modell Parameter"):
        if "ARIMA" in selected_models:
            st.subheader("ARIMA Parameter")
            p = st.slider("AR (p)", 0, 7, 5)
            d = st.slider("Differenzierung (d)", 0, 2, 1)
            q = st.slider("MA (q)", 0, 7, 0)
            params["ARIMA"] = {"order": (p,d,q)}
            
        if "SARIMA" in selected_models:
            st.subheader("SARIMA Parameter")
            seasonal_order = (
                st.slider("Saisonale AR (P)", 0, 2, 1),
                st.slider("Saisonale Differenzierung (D)", 0, 1, 1),
                st.slider("Saisonale MA (Q)", 0, 2, 1),
                12  # Fixed seasonal period
            )
            params["SARIMA"] = {"seasonal_order": seasonal_order}
            
        if "Prophet" in selected_models:
            st.subheader("Prophet Parameter")
            yearly_seasonality = st.selectbox("Yearly Seasonality", [True, False, "auto"], index=2)
            weekly_seasonality = st.selectbox("Weekly Seasonality", [True, False, "auto"], index=2)
            params["Prophet"] = {
                "yearly_seasonality": yearly_seasonality,
                "weekly_seasonality": weekly_seasonality
            }
            
        if any(x in selected_models for x in ["LSTM", "GRU"]):
            st.subheader("Neural Network Parameter")
            epochs = st.slider("Epochs", 10, 200, 50)
            batch_size = st.slider("Batch Size", 1, 32, 1)
            for model in ["LSTM", "GRU"]:
                if model in selected_models:
                    params[model] = {"epochs": epochs, "batch_size": batch_size}
            
    return params

def display_tab():
    with st.expander("Informationen zu diesem Tab:"):
        st.write("""
            In diesem Tab können Sie eine **univariate Absatzprognose** für ein ausgewähltes Produkt erstellen. 
            Das bedeutet, dass (zukünftige) Verkaufszahlen einzig auf Grundlage der vergangenen Verkaufsdaten 
            dieses einen Produkts geschätzt werden. Dabei wird keine echte Prognose für die Zukunft generiert. 
            Vielmehr wird der Datensatz beschnitten, um die "Prognose" mit tatsächlich eingetroffenen Absätzen vergleichen zu können.

            **Ablauf der Prognoseerstellung:**
            1. **Produkt auswählen:** Sie bestimmen zunächst, für welches Produkt die Prognose durchgeführt werden soll.
            2. **Zeitraum festlegen:** Anschließend wählen Sie einen Prognosehorizont (z. B. in Monaten), für den Vorhersagen erstellt werden.
            3. **Modelle auswählen und konfigurieren:** Wählen Sie aus verschiedenen Prognosemodellen – von klassischen statistischen Verfahren bis hin zu KI-basierten Ansätzen – und passen Sie bei Bedarf Modellparameter an.
            4. **Prognose berechnen:** Nach dem Start werden die Modelle mit den historischen Daten trainiert und liefern daraufhin Vorhersagen für den gewählten Zeitraum. Wenn verfügbar, werden auch Konfidenzintervalle für die Prognosen angezeigt, um die Unsicherheit der Schätzungen besser einschätzen zu können.
            5. **Ergebnisse interpretieren:** Die historischen Daten, Prognosen und gegebenenfalls vorhandenen Unsicherheitsbereiche werden in interaktiven Grafiken dargestellt. Zusätzlich erhalten Sie Kennzahlen (Metriken) zur Prognosegüte, zur Trainingszeit und zum Ressourcenverbrauch, um verschiedene Modelle miteinander vergleichen zu können.
            6. **Daten herunterladen:** Abschließend können Sie sowohl die Prognosen als auch die Metriken als CSV-Dateien herunterladen, um sie extern weiterzuverarbeiten.

            Nutzen Sie diesen Abschnitt, um verlässliche Absatzprognosen zu erhalten, unterschiedliche Modellansätze zu testen und das für Ihre Bedürfnisse geeignete Verfahren abzuleiten.
        """)

    with st.expander("Modelle und ihre Eigenschaften - Kurzform"):
        st.write("""
            **Verwendete Modelle im Überblick:**

            - **ARIMA (AutoRegressive Integrated Moving Average):**
              Ein klassisches statistisches Modell, das vergangene Werte (autoregressiv) und Prognosefehler (Moving Average)
              nutzt. Gut für relativ stabile, nicht allzu stark saisonale Zeitreihen.

            - **Prophet:**
              Von Meta (Facebook) entwickeltes Modell, das durch einfache Handhabung und Berücksichtigung von saisonalen 
              Mustern, Feiertagen und Ausreißern überzeugt. Besonders geeignet bei unregelmäßigen Daten und Geschäftsdaten.

            - **LSTM (Long Short-Term Memory):**
              Ein neuronales Netzwerk, das speziell für Zeitreihen entwickelt wurde. LSTMs erkennen komplexe, langfristige 
              Abhängigkeiten und nicht-lineare Zusammenhänge in großen Datensätzen.

            - **GRU (Gated Recurrent Unit):**
              Eine vereinfachte Variante von LSTMs, die schneller trainiert und weniger komplex ist, aber dennoch ähnlich 
              gute Vorhersagen liefern kann. Effizient für zeitabhängige Muster.

            - **SES (Simple Exponential Smoothing):**
              Eine einfache Glättungsmethode, bei der neuere Daten stärker gewichtet werden als ältere. Für relativ 
              konstante Daten ohne starken Trend oder ausgeprägte Saisonalität geeignet.

            - **SARIMA (Seasonal ARIMA):**
              Eine Erweiterung von ARIMA, die saisonale Muster in den Daten modellieren kann. Optimal bei klar ausgeprägten 
              wiederkehrenden Mustern.

            - **Holt-Winters (Triple Exponential Smoothing):**
              Berücksichtigt neben dem Basistrend auch saisonale Komponenten. Eignet sich für Daten mit stabilen saisonalen 
              Mustern sowie langfristigen Trends.

            - **XGBoost (Extreme Gradient Boosting):**
              Ein leistungsstarkes, baumbasiertes Machine-Learning-Modell, das gut nicht-lineare Zusammenhänge erfassen kann. 
              Benötigt meist etwas Feature-Engineering, ist aber sehr flexibel.

            - **Random Forest:**
              Ein Ensemble aus vielen Entscheidungsbäumen. Robust, vielseitig und einfach anwendbar, jedoch nicht speziell 
              auf Zeitreihen zugeschnitten.

            - **Moving Average (Gleitender Durchschnitt):**
              Eine sehr einfache Methode, die den Mittelwert der zuletzt beobachteten Werte als Prognose verwendet. Eignet sich 
              als Basis- oder Referenzmodell.

            - **Seasonal Naive (Saisonale naive Prognose):**
              Nutzt den vergangenen Wert derselben Saisonperiode als Vorhersage für die Zukunft. Sehr einfach, aber sinnvoll, 
              wenn ausgeprägte, stabile Muster vorhanden sind.

            - **Ensemble:**
              Eine Kombination mehrerer Modelle, deren Vorhersagen gemischt werden, um Gesamtgenauigkeit und Robustheit zu erhöhen.
              
            Durch die Kombination unterschiedlicher Ansätze, deren Eigenschaften und Stärken Sie hier kennenlernen, können Sie 
            das für Ihre Datensituation und Ihre Anforderungen passende Modell finden und so Ihre Prognosequalität steigern.
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

    st.header("Univariate Absatzprognose")


    # Produkt auswählen
    selected_product = st.selectbox(
        "Wähle ein Produkt für die Prognose", 
        st.session_state.selected_products_in_data
    )
    
    # Prognosehorizont auswählen
    forecast_horizon = st.slider(
        "Wähle den Prognosehorizont (in Monaten)", 
        min_value=1, 
        max_value=24, 
        value=12,
        step=1
    )

    # Modelle auswählen
    available_models = [
        "ARIMA", "Prophet", "LSTM", "GRU", "SES", 
        "SARIMA", "Holt-Winters", "XGBoost", 
        "Random Forest", "Moving Average", "Seasonal Naive", "Ensemble"
    ]
    selected_models = st.multiselect(
        "Wähle die Modelle für die Prognose", 
        available_models, 
        default=["ARIMA", "Prophet"]
    )

    # Get model parameters
    model_params = get_model_params(selected_models)

    if st.button("Prognose durchführen"):
        if st.session_state.df is not None and st.session_state.date_column is not None:
            try:
                # Daten vorbereiten
                df = st.session_state.df[[st.session_state.date_column, selected_product]].dropna()
                df[st.session_state.date_column] = pd.to_datetime(df[st.session_state.date_column], format="%d.%m.%Y", errors="coerce")
                df.set_index(st.session_state.date_column, inplace=True)
                df = df.sort_index()
                
                # Resample auf monatliche Frequenz
                df = df.resample('M').sum()

                # Prepare data for modeling
                model_data = df[selected_product]

                n = len(df)
                p = forecast_horizon

                if p >= n:
                    st.error("Prognosehorizont ist zu groß für die vorhandenen Daten.")
                    return

                # Split in Training und Test
                train = model_data.iloc[:-p]
                test = model_data.iloc[-p:]

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
                    "Seasonal Naive": SeasonalNaiveModel,
                }

                forecasts = {}
                metrics = {}
                confidence_intervals = {}

                with st.spinner('Berechne Prognosen...'):
                    progress_bar = st.progress(0)
                    for idx, model_name in enumerate(selected_models):
                        if model_name == "Ensemble" and len(selected_models) > 1:
                            selected_model_objects = []
                            total_train_time = 0
                            total_memory_used = 0
                            for m in selected_models:
                                if m != "Ensemble":
                                    trained_model = train_model(models_dict[m], train, **(model_params.get(m, {})))
                                    selected_model_objects.append(trained_model)
                                    total_train_time += trained_model.train_time
                                    total_memory_used += trained_model.memory_used
                            model = EnsembleModel(selected_model_objects)
                            model.train_time = total_train_time
                            model.memory_used = total_memory_used
                        else:
                            model = train_model(
                                models_dict[model_name],
                                train,
                                **(model_params.get(model_name, {}))
                            )
                            
                        try:
                            # Generate forecast
                            forecast = model.predict(p)
                            forecasts[model_name] = forecast
                            
                            # Get confidence intervals if available
                            if hasattr(model, 'predict_interval'):
                                lower, upper = model.predict_interval(p)
                                confidence_intervals[model_name] = (lower, upper)
                            
                            # Calculate metrics with performance data
                            metrics[model_name] = calculate_metrics(
                                test, forecast,
                                execution_time=model.train_time,
                                memory_used=model.memory_used
                            )
                            
                        except Exception as e:
                            st.warning(f"Fehler bei Modell {model_name}: {str(e)}")
                            continue
                            
                        progress_bar.progress((idx + 1) / len(selected_models))

                if forecasts:
                    # Visualization
                    fig = go.Figure()

                    # Plot training data
                    fig.add_trace(go.Scatter(
                        x=train.index, 
                        y=train,
                        mode='lines',
                        name='Training',
                        line=dict(color='blue')
                    ))

                    # Plot test data
                    fig.add_trace(go.Scatter(
                        x=test.index,
                        y=test,
                        mode='lines',
                        name='Test',
                        line=dict(color='green')
                    ))

                    # Plot forecasts and confidence intervals
                    colors = px.colors.qualitative.Set3
                    for i, (name, forecast) in enumerate(forecasts.items()):
                        color = colors[i % len(colors)]
                        
                        # Plot forecast
                        fig.add_trace(go.Scatter(
                            x=forecast.index,
                            y=forecast,
                            mode='lines',
                            name=f'Prognose ({name})',
                            line=dict(color=color)
                        ))
                        
                        # Plot confidence intervals if available
                        if name in confidence_intervals:
                            lower, upper = confidence_intervals[name]
                            fig.add_trace(go.Scatter(
                                x=forecast.index,
                                y=upper,
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=forecast.index,
                                y=lower,
                                mode='lines',
                                fill='tonexty',
                                name=f'KI ({name})',
                                line=dict(width=0),
                                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}'
                            ))

                    fig.update_layout(
                        title=f"Prognose für {selected_product}",
                        xaxis_title="Datum",
                        yaxis_title="Wert",
                        height=600
                    )

                    st.plotly_chart(fig)

                    # Display and download metrics
                    if metrics:
                        st.subheader("Metriken")
                        metrics_df = pd.DataFrame(metrics).T
                        metrics_df_styled_easy = format_summary_table(metrics_df, metrics_df.columns[1:8].tolist(), decimal_places=3)
                        st.session_state.metrics_df_styled_easy = metrics_df_styled_easy
                        st.dataframe(metrics_df_styled_easy)
                        
                        # Download buttons
                        col1, col2 = st.columns(2) 
                        with col1:
                            # Download forecasts
                            forecasts_df = pd.DataFrame(forecasts)
                            csv_forecasts = forecasts_df.to_csv()
                            st.download_button(
                                "Download Prognosen (CSV)",
                                csv_forecasts,
                                "forecasts.csv",
                                "text/csv",
                                key='download-forecasts'
                            )
                        
                        with col2:
                            # Download metrics
                            csv_metrics = metrics_df.to_csv()
                            st.download_button(
                                "Download Metriken (CSV)",
                                csv_metrics,
                                "metrics.csv",
                                "text/csv",
                                key='download-metrics'
                            )

                    # Add performance visualization
                    if metrics:
                        st.subheader("Performance Vergleich")
                        
                        # Create performance comparison plot
                        perf_fig = go.Figure()
                        
                        # Extract metrics for visualization
                        model_names = list(metrics.keys())
                        smape_values = [m['sMAPE'] for m in metrics.values()]
                        time_values = [m['Training_Time_s'] for m in metrics.values()]
                        memory_values = [m['Memory_Usage_MB'] for m in metrics.values()]
                        
                        # Create bubble chart
                        perf_fig.add_trace(go.Scatter(
                            x=time_values,
                            y=smape_values,
                            mode='markers',
                            marker=dict(
                                size=[m/5 for m in memory_values],  # Scale bubble size
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

            except Exception as e:
                st.error(f"Ein Fehler ist aufgetreten: {str(e)}")
                st.exception(e)
        else:
            st.error("Bitte laden Sie zuerst eine CSV-Datei hoch und wählen Sie die Datumsspalte sowie die Produkte aus.")