# tabs/forecast.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from setup_module.helpers import *
from setup_module.session_state import get_app_state
from setup_module.evaluation import calculate_metrics
from setup_module.forecast_helpers import (
    train_forecast_model,
    get_model_params_ui,
    create_forecast_chart,
    calculate_forecast_metrics,
    display_forecast_metrics,
    prepare_forecast_data,
    split_train_test,
    get_available_models,
    create_model_selection_ui
)
# ✨ NEU: Model Registry statt direkter Imports
from setup_module.model_registry import get_model_registry
from app.models import EnsembleModel  # Nur Ensemble noch direkt

# ✨ NEUE UX: UI Components
from setup_module.design_system import UIComponents
from setup_module.ui_helpers import (
    display_model_selector_with_info,
    safe_execute,
    run_with_progress,
    export_dialog
)
# ✨ SMART DEFAULTS: Intelligente Empfehlungen
from setup_module.smart_defaults import SmartDefaults
# ✨ CONTEXTUAL HELP: Hilfe-System für Produktionsplaner
from setup_module.help_system import HelpSystem, show_smart_warning, show_parameter_help

UI = UIComponents()


def display_tab():
    """Tab mit verbesserter UX - Multi-Model-Vergleich."""
    state = get_app_state()
    
    # ✨ PROFESSIONELL: Kompakte Info-Box
    UI.info_message("""
    **Univariate Absatzprognose mit Modellvergleich:** Vergleichen Sie mehrere Prognosemodelle gleichzeitig 
    und bewerten Sie deren Genauigkeit anhand historischer Daten. Die Modelle werden auf einem Teil der Daten 
    trainiert und auf dem Rest getestet.
    """, collapsible=True)

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

    # Datum Filter für Analysen
    st.subheader("Datumszeitraum für Analysen")
    
    # Initialisiere date_range in session_state, falls noch nicht vorhanden
    if 'start_date' not in st.session_state or 'end_date' not in st.session_state:
        # Standardmäßig gesamten Zeitraum verwenden
        if state.data.df is not None and state.data.date_column is not None:
            try:
                temp_df = state.data.df
                if temp_df is not None:
                    temp_df = temp_df.copy()
                    temp_df[state.data.date_column] = pd.to_datetime(
                        temp_df[state.data.date_column], 
                        format="%d.%m.%Y", 
                        errors="coerce"
                    )
                    state.data.start_date = temp_df[state.data.date_column].min().date()
                    state.data.end_date = temp_df[state.data.date_column].max().date()
                else:
                    state.data.start_date = pd.Timestamp.now().date() - pd.DateOffset(years=3)
                    state.data.end_date = pd.Timestamp.now().date()
            except:
                state.data.start_date = pd.Timestamp.now().date() - pd.DateOffset(years=3)
                state.data.end_date = pd.Timestamp.now().date()
        else:
            state.data.start_date = pd.Timestamp.now().date() - pd.DateOffset(years=3)
            state.data.end_date = pd.Timestamp.now().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Startdatum", 
            value=state.data.start_date,
            key="forecast_start_date"
        )
    with col2:
        end_date = st.date_input(
            "Enddatum", 
            value=state.data.end_date,
            key="forecast_end_date"
        )
    
    # Validierung und Speichern der Daten
    if start_date <= end_date:
        state.data.start_date = start_date
        state.data.end_date = end_date
        date_filter_active = True
    else:
        st.error("Fehler: Enddatum muss nach Startdatum liegen.")
        date_filter_active = False
    
    st.info("""
        Der gewählte Datumsbereich wird für beide Prognosetypen (univariat und multivariat) angewendet. 
        Im Tab 'Multivariate Absatzprognose' kann dieser Bereich bei Bedarf weiter angepasst werden.
    """)

    st.header("Univariate Absatzprognose")


    # Produkt auswählen
    selected_product = st.selectbox(
        "Wähle ein Produkt für die Prognose", 
        state.data.selected_products
    )
    
    # Produktdaten für Smart Defaults vorbereiten
    df = state.data.df.copy()
    df[state.data.date_column] = pd.to_datetime(df[state.data.date_column])
    df.set_index(state.data.date_column, inplace=True)
    product_data = df[selected_product].resample('M').sum()
    
    # ✨ SMART DEFAULTS: Intelligente Analyse
    recommendations = SmartDefaults.get_smart_recommendations(product_data)
    
    # ✨ CONTEXTUAL HELP: Datenqualität prüfen
    quality_warnings = HelpSystem.check_data_quality(product_data)
    if quality_warnings:
        with st.expander("⚠️ Hinweise zur Datenqualität", expanded=True):
            for warning in quality_warnings:
                st.markdown(warning)
    
    # Prognosehorizont mit Smart Default + Help
    col_horizon, col_smart, col_help = st.columns([4, 1, 1])
    with col_horizon:
        forecast_horizon = st.slider(
            "Wähle den Prognosehorizont (in Monaten)", 
            min_value=1, 
            max_value=recommendations['horizon']['max'], 
            value=recommendations['horizon']['recommended'],
            step=1,
            help=recommendations['horizon']['reason']
        )
    with col_smart:
        st.markdown("<div style='padding-top: 28px;'>", unsafe_allow_html=True)
        if st.button("🎯 Smart-Tipps", key='show_forecast_recommendations'):
            st.session_state['show_forecast_recs'] = not st.session_state.get('show_forecast_recs', False)
    with col_help:
        st.markdown("<div style='padding-top: 28px;'>", unsafe_allow_html=True)
        if st.button("📖", key='btn_horizon_help_multi', help="Was ist ein Prognosehorizont?"):
            st.session_state['popup_horizon_help_multi'] = not st.session_state.get('popup_horizon_help_multi', False)
    
    # ✨ HELP: Prognosehorizont Erklärung
    if st.session_state.get('popup_horizon_help_multi', False):
        HelpSystem.show_glossar_popup('Prognosehorizont', use_simple=False)
    
    # ✨ CONTEXTUAL WARNING: Prognosehorizont zu lang?
    horizon_warning = HelpSystem.check_forecast_horizon(len(product_data), forecast_horizon)
    if horizon_warning:
        show_smart_warning(horizon_warning, warning_type="warning")
    
    # Smart Recommendations anzeigen
    if st.session_state.get('show_forecast_recs', False):
        with st.expander("Intelligente Datenanalyse", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                season = recommendations['seasonality']
                if season['has_seasonality']:
                    st.success(f"**Saisonalität erkannt**")
                    st.caption(season['reason'])
                else:
                    st.info(f"**Keine Saisonalität**")
                    st.caption(season['reason'])
                
                with st.popover("📖 Was ist Saisonalität?"):
                    entry = HelpSystem.GLOSSAR['Saisonalität']
                    st.caption(entry['full_name'])
                    st.markdown(entry['detailed'])
            
            with col2:
                st.success(f"**Empfohlener Horizont: {recommendations['horizon']['recommended']} Monate**")
                st.caption(recommendations['horizon']['reason'])
            
            st.markdown("---")
            st.markdown("**Empfohlene Modelle für Ihre Daten:**")
            
            models_rec = recommendations['models']
            for model in models_rec['primary']:
                if model in models_rec['reasons']:
                    col_m1, col_m2 = st.columns([6, 1])
                    with col_m1:
                        st.markdown(f"**{model}**: {models_rec['reasons'][model]}")
                    with col_m2:
                        if model in HelpSystem.GLOSSAR:
                            with st.popover("📖"):
                                entry = HelpSystem.GLOSSAR[model]
                                st.caption(entry['full_name'])
                                st.markdown(entry['detailed'])

    # ✨ NEU: Modellauswahl mit intelligenter UI und Smart-Empfehlungen
    if st.button("❓ Wie wähle ich Modelle?", key='model_selection_help_multi', help="Hilfe zur Modellauswahl"):
        st.session_state['show_model_help_multi'] = not st.session_state.get('show_model_help_multi', False)
    
    if st.session_state.get('show_model_help_multi', False):
        with st.expander("💡 Hilfe zur Modellauswahl", expanded=True):
            st.markdown(HelpSystem.explain_parameter('model_selection'))
    
    selected_models, _ = display_model_selector_with_info(
        key_prefix="forecast_main_selector",
        multi_select=True,
        default_models=recommendations['models']['primary'][:2],  # Top 2 Empfehlungen
        data_length=len(product_data),
        has_seasonality=recommendations['seasonality']['has_seasonality']
    )
    
    # ✨ CONTEXTUAL WARNING: Modell-Eignung für alle gewählten Modelle
    for model in selected_models:
        model_warning = HelpSystem.check_model_suitability(
            model,
            len(product_data),
            recommendations['seasonality']['has_seasonality']
        )
        if model_warning:
            show_smart_warning(model_warning, warning_type="info")

    # Get model parameters mit zentraler UI
    model_params = get_model_params_ui(selected_models)

    if UI.primary_button("Prognose durchführen", key="forecast_main_btn", help="Startet die Berechnung mit den ausgewählten Modellen"):
        if state.data.df is not None and state.data.date_column is not None:
            try:
                # Daten vorbereiten
                df = state.data.df[[state.data.date_column, selected_product]].dropna()
                df[state.data.date_column] = pd.to_datetime(df[state.data.date_column], format="%d.%m.%Y", errors="coerce")
                
                # Auf gewählten Datumsbereich beschränken, falls aktiviert
                if date_filter_active:
                    mask = (df[state.data.date_column].dt.date >= state.data.start_date) & \
                           (df[state.data.date_column].dt.date <= state.data.end_date)
                    df = df[mask]
                
                df.set_index(state.data.date_column, inplace=True)
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

                # ✨ NEU: Hole Registry statt hardcoded dict
                registry = get_model_registry()

                forecasts = {}
                metrics = {}
                confidence_intervals = {}

                # ✨ NEU: Training mit Progress-Anzeige
                def train_all_models():
                    """Trainiert alle ausgewählten Modelle mit Progress-Tracking."""
                    tasks = []
                    task_names = []
                    for model_name in selected_models:
                        tasks.append(lambda m=model_name: train_single_model(m))
                        task_names.append(f'Trainiere {model_name}')
                    return run_with_progress(tasks, task_names)
                
                def train_single_model(model_name):
                    """Trainiert ein einzelnes Modell."""
                    if model_name == "Ensemble" and len(selected_models) > 1:
                        selected_model_objects = []
                        total_train_time = 0
                        total_memory_used = 0
                        for m in selected_models:
                            if m != "Ensemble":
                                # ✨ NEU: Hole aus Registry
                                model_class = registry.get(m)
                                trained_model = train_forecast_model(model_class, train, **(model_params.get(m, {})))
                                selected_model_objects.append(trained_model)
                                total_train_time += trained_model.train_time
                                total_memory_used += trained_model.memory_used
                        model = EnsembleModel(selected_model_objects)
                        model.train_time = total_train_time
                        model.memory_used = total_memory_used
                    else:
                        # ✨ NEU: Hole aus Registry
                        model_class = registry.get(model_name)
                        model = train_forecast_model(
                            model_class,
                            train,
                            **(model_params.get(model_name, {}))
                        )
                    
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
                    
                    return model_name
                
                # Führe Training aus mit Error-Handling
                success, results = safe_execute(
                    train_all_models,
                    error_message="Fehler beim Trainieren der Modelle",
                    success_message=f"{len(selected_models)} Modelle erfolgreich trainiert!"
                )

                if success and forecasts:
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
                        UI.section_header("Metriken", help_text="Vergleich der Modellperformance")
                        metrics_df = pd.DataFrame(metrics).T
                        metrics_df_styled_easy = format_summary_table(metrics_df, metrics_df.columns[1:8].tolist(), decimal_places=3)
                        st.session_state.metrics_df_styled_easy = metrics_df_styled_easy
                        st.dataframe(metrics_df_styled_easy)
                        
                        # ✨ NEU: Kombinierter Export mit Tabs
                        with st.expander("Ergebnisse exportieren"):
                            export_tab1, export_tab2 = st.tabs(["📊 Prognosen", "📈 Metriken"])
                            
                            with export_tab1:
                                forecasts_df = pd.DataFrame(forecasts)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    format_choice_prog = st.radio(
                                        "Export-Format",
                                        ["CSV", "Excel"],
                                        help="Wählen Sie das gewünschte Dateiformat",
                                        key="export_format_forecast_prognosen"
                                    )
                                with col2:
                                    add_timestamp_prog = st.checkbox(
                                        "Zeitstempel im Dateinamen",
                                        value=True,
                                        help="Fügt Datum und Uhrzeit zum Dateinamen hinzu",
                                        key="export_timestamp_forecast_prognosen"
                                    )
                                
                                timestamp_prog = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp_prog else ""
                                filename_prog = f"forecast_prognosen_{timestamp_prog}" if timestamp_prog else "forecast_prognosen"
                                
                                if format_choice_prog == "CSV":
                                    meta_str = "\n".join([
                                        f"# Produkt: {selected_product}",
                                        f"# Prognosehorizont: {forecast_horizon} Monate",
                                        f"# Modelle: {', '.join(selected_models)}",
                                        f"# Trainingszeitraum: {train.index[0]} bis {train.index[-1]}",
                                        f"# Prognosezeitraum: {test.index[0]} bis {test.index[-1]}"
                                    ])
                                    csv_data = meta_str + "\n" + forecasts_df.to_csv(index=True)
                                    st.download_button(
                                        label="📥 CSV herunterladen",
                                        data=csv_data,
                                        file_name=f"{filename_prog}.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                        key="download_csv_forecast_prognosen"
                                    )
                                else:
                                    from io import BytesIO
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        forecasts_df.to_excel(writer, sheet_name='Prognosen', index=True)
                                        meta_df = pd.DataFrame([
                                            ['Produkt', selected_product],
                                            ['Prognosehorizont', f"{forecast_horizon} Monate"],
                                            ['Modelle', ', '.join(selected_models)],
                                            ['Trainingszeitraum', f"{train.index[0]} bis {train.index[-1]}"],
                                            ['Prognosezeitraum', f"{test.index[0]} bis {test.index[-1]}"]
                                        ], columns=['Parameter', 'Wert'])
                                        meta_df.to_excel(writer, sheet_name='Metadaten', index=False)
                                    excel_data = output.getvalue()
                                    st.download_button(
                                        label="📥 Excel herunterladen",
                                        data=excel_data,
                                        file_name=f"{filename_prog}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True,
                                        key="download_excel_forecast_prognosen"
                                    )
                            
                            with export_tab2:
                                col1, col2 = st.columns(2)
                                with col1:
                                    format_choice_met = st.radio(
                                        "Export-Format",
                                        ["CSV", "Excel"],
                                        help="Wählen Sie das gewünschte Dateiformat",
                                        key="export_format_forecast_metriken"
                                    )
                                with col2:
                                    add_timestamp_met = st.checkbox(
                                        "Zeitstempel im Dateinamen",
                                        value=True,
                                        help="Fügt Datum und Uhrzeit zum Dateinamen hinzu",
                                        key="export_timestamp_forecast_metriken"
                                    )
                                
                                timestamp_met = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp_met else ""
                                filename_met = f"forecast_metriken_{timestamp_met}" if timestamp_met else "forecast_metriken"
                                
                                if format_choice_met == "CSV":
                                    meta_str = "\n".join([
                                        f"# Produkt: {selected_product}",
                                        f"# Modelle: {', '.join(selected_models)}"
                                    ])
                                    csv_data = meta_str + "\n" + metrics_df.to_csv(index=True)
                                    st.download_button(
                                        label="📥 CSV herunterladen",
                                        data=csv_data,
                                        file_name=f"{filename_met}.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                        key="download_csv_forecast_metriken"
                                    )
                                else:
                                    from io import BytesIO
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        metrics_df.to_excel(writer, sheet_name='Metriken', index=True)
                                        meta_df = pd.DataFrame([
                                            ['Produkt', selected_product],
                                            ['Modelle', ', '.join(selected_models)]
                                        ], columns=['Parameter', 'Wert'])
                                        meta_df.to_excel(writer, sheet_name='Metadaten', index=False)
                                    excel_data = output.getvalue()
                                    st.download_button(
                                        label="📥 Excel herunterladen",
                                        data=excel_data,
                                        file_name=f"{filename_met}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True,
                                        key="download_excel_forecast_metriken"
                                    )

                    # Add performance visualization
                    if metrics:
                        UI.section_header("Performance Vergleich", help_text="Visualisierung von Genauigkeit, Trainingszeit und Speicherverbrauch")
                        
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