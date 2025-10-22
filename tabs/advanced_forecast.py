import streamlit as st
from setup_module.helpers import *
from setup_module.session_state import get_app_state
from setup_module.error_handler import ErrorHandler, UserFeedback
from setup_module.exceptions import (
    DataValidationError, APIConnectionError, APIDataError, 
    ModelTrainingError, DataLoadError
)
from setup_module.logging_config import log_api_call, log_data_operation, log_model_training
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objects as go
from .advanced.questionnaire import get_user_inputs
from .advanced.api_fetch import fetch_indices_data
from .advanced.analysis import perform_analysis
from .advanced.visualization import display_results
from setup_module.evaluation import calculate_metrics
from app.models_multi import build_lstm_model, build_xgboost_model
import pandas as pd
from io import BytesIO
import time  # Add missing import for time.sleep()
import os
# ✨ UX/UI Components
from setup_module.design_system import UIComponents
from setup_module.ui_helpers import (
    safe_execute,
    export_dialog
)

UI = UIComponents()

def display_tab():
    """Tab mit strukturiertem Session State Management."""
    state = get_app_state()
    
    # ✨ PROFESSIONELL: Kompakte Info-Box
    UI.info_message("""
    **Datenanreicherung & Zusammenhangsanalyse:** Identifizieren Sie externe Einflussfaktoren (Indizes), 
    die mit Ihren Verkaufszahlen korrelieren. Das System analysiert automatisch Industrieindizes, Marktindizes 
    und regionale Indikatoren mittels Pearson-Korrelation, Granger-Causality und Feature Importance.
    """, collapsible=True)


    st.header("Datenanreicherung")

    product = st.selectbox(
        "Produkt für Analyse auswählen:",
        state.data.selected_products
    )

    # den ausgewählten Wert in den session_state speichern (für forecast_complex)
    st.session_state['selected_product'] = product

    if product:
        try:
            # Get and prepare product time series
            df = state.data.df
            if df is None:
                UserFeedback.error("Keine Daten geladen")
                return
            df = df.copy()
            df[state.data.date_column] = pd.to_datetime(df[state.data.date_column])
            df.set_index(state.data.date_column, inplace=True)

            # Get min and max dates from the dataset
            min_date = df.index.min().date()
            max_date = df.index.max().date()

            product_series = df[product].resample('M').sum()
            product_df = pd.DataFrame(product_series)

            if product_df.empty:
                UserFeedback.error(
                    "Keine gültigen Daten für ausgewähltes Produkt gefunden",
                    help_text="Überprüfen Sie, ob das Produkt Verkaufsdaten enthält."
                )
                return
                return

            # Neue Sektion für eigene CSV-Daten
            st.subheader("Eigene Daten hinzufügen (optional)")
            
            with st.expander("Eigene Daten hinzufügen"):
                    st.write("""
                        Wenn Sie eigene interne Daten nutzen möchten, wie beispielsweise **Marketingdaten**, **CRM-Daten** oder andere betriebliche Informationen,
                        können Sie diese hier hochladen. Die CSV-Datei muss folgendes Format haben:
                        
                        - **Erste Spalte:** Datum im Format YYYY-MM-DD  
                        - **Weitere Spalten:** Interne Attribute/Features (z.B. Marketingausgaben, Leads, Verkaufszahlen)
                        
                        Beispiel:
                        ```
                        Datum,Marketingausgaben,CRM_Leads
                        2020-01-01,10000,250
                        2020-02-01,12000,300
                        ```
                        
                        Die Daten werden auf Monatsebene aggregiert und mit den anderen Indexdaten verknüpft.
                    """)
                
            # Erstelle eine Beispiel-CSV-Datei zum Herunterladen
            example_csv = """Datum,Umsatz_Marketing,Mitarbeiter_Anzahl,Kundenzufriedenheit,Marktanteil
2020-01-01,10000,250,4.2,0.15
2020-02-01,12000,255,4.3,0.16
2020-03-01,15000,260,4.1,0.16
2020-04-01,13000,265,4.0,0.17
2020-05-01,14000,270,4.2,0.17
2020-06-01,16000,275,4.4,0.18"""
                
            st.download_button(
                label="Beispiel-CSV herunterladen",
                data=example_csv,
                file_name="custom_data_example.csv",
                mime="text/csv",
            )
                
            uploaded_file = st.file_uploader("CSV-Datei mit eigenen Daten hochladen", type=["csv"])
            custom_data = None
                
            if uploaded_file is not None:
                try:
                    # Lade die CSV-Datei
                    custom_data = pd.read_csv(uploaded_file)
                        
                    # Zeige eine Vorschau der Daten
                    st.write("Vorschau der hochgeladenen Daten:")
                    st.write(custom_data.head())
                        
                    # Überprüfe, ob die erste Spalte ein Datum ist
                    date_col = custom_data.columns[0]
                    try:
                        custom_data[date_col] = pd.to_datetime(custom_data[date_col])
                        custom_data.set_index(date_col, inplace=True)
                            
                        # Auf Monatsende umstellen
                        custom_data = custom_data.resample('M').last()
                            
                        # Prüfen, ob der Datenzeitraum mit dem Produktzeitraum übereinstimmt
                        overlap_start = max(custom_data.index.min(), product_df.index.min())
                        overlap_end = min(custom_data.index.max(), product_df.index.max())
                            
                        if overlap_start > overlap_end:
                            UserFeedback.error("Die hochgeladenen Daten haben keine zeitliche Überlappung mit den Produktdaten.")
                            custom_data = None
                        else:
                            UserFeedback.success(f"Daten erfolgreich geladen. Zeitraum: {custom_data.index.min().date()} bis {custom_data.index.max().date()}")
                                
                            # Stellt sicher, dass wir keine doppelten Spaltennamen haben
                            rename_dict = {}
                            for col in custom_data.columns:
                                rename_dict[col] = f"CUSTOM_{col}"
                                
                            custom_data = custom_data.rename(columns=rename_dict)
                                
                            # Zeige die umbenannten Spalten an
                            st.write("Spalten für die Analyse:", custom_data.columns.tolist())
                                
                    except Exception as e:
                        UserFeedback.error(f"Die erste Spalte muss ein gültiges Datum enthalten: {str(e)}")
                        custom_data = None
                            
                except Exception as e:
                    UserFeedback.error(f"Fehler beim Laden der Datei: {str(e)}")
                    custom_data = None

            # Unpack new 9 values from questionnaire:
            (industries, locations, fmp_stocks, fmp_commodities, metal_prices,
             series_per_category, number_of_sources, start_date, end_date) = get_user_inputs(min_date, max_date)

            if not any([industries, locations, fmp_stocks, fmp_commodities, metal_prices]) and custom_data is None:
                UserFeedback.warning("Bitte wählen Sie mindestens eine Kategorie oder geben Sie Ticker ein oder laden Sie eigene Daten hoch")
                return

            # ✨ NEU: Professioneller Button
            if UI.primary_button("Analyse starten", key="advanced_analysis_btn", help="Startet Datenanreicherung und Zusammenhangsanalyse"):
                # Create a placeholder for the progress bar
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Hide all log messages in a spinner
                with st.spinner(""):
                    # Update progress
                    status_text.text("Hole und analysiere Daten... (10%)")
                    progress_bar.progress(10)
                    
                    # Filter data for selected date range
                    mask = (product_df.index.date >= start_date) & (product_df.index.date <= end_date)
                    product_df = product_df[mask]
                    
                    # Update progress
                    status_text.text("Lade externe Indexdaten... (30%)")
                    progress_bar.progress(30)

                    # Use log_capture=True to suppress logs to streamlit UI
                    indices_data, index_names, _ = fetch_indices_data(
                        industries=industries,
                        locations=locations,
                        date_range=product_df.index,
                        series_per_category=series_per_category,
                        stocks=fmp_stocks,
                        commodities=fmp_commodities,
                        metals=metal_prices,
                        number_of_sources=number_of_sources,
                        log_capture=True  # Add this parameter to suppress logs
                    )
                    
                    # Füge eigene Daten hinzu, wenn vorhanden
                    if custom_data is not None:
                        status_text.text("Füge eigene Daten hinzu... (40%)")
                        progress_bar.progress(40)
                        
                        # Filtere auf den gleichen Zeitraum
                        try:
                            mask = (custom_data.index.date >= start_date) & (custom_data.index.date <= end_date)
                            filtered_custom_data = custom_data[mask]
                            
                            if not filtered_custom_data.empty:
                                # Wenn indices_data leer ist, erstelle ein DataFrame mit dem richtigen Index
                                if indices_data.empty:
                                    indices_data = pd.DataFrame(index=product_df.index)
                                
                                # Für jede Spalte in den benutzerdefinierten Daten
                                for col in filtered_custom_data.columns:
                                    # Füge Spalte zum indices_data hinzu
                                    temp_series = filtered_custom_data[col].reindex(indices_data.index)
                                    indices_data[col] = temp_series
                                    
                                    # Füge Spaltennamen zu index_names hinzu
                                    index_names[col] = f"Custom: {col}"
                                
                                UserFeedback.success(f"{len(filtered_custom_data.columns)} eigene Datenreihen hinzugefügt")
                            else:
                                UserFeedback.warning("Keine eigenen Daten im gewählten Zeitraum gefunden")
                        except Exception as e:
                            UserFeedback.error(f"Fehler beim Verarbeiten der eigenen Daten: {str(e)}")

                    if indices_data.empty:
                        progress_placeholder.empty()
                        UserFeedback.error("Keine gültigen Indexdaten gefunden")
                        return

                    # Update progress
                    status_text.text("Verarbeite Daten... (50%)")
                    progress_bar.progress(50)
                    
                    # Debug information in an expander so it's hidden by default
                    with st.expander("Debug: Datenfrequenz-Info", expanded=False):
                        st.write("Prüfe Datenfrequenz...")
                        # Check the frequency of indices data
                        st.write(f"Shape der Indexdaten: {indices_data.shape}")
                        if not indices_data.empty:
                            # Check the frequency of index data
                            date_diffs = indices_data.index.to_series().diff().value_counts()
                            st.write("Zeitliche Abstände zwischen den Daten (Tage):")
                            st.write(date_diffs)
                            # Sample of the data
                            st.write("Stichprobe der Indexdaten (erste 10 Zeilen):")
                            st.write(indices_data.head(10))
                            # Fix for the 'nunique' error - use numpy's unique function instead
                            sample_col = indices_data.columns[0] if len(indices_data.columns) > 0 else None
                            if sample_col:
                                def check_repeated_values(window):
                                    # Use numpy's unique function which works with ndarray
                                    return len(np.unique(window)) == 1
                                
                                # Use apply with a function that counts unique values
                                repeated_values = []
                                for i in range(2, len(indices_data)):
                                    if i >= 3:
                                        window = indices_data[sample_col].iloc[i-3:i].values
                                        if len(np.unique(window)) == 1:
                                            repeated_values.append(i)
                                st.write(f"Anzahl der Stellen mit 3 identischen Werten in Folge: {len(repeated_values)}")
                    
                    # Update progress
                    status_text.text("Interpoliere Datenwerte... (60%)")
                    progress_bar.progress(60)
                    
                    # Apply more aggressive interpolation to quarterly patterns
                    is_quarterly = False
                    # Look for 3-month repeating patterns in all columns
                    quarterly_pattern_cols = []
                    for col in indices_data.columns:
                        # Use a rolling window to check if values repeat every 3 months
                        values = indices_data[col].values
                        is_quarterly = True
                        for i in range(0, len(values)-2, 3):
                            if i+2 < len(values):
                                if not (values[i] == values[i+1] == values[i+2]):
                                    is_quarterly = False
                                    break
                        if is_quarterly and len(values) >= 9:
                            quarterly_pattern_cols.append(col)
                    
                    # If we detected quarterly patterns
                    if quarterly_pattern_cols:
                        with st.expander("Hinweis zur Dateninterpolation", expanded=False):
                            st.warning(f"""
                                In {len(quarterly_pattern_cols)} von {indices_data.shape[1]} Zeitreihen wurde ein quartalsweises Muster erkannt.
                                Diese Daten werden interpoliert, um monatliche Werte zu erhalten.
                            """)
                        # Create a copy with interpolated values for quarterly columns
                        for col in quarterly_pattern_cols:
                            # Extract the quarterly values (every 3rd value)
                            quarterly_values = indices_data[col].iloc[::3]
                            # Create a temporary series with the quarterly values
                            temp_series = pd.Series(index=indices_data.index)
                            temp_series.iloc[::3] = quarterly_values
                            # Interpolate to fill in the missing months with cubic interpolation
                            indices_data[col] = temp_series.interpolate(method='cubic')
                    
                    # Apply interpolation to create unique monthly values
                    indices_data = indices_data.interpolate(method='cubic')

                    # Update progress
                    status_text.text("Führe Zusammenhangsanalyse durch... (80%)")
                    progress_bar.progress(80)
                    
                    common_dates = product_df.index.intersection(indices_data.index)
                    if len(common_dates) == 0:
                        progress_placeholder.empty()
                        UserFeedback.error("Keine überlappenden Zeiträume zwischen Produkt und Indizes")
                        return

                    product_df = product_df.loc[common_dates]
                    indices_data = indices_data.loc[common_dates]

                    # Perform analysis
                    results = perform_analysis(product_df[product], indices_data)
                    
                    # Update progress
                    status_text.text("Bereite Ergebnisdarstellung vor... (90%)")
                    progress_bar.progress(90)
                    
                    # Validate results
                    if (not isinstance(results, dict) or 
                        'data' not in results or 
                        not isinstance(results['data'], pd.DataFrame) or 
                        results['data'].empty):
                        progress_placeholder.empty()
                        UserFeedback.error("Analysis produced no valid results")
                        return

                    results['index_names'] = index_names

                    # Complete the progress bar
                    status_text.text("Analyse abgeschlossen! (100%)")
                    progress_bar.progress(100)

                    # Remove the progress indicators after completion
                    time.sleep(1)
                    progress_placeholder.empty()

                    # Display results
                    display_results(results, number_of_sources)

                    # Create downloadable DataFrame
                    if isinstance(results.get('data'), pd.DataFrame) and not results['data'].empty:
                        UI.section_header("Daten herunterladen", help_text="Exportieren Sie die angereicherten Daten für weitere Analysen")
                        export_dialog(
                            data=results['data'],
                            filename_base="angereicherte_daten",
                            metadata={
                                'Produkt': product,
                                'Zeitraum': f"{start_date} bis {end_date}",
                                'Anzahl Indizes': len(results['data'].columns) - 1
                            }
                        )
                    if isinstance(results.get('data'), pd.DataFrame) and not results['data'].empty:
                        # Store results in session state for multivariate forecast
                        state.forecast.multivariate_data = results['data']
                        # Display success message
                        UserFeedback.success("Daten erfolgreich vorbereitet. Wechseln Sie zum 'Modellvergleich-Komplex' Tab für die multivariate Prognose.")
        except Exception as e:
            UserFeedback.error(f"Ein Fehler ist aufgetreten: {str(e)}")
            st.exception(e)
    else:
        UserFeedback.warning("Bitte wählen Sie zuerst ein Produkt aus.")