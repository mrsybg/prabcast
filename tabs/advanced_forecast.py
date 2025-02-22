import streamlit as st
from setup_module.helpers import *
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

def display_tab():
    
    with st.expander("Informationen zu diesem Tab:"):
        st.write("""
            In diesem Tab können Sie eine **Datenanreicherung und Zusammenhangsanalyse** für ein ausgewähltes Produkt durchführen.
            Ziel ist es, externe Einflussfaktoren (Indizes) zu identifizieren, die mit den Verkaufszahlen korrelieren oder diese 
            sogar ursächlich beeinflussen.

            **Ablauf der Analyse:**
            1. **Produkt auswählen:** Wählen Sie das zu analysierende Produkt aus.
            2. **Fragebogen ausfüllen:** Bestimmen Sie, welche Arten von Indizes untersucht werden sollen:
            - Industrieindizes (z.B. Produktionsindizes): Wählen Sie hier die relevanten Branchen für Ihr Unternehmen.
            - Marktindizes (z.B. Aktienindizes): Wählen Sie die für Ihr Unternehmen relevanten Märkte aus. Anschließend 
            werden diese Märkte auf Aktienindizes, die den Absatz Ihres Produktes beeinflussen könnten, untersucht.
            - Regionale Indikatoren (z.B. BIP, Arbeitslosigkeit): Hier werden regionale Kennzahlen, wie z.B. die Inflation oder Arbeitslosigkeit untersucht.
            - Anzahl der wichtigsten Indizes: Die Anzahl der Indizes, die der Produktzeitreihe angehängt wird.
            - Anzahl der Zeitreihen pro Kategorie:
            3. **Analyse durchführen:** Das System:
            - Holt automatisch relevante Indexdaten
            - Berechnet **Pearson-Korrelationen**, um lineare Beziehungen zwischen Indizes und Absatzzahlen zu erkennen.
            - Führt eine **Granger-Causality-Analyse** durch, um mögliche kausale Zusammenhänge zu ermitteln, d.h. ob Veränderungen 
                in einem Index zeitlich vor Veränderungen im Absatz auftreten.
            - Bestimmt die **Feature Importance** ausgewählter Modelle, um herauszufinden, welche Indizes im Vergleich zu anderen 
                besonders einflussreich für die Vorhersage sind.
            - Identifiziert die stärksten Zusammenhänge und visualisiert sie in interaktiven Grafiken.
            4. **Ergebnisse verwerten:** 
            - Untersuchen Sie die gefundenen Zusammenhänge in interaktiven Visualisierungen.
            - Laden Sie die aufbereiteten Daten für weitere Analysen herunter.
            - Nutzen Sie die angereicherten Daten und identifizierten Einflussfaktoren als Grundlage für präzisere 
                multivariate Prognosen in anderen Tabs.

            Durch diese Analysen erhalten Sie tieferen Einblick in die Faktoren, die Ihre Verkaufszahlen beeinflussen, 
            und können so besser informierte Entscheidungen und genauere Zukunftsprognosen treffen.
        """)


    st.header("Datenanreicherung")

    product = st.selectbox(
        "Produkt für Analyse auswählen:",
        st.session_state.selected_products_in_data
    )

    # den ausgewählten Wert in den session_state speichern (für forecast_complex)
    st.session_state['selected_product'] = product

    if product:
        try:
            # Get and prepare product time series
            df = st.session_state.df.copy()
            df[st.session_state.date_column] = pd.to_datetime(df[st.session_state.date_column])
            df.set_index(st.session_state.date_column, inplace=True)

            # Get min and max dates from the dataset
            min_date = df.index.min().date()
            max_date = df.index.max().date()

            product_series = df[product].resample('M').sum()
            product_df = pd.DataFrame(product_series)

            if product_df.empty:
                st.error("No valid data found for selected product")
                return

            # Unpack new 9 values from questionnaire:
            (industries, locations, fmp_stocks, fmp_commodities, metal_prices,
             series_per_category, number_of_sources, start_date, end_date) = get_user_inputs(min_date, max_date)

            if not any([industries, locations, fmp_stocks, fmp_commodities, metal_prices]):
                st.warning("Bitte wählen Sie mindestens eine Kategorie oder geben Sie Ticker ein")
                return

            if st.button("Analyse starten", key="advanced_analysis_start_button"):
                with st.spinner("Hole und analysiere Daten..."):
                    # Filter data for selected date range
                    mask = (product_df.index.date >= start_date) & (product_df.index.date <= end_date)
                    product_df = product_df[mask]

                    indices_data, index_names, _ = fetch_indices_data(
                        industries=industries,
                        locations=locations,
                        date_range=product_df.index,
                        series_per_category=series_per_category,
                        stocks=fmp_stocks,          # Changed from fmp_stocks
                        commodities=fmp_commodities,  # Changed from fmp_commodities
                        metals=metal_prices,         # Changed from metal_prices
                        number_of_sources=number_of_sources
                    )

                    if indices_data.empty:
                        st.error("Keine gültigen Indexdaten gefunden")
                        return

                    # Ensure monthly data is on 'M' frequency and handle missing values
                    # Changed from mean() to last() to use the last value of each month
                    indices_data = indices_data.resample('M').last()
                    indices_data = indices_data.ffill().bfill()

                    common_dates = product_df.index.intersection(indices_data.index)
                    if len(common_dates) == 0:
                        st.error("Keine überlappenden Zeiträume zwischen Produkt und Indizes")
                        return

                    product_df = product_df.loc[common_dates]
                    indices_data = indices_data.loc[common_dates]

                    # Perform analysis
                    results = perform_analysis(product_df[product], indices_data)
                    
                    # Validate results
                    if (not isinstance(results, dict) or 
                        'data' not in results or 
                        not isinstance(results['data'], pd.DataFrame) or 
                        results['data'].empty):
                        st.error("Analysis produced no valid results")
                        return

                    results['index_names'] = index_names

                    # Display results
                    display_results(results, number_of_sources)

                    # Create downloadable DataFrame
                    if isinstance(results.get('data'), pd.DataFrame) and not results['data'].empty:
                        st.header("Daten herunterladen")
                        output = BytesIO()
                        results['data'].to_excel(output, index=True)
                        st.download_button(
                            label="Excel-Datei herunterladen",
                            data=output.getvalue(),
                            file_name="prognose_daten.xlsx",
                            mime="application/vnd.ms-excel"
                        )

                    if isinstance(results.get('data'), pd.DataFrame) and not results['data'].empty:
                        # Store results in session state for multivariate forecast
                        st.session_state.multivariate_data = results['data']
                        
                        # Display success message
                        st.success("Daten erfolgreich vorbereitet. Wechseln Sie zum 'Modellvergleich-Komplex' Tab für die multivariate Prognose.")

        except Exception as e:
            st.error(f"Ein Fehler ist aufgetreten: {str(e)}")
            st.exception(e)
    else:
        st.warning("Bitte wählen Sie zuerst ein Produkt aus.")
