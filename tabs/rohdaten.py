# tabs/rohdaten.py
from setup_module.helpers import *
import plotly.express as px

def display_tab():

        # Collapsible text section
    with st.expander("Informationen zu diesem Tab"):
        st.write("""
            In diesem Tab können Sie den Gesamtabsatz aller Produkte (aus der Datenansicht) und die Rohdaten einzelner
             Produkte analysieren. Zu den Beiden Ansichten werden Kennwerte bereitgestellt, die unten erklärt werden.
             
             **Wie funktioniert der Prozess in diesem Tab?**
             
            1. **Datumsfilterung**: Wählen Sie einen Zeitraum aus, um die Daten zu filtern.
            2. **Gesamtabsatz**: Aggregierte Ansicht der Gesamtabsatzzahlen aller Produkte. Wählen Sie das gewünschte Aggregationslevel.
            3. **Einzelproduktanalyse**: Rohdaten der Absatzzahlen einzelner Produkte über das Auswahlmenü.
            4. **Kennwerttabelle**: Unter den Diagrammen finden Sie die Berechnung wichtiger Kennwerte für die aggregierten Daten und Einzelprodukte.
            
            **Erklärung der Kennwerte**: Alle Kennwerte beziehen sich auf das aktuelle Diagramm (ändert sich das Diagramm (z.B. durch Änderung der Aggregationsebene) ändern sich auch die Kennwerte)
            
            - **count**: Anzahl der Datenpunkte
            - **mean**: Arithmetisches Mittel 
            - **std**: Standardabweichung
            - **min**: Minimalmalwert
            - **25%**: Wert, unter dem 25% der Datenpunkte liegen
            - **50%**: Wert, unter dem 50% der Datenpunkte liegen 
            - **75%**: Wert, unter dem 75% der Datenpunkte liegen
            - **max**: Maximalwert
        """)
    
    # Einheitliche Datumsfilterung mit eindeutigem Präfix
    start_date, end_date = create_date_filter("rohdaten")
    
    # Filtern der Daten basierend auf dem ausgewählten Zeitraum
    filtered_df = filter_df(start_date, end_date)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            aggregation_level = aggregation_select_box(1)
            freq = get_resampling_frequency(aggregation_level)

            # Berechne die Gesamtsumme aller Produkte pro Zeiteinheit
            filtered_df['Gesamt_verkaufte_Einheiten'] = filtered_df[st.session_state.selected_products_in_data].sum(axis=1)

            # Aggregiere die Daten
            aggregated_df = filtered_df[['Gesamt_verkaufte_Einheiten']].resample(freq).sum()

            # Visualisierung
            bar_fig = px.bar(
                aggregated_df.reset_index(),
                x=st.session_state.date_column,
                y='Gesamt_verkaufte_Einheiten',
                title=f"Gesamtabsatz aller Produkte auf {aggregation_level}",
            )

            bar_fig.update_layout(
                yaxis_title='Abgesetzte Einheiten',
            )

            st.plotly_chart(bar_fig)

            # Kennwerttabelle
            summary_table = create_summary_table(aggregated_df)
            summary_table = format_summary_table(summary_table, ["count", "mean", "std", "min", "25%", "50%", "75%", "max"])

            st.write("Kennwerte für die aggregierte Gesamtansicht:")
            st.dataframe(summary_table, use_container_width=True)

    with col2:
        with st.container(border=True):
            # Einzelproduktanalyse
            selected_product = st.selectbox(
                "Wähle ein Produkt für das Balkendiagramm", 
                st.session_state.selected_products_in_data
            )
            
            # Visualisierung Einzelprodukt
            bar_fig = px.bar(
                filtered_df,
                y=selected_product,
                title=f"Rohdaten von {selected_product} im gewählten Zeitraum"
            )
            st.plotly_chart(bar_fig)

            # Kennwerttabelle Einzelprodukt
            product_summary_table = create_summary_table(
                filtered_df, 
                selected_products_in_data=st.session_state.selected_products_in_data
            )
            product_summary_table = format_summary_table(product_summary_table,
                                                 ["count", "mean", "std", "min", "25%", "50%", "75%", "max"])
            st.write("Kennwerte für die Produktansicht:")
            st.dataframe(product_summary_table, use_container_width=True)