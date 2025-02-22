# tabs/aggregation.py
from setup_module.helpers import *
import plotly.express as px

def display_tab():
    # Collapsible text section
    with st.expander("Informationen zu diesem Tab"):
        st.write("""
            In diesem Tab können Sie aggregierte Daten analysieren und visualisieren:
            
            **Wie funktioniert der Prozess in diesem Tab?**
            
            1. **Datumsfilterung**: Wählen Sie einen Zeitraum aus, um die Daten zu filtern und nur relevante Daten anzuzeigen.
            2. **Produktwahl**: Wählen Sie die Produkte aus, die Sie analysieren möchten, um gezielte Einblicke zu erhalten.
            3. **Aggregationslevel**: Bestimmen Sie das Aggregationslevel (z.B. monatlich, quartalsweise, jährlich), um die Daten in verschiedenen Zeitintervallen zu betrachten.
            4. **Visualisierung**: Es wird ein Balkendiagramm erstellt, das die aggregierten Daten der ausgewählten Produkte für den Zeitraum zeigt.
            
            **Warum ist das nützlich?**
            
            Die Aggregation in diesem Tab ist nützlich, um die Daten auf einer übersichtlicheren und 
            interpretierbareren Ebene darzustellen. Anstatt alle Rohdaten einzeln zu betrachten, werden die 
            Informationen nach bestimmten Zeitintervallen (Aggregationslevel) zusammengefasst. Dies bietet mehrere Vorteile:
            - **Trends erkennen**:  Durch die Aggregation können längerfristige Trends und Muster in den Daten besser 
            sichtbar gemacht werden. Beispielsweise lassen sich saisonale Schwankungen oder Wachstumsraten einfacher 
            erkennen, wenn die Daten über Monate oder Jahre aggregiert werden.
            - **Reduzierte Komplexität**: Wenn sehr viele Einzelpunkte vorliegen, kann die Analyse der Daten schnell 
            unübersichtlich werden. Die Aggregation hilft, die Datenmenge zu reduzieren und die Übersichtlichkeit 
            zu erhöhen, ohne wichtige Informationen zu verlieren.
            - **Vergleiche ermöglichen**: Aggregierte Daten bieten eine gute Basis für den Vergleich von Zeitperioden, 
            zum Beispiel zwischen einzelnen Wochen oder Monaten. Das ermöglicht dem Benutzer, Entwicklungen zu 
            verfolgen oder Veränderungen im Laufe der Zeit besser zu verstehen.
            - **Glättung von Schwankungen**: Durch Aggregation können kurzfristige, möglicherweise zufällige 
            Schwankungen in den Daten ausgeglichen werden. Dies erleichtert die Erkennung von allgemeinen Tendenzen.
        """)
    
    # Use centralized date filter with unique prefix
    start_date, end_date = create_date_filter("aggregation")

    with st.container(border=True):
        # Daten filtern
        filtered_df = filter_df(start_date, end_date)

        # Aggregationslevel wählen
        aggregation_level = aggregation_select_box(2)
        freq = get_resampling_frequency(aggregation_level)

        # Produkte auswählen
        selected_products_agg = st.multiselect(
            "Wähle Produkte für das Stapeldiagramm:", 
            st.session_state.selected_products_in_data, 
            default=[st.session_state.selected_products_in_data[0]]
        )

        if selected_products_agg:
            # Summiere die Produkte basierend auf dem Aggregationslevel
            aggregated_df = filtered_df[selected_products_agg].resample(freq).sum()

            # Berechne die Gesamtsummen der gestapelten Balken pro Datum
            total_sums = aggregated_df.sum(axis=1)

            # Balkendiagramm erstellen
            bar_fig = px.bar(
                aggregated_df.reset_index(),
                x=st.session_state.date_column,
                y=aggregated_df.columns,
                title=f"Summe der Produkte auf {aggregation_level}",
                labels={col: col for col in selected_products_agg},
                text_auto=True,
                barmode='stack'
            )

            # Füge die aufsummierten Werte als Text hinzu
            for i, date in enumerate(aggregated_df.index):
                total_value = total_sums[date]
                bar_fig.add_annotation(
                    x=date,
                    y=total_value,
                    text=str(total_value),
                    showarrow=False,
                    yshift=15,
                    textangle=-90
                )

            bar_fig.update_layout(
                yaxis_title='Produkte',
            )

            # Zeige das Balkendiagramm an
            st.plotly_chart(bar_fig)
        else:
            st.warning("Bitte wähle mindestens ein Produkt aus.")