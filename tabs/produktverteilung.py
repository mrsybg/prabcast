from setup_module.helpers import *
from setup_module.session_state import get_app_state
import plotly.express as px

def display_tab():
    """Tab mit strukturiertem Session State Management."""
    state = get_app_state()
    
    with st.expander("Informationen zu diesem Tab"):
        st.write("""
            
            In diesem Tab können Sie ein Kuchendiagramm erstellen, das die Verteilung der Produktverkäufe analysiert und visualisiert:
            
            **Wie funktioniert der Prozess in diesem Tab?**
            
            1. **Datumsfilterung**: Wählen Sie einen Zeitraum aus, um die Daten zu filtern und nur relevante Daten anzuzeigen.
            2. **Produktwahl**: Wählen Sie die Produkte aus, deren Verkaufsverteilung Sie analysieren möchten.
            3. **Visualisierung**: Es wird ein Kuchendiagramm erstellt, das die prozentuale Verteilung der ausgewählten Produkte zeigt.
            
            **Warum ist das nützlich?**
            
            Ein Kuchendiagramm eignet sich hervorragend, um die verkauften Artikel eines Unternehmens als 
            Anteile am Gesamtverkauf darzustellen. Es ermöglicht auf einen Blick, die Verteilung und den relativen 
            Anteil jedes Produkts zu erkennen. Durch die einfache Visualisierung können Benutzer schnell 
            Spitzenprodukte identifizieren und strategische Entscheidungen treffen. Kuchendiagramme sind besonders 
            nützlich, um komplexe Verkaufsdaten verständlich und übersichtlich zu präsentieren. Jeder Artikel wird 
            im Vergleich zum Ganzen gesehen, wodurch Ungleichgewichte oder ein starkes Übergewicht bestimmter 
            Produkte sofort auffallen.
            """)

    # Use centralized date filter with unique prefix
    start_date, end_date = create_date_filter("produktverteilung")

    # Filtern der Daten basierend auf dem ausgewählten Zeitraum
    filtered_df = filter_df(start_date, end_date)

    with st.container(border=True):


        selected_products_pie = st.multiselect("Wähle Produkte für die Produktverteilung:", state.data.selected_products,
                                               default = state.data.selected_products)

        if selected_products_pie:
            col4, col5 = st.columns(2)

            with col4:

                # Kuchendiagramm für Produktverkäufe
                sales_sums = filtered_df[selected_products_pie].sum()
                pie_fig = px.pie(
                    sales_sums,
                    names=sales_sums.index,
                    values=sales_sums.values,
                    title="Prozentuale Produktverkäufe"
                )

                st.plotly_chart(pie_fig)
            with col5:

                # Berechnung der Prozentwerte
                total_sum = sales_sums.sum()
                product_percentages = round(((sales_sums / total_sum) * 100),2)

                # Erstellen eines DataFrames für die Tabelle
                pie_chart_data = pd.DataFrame({
                    "Produkt": sales_sums.index,
                    "Prozentwert (%)": product_percentages.values,
                    "Absoluter Wert": sales_sums.values
                })

                #Sortierung nach Größe und Tabelle darstellen
                pie_chart_data = pie_chart_data.sort_values("Absoluter Wert", ascending=False)
                pie_chart_data = format_summary_table(pie_chart_data,
                                                     ["Prozentwert (%)", "Absoluter Wert"])
                st.write("")
                st.write("")
                st.dataframe(pie_chart_data)

        else:
            st.write("Bitte zuerst Produkte auswählen.")