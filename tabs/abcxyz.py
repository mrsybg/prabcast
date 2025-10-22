# tabs/abcxyz.py

import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from setup_module.helpers import *
from setup_module.session_state import get_app_state
from setup_module.error_handler import ErrorHandler, UserFeedback
from setup_module.exceptions import DataValidationError, ModelTrainingError
from setup_module.logging_config import log_data_operation, log_model_training


def plot_abc_xyz(data, aggregation_level, start_date, end_date,a_threshold,b_threshold, x_threshold, y_threshold, date_column, selected_products):
    """
    Führt eine ABC-XYZ Analyse basierend auf Aggregation und Beständigkeit durch.
    """
    # Überprüfe, ob die notwendigen Spalten vorhanden sind
    if not st.session_state.get("selected_products_in_data"):
        UserFeedback.error("Keine Produktspalten ausgewählt. Bitte wählen Sie Produkte in der Datenansicht aus.")
        return

    # Reshape die Daten von breit nach lang
    long_df = data.melt(
        id_vars=[date_column],
        value_vars=selected_products,
        var_name='Produkt',
        value_name='Gesamtverkauf'
    )

    # Überprüfe, ob die Spalte 'Produkt' existiert
    if 'Produkt' not in long_df.columns:
        UserFeedback.error("Die Spalte 'Produkt' fehlt nach dem Melting der Daten. Überprüfen Sie die Produktspalten.")
        return

    # Filtere den Zeitraum
    long_df[date_column] = pd.to_datetime(long_df[date_column])
    filtered_data = long_df[
        (long_df[date_column] >= pd.to_datetime(start_date)) & 
        (long_df[date_column] <= pd.to_datetime(end_date))
    ]

    # Aggregation setup - nutze zentrale Funktion
    freq = get_resampling_frequency(aggregation_level)

    # Anzahl der Perioden richtig berechnen
    total_periods = len(pd.date_range(start=start_date, end=end_date, freq=freq))

    # Beständigkeit berechnen: Anzahl der Perioden mit Verkäufen > 0
    try:
        sales_continuity = (
            filtered_data.set_index(date_column)
                       .groupby('Produkt')
                       .resample(freq)['Gesamtverkauf']
                       .apply(lambda x: (x > 0).sum())
                       .unstack(level=0)
                       .fillna(0)
        )
        sales_continuity = sales_continuity.sum()
    except Exception as e:
        UserFeedback.error(f"Fehler bei der Berechnung der Beständigkeit: {e}")
        return

    # Sortiere Produkte nach Gesamtverkauf für ABC
    total_sales = filtered_data.groupby('Produkt')['Gesamtverkauf'].sum()
    sorted_products = total_sales.sort_values(ascending=False)

    # ABC-Klassifikation
    sorted_sales_sum = sorted_products.sum()
    cumsum_percentage = (sorted_products.cumsum() / sorted_sales_sum) * 100
    abc_class = pd.Series('C', index=sorted_products.index)
    abc_class[cumsum_percentage <= a_threshold] = 'A'
    abc_class[cumsum_percentage.between(a_threshold, b_threshold, inclusive='left')] = 'B'

    # XYZ-Klassifikation basierend auf Beständigkeit
    continuity_percentage = (sales_continuity / total_periods) * 100
    # Korrigiere mögliche Überschreitungen von 100%
    continuity_percentage = continuity_percentage.clip(upper=100)

    xyz_class = pd.Series('Z', index=sorted_products.index)
    xyz_class[continuity_percentage >= x_threshold] = 'X'
    xyz_class[continuity_percentage.between(y_threshold, x_threshold, inclusive='left')] = 'Y'

    # Erstelle das Klassifikations-DataFrame
    classification = pd.DataFrame({
        'Produkt': sorted_products.index,
        'Gesamtverkauf': sorted_products.values,
        'Anteil_kumuliert': cumsum_percentage.round(2),
        'Beständigkeit (%)': continuity_percentage[sorted_products.index].round(2),
        'ABC': abc_class,
        'XYZ': xyz_class
    })

    classification_2 = classification.set_index('Produkt')
    classification_2 = format_summary_table(classification_2, ["Gesamtverkauf", "Anteil_kumuliert", "Beständigkeit (%)"])

    # Ergebnisse anzeigen
    st.subheader("ABC-XYZ Analyse Ergebnisse")
    st.dataframe(classification_2)

    # Erstelle die ABC-XYZ Matrix
    matrix = classification.pivot_table(
        index='ABC',
        columns='XYZ',
        values='Produkt',
        aggfunc=lambda x: ', '.join(x)
    ).fillna('')

    st.subheader("ABC-XYZ Matrix")
    st.table(matrix)

    # Pareto Analyse
    fig = go.Figure()

    # Balken für Gesamtverkauf
    fig.add_trace(go.Bar(
        x=sorted_products.index.tolist(),
        y=sorted_products.values,
        name='Gesamtverkauf',
        marker=dict(color='skyblue'),
        hovertemplate='Produkt %{x}<br>Gesamtverkauf: %{y}'
    ))

    # Linie für kumulierten Anteil
    fig.add_trace(go.Scatter(
        x=sorted_products.index.tolist(),
        y=cumsum_percentage,
        mode='lines+markers',
        name='Kumulierter Anteil',
        line=dict(color='orange', width=2),
        marker=dict(size=6),
        yaxis='y2',
        hovertemplate='Produkt %{x}<br>Kumuliert: %{y:.1f}%',
    ))

    # Layout anpassen mit zwei y-Achsen
    fig.update_layout(
        xaxis_title='Produkte (sortiert nach Verkaufsmenge)',
        yaxis=dict(
            title=dict(text='Gesamtverkauf', font=dict(color='skyblue')),
            tickfont=dict(color='skyblue'),
            position=0
        ),
        yaxis2=dict(
            title=dict(text='Kumulierter Anteil (%)', font=dict(color='orange')),
            tickfont=dict(color='orange'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.01, y=0.99),
        barmode='group',
        height=500
    )

    # Referenzlinien hinzufügen
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="A-B Grenze (80%)", annotation_position="bottom right")
    fig.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="B-C Grenze (95%)", annotation_position="bottom right")

    # Plot anzeigen
    st.subheader("Pareto Analyse")
    st.plotly_chart(fig, use_container_width=True)

def display_tab():
    """Tab mit strukturiertem Session State Management."""
    state = get_app_state()
    
    # Collapsible text section with detailed information
    with st.expander("Informationen zu diesem Tab"):
        st.write("""
            In diesem Tab können Sie eine **ABC-XYZ-Analyse** durchführen. Diese setzt sich aus der ABC-Analyse 
            (https://refa.de/service/refa-lexikon/abc-analyse) und der XYZ-Analyse 
            (https://refa.de/service/refa-lexikon/xyz-analyse) zusammen.

            Die **ABC-XYZ-Analyse** ist eine erweiterte Klassifizierungsmethode, die Produkte basierend auf zwei Dimensionen kategorisiert:

            - **ABC-Klassifizierung**: Teilt Produkte in Kategorien nach ihrer Relevanz:
                - **A**: Produkte mit dem höchsten Umsatzvolumen.
                - **B**: Produkte mit mittlerem Umsatzvolumen.
                - **C**: Produkte mit dem niedrigsten Umsatzvolumen.
                
            Durch die ABC-Analyse werden die Produkte in absteigender Reihenfolge des Zielwerts sortiert und basierend 
            auf ihrer kumulierten Leistung in die Kategorien A, B und C eingeteilt. So können Unternehmen ihre 
            Ressourcen auf die wichtigsten Produkte konzentrieren.

            - **XYZ-Klassifizierung**: Teilt Produkte basierend auf der Variabilität der Nachfrage:
                - **X**: Produkte mit geringer Nachfrageschwankung.
                - **Y**: Produkte mit mittlerer Nachfrageschwankung.
                - **Z**: Produkte mit hoher Nachfrageschwankung.
                
            Die XYZ-Analyse ist ein Verfahren, das Produkte nach der Vorhersagbarkeit und Stabilität ihrer Nachfrage 
            klassifiziert. Sie ergänzt oft die ABC-Analyse und wird verwendet, um das Bestandsmanagement und die 
            Produktionsplanung zu optimieren. Die Einteilung erfolgt in drei Kategorien. Die XYZ-Klassifikation wird hier
             basierend auf der Beständigkeit der Verkäufe durchgeführt. Eine andere Möglichkeit wäre die Analyse des Variationskoeffizienten.

            **Wie funktioniert der Prozess in diesem Tab?**

            **1. Parameter Auswahl**: Es werden alle Produkte aus der Datenansicht analysiert. Wählen Sie das gewünschte Aggregationslevel, sowie den zu analysierenden Zeitraum. Vor dem starten der Analyse können Sie zusätzlich die Schwellenwerte anpassen.

            **2. Datenfilterung und -bereinigung**: Die ausgewählten Daten werden gefiltert und bereinigt, um nur relevante und konsistente Daten für die Analyse zu verwenden.

            **3. Berechnung der Umsätze und Nachfragevariabilität**: Für jedes Produkt werden die Gesamtumsätze und die Standardabweichung der Nachfrage berechnet, um die Variabilität festzustellen.

            **4. Klassifizierung der Produkte**:
            
            - Basierend auf den berechneten Umsätzen werden die Produkte in A, B oder C Kategorien eingeteilt.
            - Anschließend wird die Nachfrageschwankung analysiert, um die Produkte in X, Y oder Z Kategorien zu platzieren.

            **5. Visualisierung der Ergebnisse**: Die Analyseergebnisse werden in Form von Pareto-Diagrammen und anderen Visualisierungen dargestellt, um klare Einblicke in die Produktkategorisierung zu ermöglichen.

            **6. Interpretation und Handlungsempfehlungen**: Anhand der Klassifizierung können strategische Entscheidungen getroffen werden, wie z.B. Lagerbestandsoptimierung, Marketingstrategien oder Fokus auf bestimmte Produktkategorien.

            **Vorteile der ABC-XYZ-Analyse:**

            - **Fokus auf wichtige Produkte**: Identifikation der Produkte, die den größten Einfluss auf den Umsatz haben.
            - **Effiziente Ressourcenallokation**: Optimierung von Lagerbeständen und Marketingaufwendungen basierend auf Produktkategorien.
            - **Verbesserte Nachfrageprognose**: Verständnis der Nachfrageschwankungen zur besseren Planung und Vorhersage.
        """)

    # Aggregationsbasis auswählen
    aggregation_level = st.selectbox(
        "Wähle die Aggregationsbasis für die Analyse:",
        options=["Wöchentlich", "Monatlich", "Quartalsweise"],
        index=1  # Standard auf "Monatlich"
    )

    # Zeitraum auswählen
    start_date, end_date = st.date_input(
        "Wähle den Zeitraum für die Analyse:",
        value=[
            state.data.df[state.data.date_column].min(),
            state.data.df[state.data.date_column].max()
        ],
        min_value=state.data.df[state.data.date_column].min(),
        max_value=state.data.df[state.data.date_column].max()
    )

    # Schwellenwerte für ABC-Klassifikation
    st.write("**Schwellenwerte für die XYZ-Klassifikation (in Prozent):**")
    with st.expander("Anleitung zur Auswahl der Schwellenwerte: (ABC)"):
        st.write("""
        - Wählen Sie die Prozentsätze, um die Klassifikation der Produkte basierend auf ihrem kumulierten Wert festzulegen.
        - Produkte in der **A-Kategorie** machen die wichtigsten Artikel aus (voreingestellter Standardwert: die obersten **80%** des Werts).
        - Produkte in der **B-Kategorie** decken den mittleren Wertbereich ab (voreingestellte Standardwerte von 80% bis **95%** des Werts).
        - Produkte in der **C-Kategorie** umfassen die restlichen Produkte mit dem geringsten Wert. (voreingestellte Standardwerte: **95%** bis 100%)
        - Achten Sie darauf, dass der Schwellenwert der B-Kategorie immer höher ist als der der A-Kategorie.
        """)
    a_threshold = st.number_input("A-Kategorie bis (%)", min_value=0, max_value=100, value=80, step=5)
    b_threshold = st.number_input("B-Kategorie bis (%)", min_value=a_threshold + 1, max_value=100, value=95, step=5)

    # Überprüfung, ob die Bedingung eingehalten wird
    if b_threshold <= a_threshold:
        UserFeedback.error("B-Kategorie muss höher sein als A-Kategorie.")


    # Schwellenwerte für XYZ-Klassifikation
    st.write("**Schwellenwerte für die XYZ-Klassifikation (in Prozent):**")
    with st.expander("Anleitung zur Auswahl der Schwellenwerte (XYZ):"):
        st.write("""
        - Wählen Sie die Prozentsätze, um die Klassifikation der Produkte festzulegen.
        - Produkte in der **X-Kategorie** sind die beständigsten Produkte (voreingestellter Standardwert: 100% bis **75%** Beständigkeit).
        - Produkte in der **Y-Kategorie** decken den mittleren Wertbereich ab (voreingestellter Standardwert: 75% bis **40%** Beständigkeit).
        - Produkte in der **Z-Kategorie** umfassen die restlichen Produkte mit einer Beständigkeit unter dem Wert der Y-Kategorie. (voreingestellter Standardwert: **40%** bis 0% Beständigkeit)
        - Achten Sie darauf, dass der Schwellenwert der Y-Kategorie immer niedriger ist als der der X-Kategorie.
        """)
    x_threshold = st.number_input("X-Kategorie ab (%)", min_value=0, max_value=100, value=75, step=5)
    y_threshold = st.number_input("Y-Kategorie ab (%)", min_value=0, max_value=x_threshold-1, value=40, step=5)

    # Überprüfung, ob die Bedingung eingehalten wird
    if y_threshold >= x_threshold:
        UserFeedback.error("X-Kategorie muss höher sein als Y-Kategorie.")

    # Analyse durchführen
    if st.button("Analyse starten"):
        plot_abc_xyz(
            state.data.df,
            aggregation_level,
            start_date,
            end_date,
            a_threshold,
            b_threshold,
            x_threshold,
            y_threshold,
            state.data.date_column,
            state.data.selected_products
        )