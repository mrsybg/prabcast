from setup_module.helpers import *
from setup_module.session_state import get_app_state
from setup_module.error_handler import ErrorHandler, UserFeedback
from setup_module.exceptions import DataValidationError, ModelTrainingError
from setup_module.logging_config import log_data_operation, log_model_training
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL

def display_tab():
    """Tab mit strukturiertem Session State Management."""
    state = get_app_state()
    
    if not state.ready_for_processing:
        UserFeedback.warning("Bitte zuerst Daten hochladen und konfigurieren.")
        return
    
    with st.expander("Informationen zu diesem Tab"):
        st.write("""
            In diesem Tab haben Sie die Möglichkeit, eine sogenannte **STL-Zerlegung** auf die 
            Verkaufsdaten eines ausgewählten Produkts anzuwenden. Die Abkürzung **STL** steht 
            für *Seasonal-Trend decomposition using Loess*, ein statistisches Verfahren, das 
            in den 1990er Jahren von R. B. Cleveland, W. S. Cleveland, J. E. McRae und I. Terpenning 
            entwickelt wurde (https://www.wessa.net/download/stl.pdf). Es wurde konzipiert, um zeitlich geordnete Daten, sogenannte Zeitreihen, 
            in ihre wesentlichen Bestandteile zu zerlegen.

            **Was ist eine STL-Zerlegung?**
            Eine Zeitreihe besteht oft aus unterschiedlichen Komponenten:
            - **Trend**: Der langfristige Verlauf, also ob die Werte über einen längeren Zeitraum 
            insgesamt steigen, fallen oder relativ stabil bleiben.
            - **Saisonalität**: Regelmäßig wiederkehrende Muster oder Schwankungen, etwa saisonale 
            Spitzen im Sommer oder zum Jahresende.
            - **Residuen** (Reste): Die zufälligen Schwankungen oder Unregelmäßigkeiten, die nicht 
            durch Trend oder Saisonalität erklärt werden können.

            Das Besondere an STL ist, dass es sich einer flexiblen Glättungsmethode (LOESS) bedient, 
            um Trend und Saisonalität von der Zeitreihe zu trennen. Dadurch ist es robuster gegenüber 
            Ausreißern und eignet sich für eine Vielzahl von Anwendungsfällen besser als traditionelle 
            Zerlegungsverfahren. STL kommt aus der statistischen Analyse von Wirtschaftsdaten und wird 
            heute breit angewendet, von Verkaufs- und Nachfragedaten über Sensormessungen bis hin zu 
            Finanz- und Wetterdaten.

            **Wie funktioniert der Prozess in diesem Tab?**
            1. **Produktauswahl**: Wählen Sie ein Produkt, um dessen historische Verkaufsdaten 
            zu analysieren.
            2. **Datumsfilter**: Grenzen Sie den Zeitraum ein, um Daten für die relevante Periode 
            zu betrachten.
            3. **Resampling & Aggregation**: Die Daten werden, sofern nötig, in regelmäßigen Intervallen 
            (z. B. monatlich) zusammengefasst, um wiederkehrende Muster besser erkennen zu können.
            4. **STL-Zerlegung**: Die Zeitreihe wird mithilfe der STL-Methodik in Trend-, 
            Saisonalitäts- und Restkomponenten zerlegt.

            **Warum ist das nützlich?**
            - **Trend**: Erkennen Sie den längerfristigen Entwicklungspfad Ihrer Verkaufszahlen 
            und prüfen Sie, ob z. B. Maßnahmen zur Markterweiterung oder saisonunabhängige 
            Werbekampagnen Erfolge zeigen.
            - **Saisonalität**: Finden Sie regelmäßig wiederkehrende Muster, etwa saisonale Spitzen 
            im Weihnachtsgeschäft oder Sommerflauten, um Ihre Produktions- und Distributionsplanung 
            darauf abzustimmen.
            - **Residuen**: Ungewöhnliche Schwankungen, die nicht durch Trend oder Saisonalität 
            erklärbar sind, weisen auf besondere Ereignisse hin (z. B. einmalige Marketingaktionen, 
            Lieferengpässe oder außergewöhnliche Marktveränderungen).

            Durch die Zerlegung der Zeitreihe in diese Komponenten können Sie die Dynamik Ihrer 
            Verkaufsdaten besser verstehen und so fundierte Entscheidungen für zukünftige 
            Planungen und Strategien treffen.
        """)

    # Produktauswahl
    local_product = st.selectbox(
        "Wähle ein Produkt für die Zerlegung", 
        state.data.selected_products
    )

    # Datumsfilter mit eindeutigem Präfix
    start_date, end_date = create_date_filter("zerlegung")

    if local_product:
        try:
            fig = zerlegung(local_product, start_date, end_date, state.data.date_column)
            st.plotly_chart(fig)
        except Exception as e:
            UserFeedback.error(f"Fehler bei der Zerlegung: {str(e)}")


def zerlegung(selected_product_local, start_date_local, end_date_local, date_column):
    # Filtere den DataFrame
    filtered_df = filter_df(start_date_local, end_date_local)
    
    # Daten monatlich resamplen
    data = filtered_df[selected_product_local].resample('M').sum().reset_index()
    
    # STL-Zerlegung
    stl = STL(data[selected_product_local], period=12)
    result = stl.fit()

    # Visualisierung
    fig = go.Figure()

    # Originaldaten
    fig.add_trace(
        go.Scatter(
            x=data[date_column],
            y=data[selected_product_local],
            mode="lines",
            name="Original"
        )
    )

    # Trend
    fig.add_trace(
        go.Scatter(
            x=data[date_column],
            y=result.trend,
            mode="lines",
            name="Trend"
        )
    )

    # Saisonalität
    fig.add_trace(
        go.Scatter(
            x=data[date_column],
            y=result.seasonal,
            mode="lines",
            name="Saisonalität"
        )
    )

    # Residuen
    fig.add_trace(
        go.Scatter(
            x=data[date_column],
            y=result.resid,
            mode="lines",
            name="Residuen"
        )
    )

    fig.update_layout(
        title="STL-Zerlegung der Zeitreihe",
        xaxis_title="Datum",
        yaxis_title="Wert",
        height=800
    )

    return fig
