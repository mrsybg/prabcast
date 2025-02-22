import streamlit as st
import os
from pathlib import Path

# Get root directory
ROOT_DIR = Path(__file__).parent.parent

# Direct imports
from tabs.upload import display_tab as upload_tab
from tabs.aggregation import display_tab as aggregation_tab
from tabs.produktverteilung import display_tab as produktverteilung_tab
from tabs.rohdaten import display_tab as rohdaten_tab
from tabs.zerlegung import display_tab as zerlegung_tab
from tabs.abcxyz import display_tab as abcxyz_tab
from tabs.statistische_tests import display_tab as statistische_tests_tab
from tabs.forecast import display_tab as forecast_tab
from tabs.advanced_forecast import display_tab as advanced_forecast_tab
from tabs.multivariate_forecast import display_tab as multivariate_forecast_tab
from tabs.forecast_simple import display_tab as forecast_simple_tab
from tabs.forecast_complex import display_tab as forecast_complex_tab
from tabs.glossar import display_tab as glossar_tab

# Page config
st.set_page_config(layout="wide", page_title="PrABCast", page_icon=":bar_chart:", initial_sidebar_state="expanded")

# Main content
with st.container():
    st.image(os.path.join(ROOT_DIR, "media", "Projektlogo.svg"), width=250)
    
    # Create tabs
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Was ist PrABCast?",
        "Datenansicht",
        "Absatzanalyse",
        "Modellvergleich",
        "Absatzprognose",
        "Glossar"
    ])

# Tab 0: Introduction
with tab0:
    st.header("Was ist PrABCast?")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
            **PrABCast** ist ein Forschungsprojekt des RIF - Instituts für Forschung und Transfer in Kooperation mit dem 
            Institut für Produktionssysteme (IPS) an der Technischen Universität Dortmund. Ziel dieses Vorhabens ist es, es Unternehmen zu erleichtern, Maschinelle Lernverfahren in der Absatz- und Bedarfsprognose einzusetzen.

            Ausgangspunkt sind dabei die zunehmenden Anforderungen an eine präzise, robuste und 
            zugleich effiziente Absatzplanung, die Unternehmen vor die Herausforderung stellen, 
            große Datenmengen und volatile Märkte in zuverlässige Prognosen zu übersetzen. 
            PrABCast greift diese Herausforderungen auf, indem es datengetriebene Ansätze 
            mit etablierten statistischen und analytischen Verfahren verbindet.

            Folgende Schwerpunkte stehen im Vordergrund:
            - **Datenaggregation und -aufbereitung**: Konsolidierung und Standardisierung großer 
              Absatzdatensätze, um eine verlässliche Basis für weiterführende Analysen zu schaffen.
            - **Produktverteilung und Kennwertberechnungen**: Einordnung von Produkten anhand 
              unterschiedlicher Kriterien, wie z. B. Absatzvolumen, Wertanteil oder 
              Nachfragevariabilität, Klassifikation von Produkten nach wirtschaftlicher Bedeutung 
              (ABC) und Nachfragedynamik (XYZ), um gezielt Prognose- und Planungsressourcen 
              einsetzen zu können.
            - **Statistische Tests und Zeitreihenanalysen**: Anwendung moderner statistischer 
              Verfahren zur Identifikation von Mustern, Trends und Saisonalitäten in den Absatzdaten; Ausgangspunkt für die zielgerichtete Auswahl von Prognosemethoden.
            - **Datenanreicherung**: Integration von externen Datenquellen, wie z. B. Konjunkturindikatoren, Wirtschaftsindikatoren und Branchenindizes.
            - **Modellvergleich mit einfachen bis komplexen Modellen**: Nutzung klassischer 
              Prognosemodelle ebenso wie fortgeschrittener Verfahren – etwa Machine-Learning-Algorithmen 
              oder multivariate Modelle – um verlässliche Vorhersagen zukünftiger Absatzmengen zu erstellen.


            Durch diese ganzheitliche Herangehensweise wird es möglich, die Prognosequalität signifikant 
            zu steigern. Für Unternehmen bedeutet dies eine nachhaltige Optimierung der Bestandsplanung, 
            höhere Lieferfähigkeit, geringere Kosten durch Fehlbestände oder Überkapazitäten, 
            sowie eine insgesamt robustere Entscheidungsfindung im Supply-Chain-Management.

            Weitere Informationen finden Sie auf der 
            [Website des Instituts für Produktionssysteme (IPS)](https://ips.mb.tu-dortmund.de/forschen-beraten/forschungsprojekte/prabcast/).
        """)
    with col2:
        st.image(os.path.join(ROOT_DIR, "media", "Projektgrafik.svg"), width=400)

# Tab 1: Upload
with tab1:
    st.header("Datenansicht")
    col1, col2 = st.columns([1, 1])
    with col1:
        upload_tab()
    with col2:
        st.write("""
            Auf dieser Seite können Sie Ihre Daten hochladen und konfigurieren. 
            - **Dateiupload**: Laden Sie Ihre Datendateien hoch.
            - **Datumsauswahl**: Wählen Sie das Datum und die Produkte aus, die Sie betrachten möchten.
            - **Zeitraumauswahl**: Definieren Sie einen Gesamtzeitraum, der später filterbar ist.
            - **Automatische Datumserkennung**: Die Anwendung erkennt automatisch Datumsformate in Ihren Daten.
            
            **Details:**
            - Nachdem Sie Ihre Daten hochgeladen haben, können Sie das Datum und die Produkte auswählen, die Sie analysieren möchten.
            - Sie können einen Gesamtzeitraum definieren, der später nach Bedarf gefiltert werden kann.
            - Die automatische Datumserkennung erleichtert die Verarbeitung Ihrer Daten, indem sie verschiedene Datumsformate erkennt und korrekt interpretiert.
            - Diese Seite bietet eine benutzerfreundliche Oberfläche, um sicherzustellen, dass Ihre Daten korrekt und effizient verarbeitet werden können.
        """)
# Tab 5: Glossar
with tab5:
    st.header("Was ist PrABCast?")
    col1, col2 = st.columns([2, 1])
        
    with col1:
            glossar_tab()
        
    with col2:
            st.image(os.path.join(ROOT_DIR, "media", "Projektlogo.svg"), width=400)



# Check if ready for further processing
if st.session_state.get('ready_for_processing', False) is True:

    # Tab 2: Sales Analysis
    with tab2:
        subtab_selection = st.tabs([
            "Übersicht",  # New subtab for overview
            "Rohdaten und Kennwerte", 
            "Aggregation", 
            "Produktverteilung", 
            "Zerlegung", 
            "ABC-XYZ Analyse",
            "Statistische Tests"
        ])

        with subtab_selection[0]:
            st.header("Übersicht der Absatzanalyse")
            st.write("""
                In der Absatzanalyse stehen Ihnen folgende Funktionen zur Verfügung:
                - **Rohdaten und Kennwerte**: Anzeige und Analyse der Rohdaten einzelner Produkte, Aggregation des Gesamtabsatz, sowie Berechnung wichtiger Kennwerte.
                - **Aggregation**: Aggregation der Daten nach verschiedenen Kriterien.
                - **Produktverteilung**: Analyse der Verteilung der Produkte am Gesamtabsatz.
                - **Zerlegung**: Dekomposition der Zeitreihen in Trend-, Saison- und Restkomponenten.
                - **ABC-XYZ Analyse**: Klassifikation der Produkte nach wirtschaftlicher Bedeutung (ABC) und Nachfragedynamik (XYZ).
                - **Statistische Tests**: Anwendung statistischer Tests zur Identifikation von Mustern und Trends in den Daten.
            """)

        with subtab_selection[1]:
            rohdaten_tab()

        with subtab_selection[2]:
            aggregation_tab()

        with subtab_selection[3]:
            produktverteilung_tab()

        with subtab_selection[4]:
            zerlegung_tab()

        with subtab_selection[5]:
            abcxyz_tab()
            
        with subtab_selection[6]:
            statistische_tests_tab()

    # Tab 3: Forecast
    with tab3:
        forecast_subtabs = st.tabs(["Übersicht", "Einfach", "Datenanreicherung", "Komplex"])

        with forecast_subtabs[0]:
            st.header("Übersicht der Prognosemodelle")
            st.write("""
                In diesem Tab können Sie verschiedene Prognosemethoden für Ihre Absatzdaten nutzen:

                **1. Einfache Prognose**
                - Univariate Zeitreihenanalyse basierend auf historischen Verkaufsdaten
                - Klassische statistische Methoden und Machine Learning Verfahren
                - Vergleich verschiedener Modelle mit Gütekriterien
                - Visualisierung der Prognosen mit Konfidenzintervallen

                **2. Datenanreicherung**
                - Identifikation relevanter externer Einflussfaktoren
                - Analyse von Industrieindizes, Marktdaten und regionalen Indikatoren
                - Korrelations- und Kausalitätsanalysen
                - Vorbereitung der Daten für multivariate Prognosen

                **3. Komplexe Prognose**
                - Multivariate Prognosemodelle unter Einbeziehung externer Faktoren
                - Fortgeschrittene Machine Learning Verfahren (LSTM, XGBoost)
                - Kombination von Absatzdaten mit identifizierten Einflussfaktoren
                - Erweiterte Prognosegenauigkeit durch zusätzliche Informationen
                
                Wählen Sie die für Ihre Anforderungen passende Methode und vergleichen Sie 
                die Ergebnisse verschiedener Modelle.
            """)

        with forecast_subtabs[1]:
            forecast_tab()

        with forecast_subtabs[2]:
            advanced_forecast_tab()

        with forecast_subtabs[3]:
            multivariate_forecast_tab()

    # Tab 4: Absatzprognose
    with tab4:
        forecast_subtabs = st.tabs(["Übersicht", "Einfach", "Komplex"])

        with forecast_subtabs[0]:
            st.header("Übersicht der Absatzprognose")
            st.write("""
                In diesem Tab können Sie zukünftige Absatzmengen prognostizieren:

                **Einfach**
                - Univariate Zeitreihenprognose basierend auf historischen Daten
                - Prognose der gewählten Monate ohne Testzeitraum
                - Möglichkeit, die Prognose herunterzuladen

                **Komplex**
                - Multivariate Prognose mit zusätzlichen Einflussfaktoren
                - Verwendung externer Variablen für präzisere Vorhersagen
                - Möglichkeit, die Prognose herunterzuladen
                
                **Vorteile**:
                
                Die Absatzprognose mithilfe von Machine-Learning-Algorithmen oder multivariate Modellen bietet 
                zahlreiche Vorteile im Vergleich zu herkömmlichen statistischen Methoden oder einfachen linearen 
                Modellen. Hier sind die wichtigsten Vorteile im Detail:
                
                1. **Verbesserte Vorhersagegenauigkeit**:
                ML-Modelle können eine hohe Vorhersagegenauigkeit erzielen, da sie in der Lage sind, komplexe Muster in großen Datenmengen zu erkennen. Während traditionelle Methoden oft lineare Annahmen über Daten treffen, können maschinelle Lernverfahren nicht-lineare Beziehungen und Wechselwirkungen zwischen Variablen besser erfassen. Dadurch liefern sie genauere Prognosen, insbesondere in dynamischen und volatilen Märkten.
                
                2. **Skalierbarkeit**:
                ML-Modelle sind in der Lage, eine große Menge an Daten zu verarbeiten, was sie besonders nützlich für Unternehmen mit umfangreichen Verkaufsdaten macht. Sie können problemlos Informationen aus mehreren Quellen (z. B. historische Verkaufsdaten, saisonale Trends, externe Daten wie Wetter oder wirtschaftliche Indikatoren) integrieren und analysieren. Das macht sie skalierbar und flexibel für Unternehmen unterschiedlicher Größe und Komplexität.
                
                3. **Anpassungsfähigkeit an Änderungen**:
                Im Gegensatz zu starren traditionellen Methoden können ML-Modelle dynamisch lernen und sich an veränderte Marktbedingungen anpassen. Durch regelmäßige Aktualisierung und Retraining des Modells können neue Daten genutzt werden, um die Vorhersagen zu verfeinern, was insbesondere in sich schnell ändernden Branchen von Vorteil ist.
                
                4. **Berücksichtigung mehrerer Faktoren**:
                ML-Modelle können eine Vielzahl von Einflussfaktoren in die Vorhersage einbeziehen, darunter historische Verkaufsdaten, Marktbedingungen, Preisänderungen, Marketingaktivitäten, saisonale Schwankungen und sogar externe Faktoren wie Wetterbedingungen oder soziale Trends. Dies ermöglicht es Unternehmen, eine ganzheitlichere und präzisere Vorhersage zu erstellen, die auf mehr als nur vergangene Verkaufszahlen basiert.
                
                5. **Erkennung von Mustern und Anomalien**:
                ML-Algorithmen sind besonders gut darin, verborgene Muster und Anomalien in den Daten zu erkennen, die von traditionellen Methoden übersehen werden könnten. Sie können z.B. Vorhersagen über außergewöhnliche Verkaufsereignisse oder Spitzen erkennen und Unternehmen darauf vorbereiten, auf solche Situationen besser zu reagieren.
                
                6. **Automatisierung und Effizienz**:
                Maschinelle Lernverfahren ermöglichen eine automatisierte und kontinuierliche Absatzprognose, was Zeit und Ressourcen spart. Unternehmen müssen nicht mehr manuell große Datenmengen analysieren und interpretieren. Durch automatisierte ML-Modelle kann die Absatzprognose kontinuierlich aktualisiert und optimiert werden, was eine schnellere Entscheidungsfindung ermöglicht.
                
                7. **Personalisierung und Segmentierung**:
                ML-Modelle sind in der Lage, Kunden oder Produktsegmente individuell zu analysieren. Dies ermöglicht personalisierte Prognosen auf der Ebene einzelner Kundengruppen oder Produkte, anstatt allgemeine Prognosen für das gesamte Unternehmen zu erstellen. Unternehmen können dadurch besser planen und gezieltere Marketing- oder Verkaufsstrategien entwickeln.
                
                8. **Vermeidung menschlicher Fehler**:
                Da ML-Modelle weitgehend automatisiert arbeiten, können sie menschliche Fehler bei der Analyse und Interpretation von Daten minimieren. Sie bieten eine objektive, datengetriebene Analyse, die frei von subjektiven Verzerrungen oder Fehlinterpretationen ist, die in traditionellen Ansätzen oft auftreten.
                
                9. **Verbesserte Entscheidungsfindung**:
                Durch die Bereitstellung genauerer und zeitnaher Vorhersagen können Unternehmen bessere Geschäftsentscheidungen treffen. Sie können die Produktion optimieren, Lagerbestände besser planen, auf Nachfrageschwankungen vorbereitet sein und gezieltere Verkaufsstrategien entwickeln, um Umsätze zu maximieren und Kosten zu senken.
                
                10. **Kostenoptimierung**:
                Eine präzise Absatzprognose führt dazu, dass Unternehmen ihre Ressourcen besser nutzen können. Mit genauen Vorhersagen können Überproduktion und damit verbundene Lagerkosten sowie Engpässe und verlorene Verkaufschancen vermieden werden. Durch die genaue Planung können Unternehmen ihre Produktions- und Lieferketten effizienter gestalten.
            """)

        with forecast_subtabs[1]:
            # Einfacher Forecast
            forecast_simple_tab()

        with forecast_subtabs[2]:
            # Komplexer Forecast
            forecast_complex_tab()

    
else:
    st.write("Bitte zuerst Dateiansicht ausfüllen.")

# Add footer with image and text
col1, col2 = st.columns(2)
with col1:
    st.image(os.path.join(ROOT_DIR, "media", "image.png"), width=100)
with col2:
    st.image(os.path.join(ROOT_DIR, "media", "igflogo.png"), width=150)

st.write("Copyright (c) 2024 - RIF - Institut für Forschung und Transfer e.V.")