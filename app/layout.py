import streamlit as st
import os
from pathlib import Path

# Get root directory
ROOT_DIR = Path(__file__).parent.parent

# Import Session State Management
from setup_module.session_state_simple import init_state, ready_for_processing

# Import UI Components
from setup_module.design_system import UI

# Initialize session state
init_state()

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
    st.image(os.path.join(ROOT_DIR, "media", "PrABCastLogo.png"), width=250)
    
    # Create tabs
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Was ist PrABCast?",
        "📊 Daten",
        "📈 Analyse",
        "🔬 Modellvergleich",
        "🎯 Prognose",
        "📚 Glossar & Services"
    ])

# Tab 0: Introduction
with tab0:
    st.header("Was ist PrABCast?")
    
    # Demonstrator notice
    st.warning("""
        **ℹ️ Hinweis zum Demonstrator**  
        Diese Anwendung ist ein **wissenschaftlicher Demonstrator** zur Veranschaulichung moderner Prognoseverfahren. 
        Um die Benutzerfreundlichkeit zu gewährleisten, erfolgt die Modellparametrisierung auf Basis **heuristischer 
        Standardwerte**. Eine produktive Implementierung mit umfassender Hyperparameter-Optimierung 
        (Grid Search, Bayesian Optimization, erweiterte Kreuzvalidierung) ist auf Anfrage bei RIF verfügbar.
    """)
    
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
    st.caption("📍 Daten > CSV hochladen und konfigurieren")
    upload_tab()
# Tab 5: Glossar
with tab5:
    st.header("Glossar")
    glossar_tab()



# Check if ready for further processing
if ready_for_processing():

    # Tab 2: Sales Analysis
    with tab2:
        subtab_selection = st.tabs([
            "Rohdaten", 
            "Aggregation", 
            "Produktverteilung", 
            "Zerlegung", 
            "ABC-XYZ",
            "Statistische Tests"
        ])

        with subtab_selection[0]:
            st.caption("📍 Analyse > Rohdaten und Kennwerte")
            rohdaten_tab()

        with subtab_selection[1]:
            st.caption("📍 Analyse > Daten aggregieren")
            aggregation_tab()

        with subtab_selection[2]:
            st.caption("📍 Analyse > Produktverteilung am Gesamtabsatz")
            produktverteilung_tab()

        with subtab_selection[3]:
            st.caption("📍 Analyse > STL-Zerlegung (Trend, Saison, Rest)")
            zerlegung_tab()

        with subtab_selection[4]:
            st.caption("📍 Analyse > Klassifikation nach Wert und Variabilität")
            abcxyz_tab()
            
        with subtab_selection[5]:
            st.caption("📍 Analyse > Stationarität und Trends")
            statistische_tests_tab()

    # Tab 3: Modellvergleich (Evaluation mit historischen Daten)
    with tab3:
        # Demonstrator notice for model comparison
        st.warning("""
            **🔬 Demonstrator-Hinweis: Vereinfachte Modellparametrisierung**  
            Die in diesem Tool implementierten Modelle verwenden **heuristische Standardparameter** zur Vereinfachung 
            der Handhabung. Für produktive Anwendungen empfehlen wir eine vollständige Hyperparameter-Optimierung mittels:
            - **Grid Search** oder **Bayesian Optimization** für systematische Parametersuche
            - **Erweiterte Kreuzvalidierung** (k-fold, Time Series Split) für robuste Modellbewertung
            - **Feature Engineering** und domänenspezifische Anpassungen
            
            RIF unterstützt Sie gerne bei der Implementierung produktionsreifer Prognoselösungen.
        """)
        
        forecast_subtabs = st.tabs(["Univariate Modelle", "Datenanreicherung", "Multivariate Modelle"])

        with forecast_subtabs[0]:
            st.caption("📍 Modellvergleich > Univariate Prognose mit Train/Test-Split")
            st.info("Hier werden verschiedene Modelle **auf historischen Daten getestet**. Der Prognosehorizont bestimmt, wie viele der letzten Monate als Test-Set verwendet werden.")
            forecast_tab()

        with forecast_subtabs[1]:
            st.caption("📍 Modellvergleich > Externe Einflussfaktoren hinzufügen")
            st.info("Reichern Sie Ihre Daten mit externen Indizes an (Wirtschaftsdaten, Branchenindizes). Diese können dann in multivariaten Modellen verwendet werden.")
            advanced_forecast_tab()

        with forecast_subtabs[2]:
            st.caption("📍 Modellvergleich > Multivariate Prognose mit Train/Test-Split")
            st.info("Testen Sie Modelle **mit externen Faktoren** auf historischen Daten. Basis sind die angereicherten Daten aus dem vorherigen Schritt.")
            multivariate_forecast_tab()

    # Tab 4: Absatzprognose (Echte Prognose in die Zukunft)
    with tab4:
        st.info("⚠️ **Wichtig:** Führen Sie zuerst den **Modellvergleich (Tab 3)** durch, um das beste Modell zu identifizieren. Hier erstellen Sie dann die **echte Prognose für zukünftige Monate**.")
        
        # Demonstrator notice for forecasting
        st.warning("""
            **🎯 Demonstrator-Hinweis: Vereinfachte Prognoseparametrisierung**  
            Die Zukunftsprognosen basieren auf **vereinfachten Modellparametern** für eine intuitive Handhabung. 
            Produktive Prognosesysteme erfordern:
            - **Optimierte Hyperparameter** durch systematische Suche (Grid Search, Random Search, Bayesian Optimization)
            - **Ensemble-Methoden** zur Erhöhung der Robustheit
            - **Kontinuierliches Monitoring** und Modell-Retraining
            - **Uncertainty Quantification** für Konfidenzintervalle
            
            Für unternehmenskritische Prognosen unterstützt RIF bei der Entwicklung maßgeschneiderter Lösungen.
        """)
        
        forecast_subtabs = st.tabs(["Univariate Prognose", "Multivariate Prognose"])

        with forecast_subtabs[0]:
            st.caption("📍 Absatzprognose > Univariate Zukunftsprognose")
            st.success("**Wann verwenden?** Nach dem Modellvergleich (Tab 3 > Univariate Modelle), um mit dem besten Modell **X Monate in die Zukunft** zu prognostizieren.")
            forecast_simple_tab()

        with forecast_subtabs[1]:
            st.caption("📍 Absatzprognose > Multivariate Zukunftsprognose")
            st.success("**Wann verwenden?** Nach Datenanreicherung (Tab 3 > Datenanreicherung) und Modellvergleich (Tab 3 > Multivariate Modelle), um mit externen Faktoren **in die Zukunft** zu prognostizieren.")
            forecast_complex_tab()

    
else:
    st.write("Bitte zuerst Dateiansicht ausfüllen.")

# Add footer with image and text
col1, col2 = st.columns(2)
with col1:
    st.image(os.path.join(ROOT_DIR, "media", "image.png"), width=100)
with col2:
    st.image(os.path.join(ROOT_DIR, "media", "igflogo.png"), width=150)

st.write("Copyright (c) 2025 - RIF - Institut für Forschung und Transfer e.V.")