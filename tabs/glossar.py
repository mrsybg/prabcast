# tabs/glossar.py
"""
Glossar & Weiteres Tab

Zeigt ein wissenschaftlich fundiertes Glossar und RIF-Services.
"""

from pathlib import Path
import streamlit as st

try:
    # Optional – falls du App-weit State brauchst
    from setup_module.session_state import get_app_state
except Exception:
    # Fallback, wenn das Modul im Dev-Kontext nicht verfügbar ist
    def get_app_state():
        return {}

ROOT_DIR = Path(__file__).parent.parent


def _render_glossary_term(term: str, data: dict) -> None:
    """Renderer für einen Glossar-Begriff im Expander."""
    with st.expander(f"**{term}** – {data.get('full_name', '')}", expanded=False):
        st.markdown(f"**Kategorie:** {data.get('category', '–')}")
        st.markdown(data.get("description", ""))
        related = data.get("related") or []
        if related:
            st.markdown(f"**Verwandte Begriffe:** {', '.join(related)}")


def display_tab() -> None:
    """Render 'Glossar & Weiteres' Tab."""
    state = get_app_state()  # aktuell nicht benötigt, aber behalten

    st.caption("Glossar & Weiteres")

    main_tabs = st.tabs(["Glossar", "Schulungen & Services"])

    # =========================
    # TAB 1: GLOSSAR
    # =========================
    with main_tabs[0]:
        st.header("Fachbegriffe zur Absatzprognose")
        st.markdown("Wissenschaftlich fundierte Erklärungen der wichtigsten Konzepte und Methoden.")

        col1, col2 = st.columns([2, 1])
        with col1:
            search_term = st.text_input("Begriff suchen", placeholder="z. B. ARIMA, Saisonalität, Prophet …")
        with col2:
            category_filter = st.multiselect(
                "Nach Kategorie filtern",
                ["Modelle", "Metriken", "Statistische Tests", "Konzepte", "Preprocessing", "ML-Methoden"],
                default=[],
            )

        st.markdown("---")

        # ---- Glossar-Daten ----
        glossary = {
            # ========== MODELLE ==========
            "ARIMA": {
                "category": "Modelle",
                "full_name": "AutoRegressive Integrated Moving Average",
                "description": """
**ARIMA** ist ein klassisches statistisches Modell für univariate Zeitreihenprognosen.

**Komponenten:**
- **AR (AutoRegressive, p)**: nutzt vergangene Werte
- **I (Integrated, d)**: Differenzenbildung zur Stationarisierung
- **MA (Moving Average, q)**: nutzt vergangene Prognosefehler

**Vorteile:** interpretierbar, gut dokumentiert  
**Nachteile:** benötigt Stationarität, univariat

**Literatur:** Box & Jenkins (1970/2015), *Time Series Analysis: Forecasting and Control*.
                """,
                "related": ["SARIMA", "Stationarität", "ADF-Test", "KPSS-Test"],
            },
            "SARIMA": {
                "category": "Modelle",
                "full_name": "Seasonal ARIMA",
                "description": """
**SARIMA** erweitert ARIMA um saisonale Komponenten **(P, D, Q)** und die **Saisonperiode m**.

**Anwendung:** ausgeprägte saisonale Muster (z. B. s=12 bei Monatsdaten mit Jahreszyklus)

**Vorteile:** explizite Saisonalität, interpretierbar  
**Nachteile:** viele Parameter, höherer Abstimmungsaufwand

**Literatur:** Box et al. (2015).
                """,
                "related": ["ARIMA", "Saisonalität", "STL-Zerlegung", "Prophet"],
            },
            "Prophet": {
                "category": "Modelle",
                "full_name": "Prophet (Meta/Facebook)",
                "description": """
**Prophet** ist ein additives Modell: **Trend + Saisonalitäten + Feiertage/Ereignisse**, robust ggü. Ausreißern und fehlenden Werten.
Unterstützt **Changepoints** für Trendwechsel.

**Vorteile:** benutzerfreundlich, praktikabel für Business-Zeitreihen  
**Nachteile:** geringere Transparenz als SARIMA, Standardannahmen

**Literatur:** Taylor & Letham (2018), *Forecasting at Scale*.
                """,
                "related": ["Saisonalität", "Changepoints", "SARIMA"],
            },
            "LSTM": {
                "category": "ML-Methoden",
                "full_name": "Long Short-Term Memory Networks",
                "description": """
**LSTM** ist ein RN-Architekturtyp, der lange Abhängigkeiten in Sequenzen modelliert (Gate-Mechanismen).

**Vorteile:** nichtlinear, multivariat, erfasst komplexe Muster  
**Nachteile:** daten- und rechenintensiv, weniger interpretierbar

**Literatur:** Hochreiter & Schmidhuber (1997).
                """,
                "related": ["XGBoost", "Feature Engineering"],
            },
            "XGBoost": {
                "category": "ML-Methoden",
                "full_name": "eXtreme Gradient Boosting",
                "description": """
**XGBoost** (Gradient Boosting Trees) eignet sich für strukturierte Features (Lags, Rolling-Stats, Kalender, exogene Variablen).

**Vorteile:** hohe Genauigkeit, schnell, robust  
**Nachteile:** Feature Engineering nötig, Unsicherheitsintervalle nicht nativ

**Literatur:** Chen & Guestrin (2016).
                """,
                "related": ["LSTM", "Feature Engineering", "Ensemble-Methoden"],
            },

            # ========== METRIKEN ==========
            "sMAPE": {
                "category": "Metriken",
                "full_name": "Symmetric Mean Absolute Percentage Error",
                "description": """
**sMAPE** misst prozentuale Abweichungen symmetrisch:

`sMAPE = 100% × mean(2 × |F - A| / (|F| + |A|))`

**Interpretation:**  
0–10 % ausgezeichnet, 10–20 % gut, 20–50 % akzeptabel, >50 % schwach.

+ skalenunabhängig, vergleichbar  
− problematisch nahe 0

**Literatur:** Makridakis (1993).
                """,
                "related": ["MAE", "RMSE", "Konfidenzintervalle"],
            },
            "MAE": {
                "category": "Metriken",
                "full_name": "Mean Absolute Error",
                "description": """
**MAE** ist der mittlere absolute Fehler in Originaleinheiten:

`MAE = mean(|F - A|)`

+ leicht interpretierbar, robust ggü. Ausreißern als RMSE  
− nicht skalenunabhängig

**Literatur:** Willmott & Matsuura (2005).
                """,
                "related": ["sMAPE", "RMSE", "MSE"],
            },

            # ========== TESTS ==========
            "ADF-Test": {
                "category": "Statistische Tests",
                "full_name": "Augmented Dickey-Fuller Test",
                "description": """
Prüft auf **Unit Root** (Nicht-Stationarität).  
**H₀:** nicht stationär. **p < 0,05** ⇒ stationär.

**Literatur:** Dickey & Fuller (1979).
                """,
                "related": ["KPSS-Test", "Stationarität", "ARIMA"],
            },
            "KPSS-Test": {
                "category": "Statistische Tests",
                "full_name": "Kwiatkowski-Phillips-Schmidt-Shin Test",
                "description": """
Komplementär zum ADF.  
**H₀:** stationär. **p < 0,05** ⇒ nicht stationär.

**Best Practice:** ADF + KPSS gemeinsam interpretieren.
**Literatur:** Kwiatkowski et al. (1992).
                """,
                "related": ["ADF-Test", "Stationarität"],
            },

            # ========== KONZEPTE ==========
            "Saisonalität": {
                "category": "Konzepte",
                "full_name": "Saisonale Muster",
                "description": """
Wiederkehrende Muster (monatlich, quartalsweise, jährlich, wöchentlich).
Erkennbar über **ACF**, **STL-Zerlegung**, visuelle Muster.

+ wichtig für Modellwahl (SARIMA/Prophet)  
**Literatur:** Cleveland et al. (1990).
                """,
                "related": ["SARIMA", "STL-Zerlegung", "Prophet"],
            },
            "Stationarität": {
                "category": "Konzepte",
                "full_name": "Stationäre Zeitreihen",
                "description": """
Konstante **Momente** (Mittelwert, Varianz) und **Autokorrelation** über die Zeit.
Wichtig v. a. für ARIMA.

**Erreichen:** Differenzen, Log-Transform, Detrending.  
**Tests:** ADF, KPSS.

**Literatur:** Hamilton (1994).
                """,
                "related": ["ADF-Test", "KPSS-Test", "ARIMA"],
            },
            "Prognosehorizont": {
                "category": "Konzepte",
                "full_name": "Forecast Horizon",
                "description": """
Anzahl zukünftiger Perioden; Unsicherheit nimmt mit dem Horizont zu.

**Daumenregel:** Horizont ≤ 20–30 % der verfügbaren Trainingslänge.

**Literatur:** Hyndman & Athanasopoulos (2021).
                """,
                "related": ["Konfidenzintervalle", "Train-Test-Split"],
            },

            # ========== PREPROCESSING ==========
            "STL-Zerlegung": {
                "category": "Preprocessing",
                "full_name": "Seasonal-Trend decomposition using Loess",
                "description": """
Zerlegt in **Trend**, **Saisonalität**, **Residuen**. Nützlich zur Exploration,
Anomalieerkennung und als Feature-Quelle.

**Literatur:** Cleveland et al. (1990).
                """,
                "related": ["Saisonalität", "Detrending", "Prophet", "SARIMA"],
            },
            "Feature Engineering": {
                "category": "Preprocessing",
                "full_name": "Feature-Konstruktion für Zeitreihen",
                "description": """
Typisch: **Lags**, **Rolling-Statistiken**, **Kalenderfeatures**, **exogene Variablen**.
Wichtig für ML-Modelle wie XGBoost/LSTM. **Datenlecks vermeiden!**

**Literatur:** Zheng & Casari (2018).
                """,
                "related": ["XGBoost", "LSTM", "Ensemble-Methoden"],
            },

            # ========== ENSEMBLE ==========
            "Ensemble-Methoden": {
                "category": "ML-Methoden",
                "full_name": "Ensemble Forecasting",
                "description": """
**Averaging**, **Weighted Average** (z. B. gewichtet nach sMAPE), **Stacking** (Meta-Modell).

+ robuster, geringeres Overfitting-Risiko  
**Literatur:** Clemen (1989), Dietterich (2000).
                """,
                "related": ["XGBoost", "Random Forest", "Model Averaging"],
            },
            
            # ========== WEITERE WICHTIGE BEGRIFFE ==========
            "RMSE": {
                "category": "Metriken",
                "full_name": "Root Mean Squared Error",
                "description": """
**RMSE** ist die Wurzel aus dem mittleren quadratischen Fehler:

`RMSE = sqrt(mean((F - A)²))`

**Vorteile:** bestraft große Fehler stärker als MAE, in Originaleinheiten  
**Nachteile:** sensitiv gegenüber Ausreißern, nicht skalenunabhängig

**Wann verwenden:** wenn große Prognosefehler besonders kritisch sind (z.B. Safety Stock Planung)

**Literatur:** Standard-Metrik in der Zeitreihenanalyse.
                """,
                "related": ["MAE", "sMAPE", "MSE"],
            },
            
            "MSE": {
                "category": "Metriken",
                "full_name": "Mean Squared Error",
                "description": """
**MSE** ist der mittlere quadratische Fehler:

`MSE = mean((F - A)²)`

+ mathematisch einfach zu optimieren (konvex)  
− schwer interpretierbar (quadrierte Einheiten), sehr sensitiv ggü. Ausreißern

**Verwendung:** hauptsächlich als Optimierungskriterium beim Training, weniger zur Bewertung.

**Literatur:** Standard-Verlustfunktion im maschinellen Lernen.
                """,
                "related": ["RMSE", "MAE", "Overfitting"],
            },
            
            "MASE": {
                "category": "Metriken",
                "full_name": "Mean Absolute Scaled Error",
                "description": """
**MASE** skaliert MAE mit der durchschnittlichen Änderung der Trainingsdaten (naiver Forecast):

`MASE = MAE(Modell) / MAE(naiver_Forecast)`

**Interpretation:**  
- MASE < 1: Modell besser als naive Prognose +  
- MASE = 1: gleich gut wie naive Prognose  
- MASE > 1: schlechter als naive Prognose −

+ skalenunabhängig, interpretierbar, robust  
**Literatur:** Hyndman & Koehler (2006).
                """,
                "related": ["MAE", "sMAPE", "Naive Forecast"],
            },
            
            "ACF/PACF": {
                "category": "Statistische Tests",
                "full_name": "Autokorrelationsfunktion / Partielle Autokorrelationsfunktion",
                "description": """
**ACF** misst die Korrelation zwischen Zeitreihe und ihren Lags.  
**PACF** misst die Korrelation nach Herausrechnung kürzerer Lags.

**Verwendung:**
- ACF: Erkennung von MA-Ordnung (q) und Saisonalität  
- PACF: Erkennung von AR-Ordnung (p)  
- Gemeinsam: ARIMA-Parameter-Bestimmung (p, q)

**In PrABCast:** Visualisiert im "Statistische Tests"-Tab.

**Literatur:** Box & Jenkins (1970).
                """,
                "related": ["ARIMA", "SARIMA", "Saisonalität"],
            },
            
            "Train-Test-Split": {
                "category": "Konzepte",
                "full_name": "Trainings- und Testdaten-Aufteilung",
                "description": """
Aufteilung der Daten in **Trainingsmenge** (Modell lernt) und **Testmenge** (Modell wird evaluiert).

**Zeitreihen-Besonderheit:** Immer chronologisch! Niemals shuffeln!  
**Typisch:** 70-80% Training, 20-30% Test.

**Zweck:**
- Vermeidung von Overfitting  
- Realistische Schätzung der Prognosequalität  
- Modellvergleich auf gleicher Basis

**In PrABCast:** Tab 3 "Modellvergleich" nutzt automatisch Train-Test-Split.

**Literatur:** Standard in der Prognosevalidierung.
                """,
                "related": ["Overfitting", "Cross-Validation", "Prognosehorizont"],
            },
            
            "Overfitting": {
                "category": "Konzepte",
                "full_name": "Überanpassung",
                "description": """
**Overfitting** tritt auf, wenn ein Modell Trainingsdaten zu genau lernt (inkl. Rauschen) 
und dadurch auf neuen Daten schlecht performt.

**Symptome:**
- Sehr gute Performance auf Training, schlechte auf Test  
- Zu viele Parameter für zu wenig Daten  
- Modell reagiert auf zufällige Schwankungen

**Vermeidung:**
- Train-Test-Split oder Cross-Validation  
- Regularisierung (L1/L2)  
- Einfachere Modelle bevorzugen  
- Mehr Daten sammeln

**In PrABCast:** Smart Defaults warnen vor zu komplexen Modellen bei wenig Daten.

**Literatur:** Fundamentales Konzept im maschinellen Lernen.
                """,
                "related": ["Train-Test-Split", "Regularisierung", "MSE"],
            },
            
            "Cross-Validation": {
                "category": "Konzepte",
                "full_name": "Kreuzvalidierung für Zeitreihen",
                "description": """
**Cross-Validation** testet Modell-Performance auf mehreren Trainings-/Test-Kombinationen.

**Zeitreihen-spezifisch:** Time Series Split (rolling window):
- Fenster 1: Train [1-50], Test [51-60]  
- Fenster 2: Train [1-60], Test [61-70]  
- Fenster 3: Train [1-70], Test [71-80]  
usw.

+ robustere Schätzung der Performance  
− rechenintensiver

**In PrABCast:** Wird für finale Modellauswahl in komplexeren Szenarien empfohlen.

**Literatur:** Hyndman & Athanasopoulos (2021).
                """,
                "related": ["Train-Test-Split", "Overfitting"],
            },
            
            "Konfidenzintervalle": {
                "category": "Konzepte",
                "full_name": "Unsicherheitsintervalle für Prognosen",
                "description": """
**Konfidenzintervalle** zeigen den Bereich, in dem der wahre Wert mit bestimmter Wahrscheinlichkeit liegt.

**Typisch:** 80% und 95% Intervalle  
- 95% CI: Mit 95% Wahrscheinlichkeit liegt der wahre Wert in diesem Bereich

**Eigenschaften:**
- Breite nimmt mit Prognosehorizont zu  
- ARIMA/Prophet liefern Intervalle automatisch  
- XGBoost/LSTM benötigen spezielle Techniken (Quantile Regression, Monte Carlo)

**In PrABCast:** Visualisiert als Schattierungen um Prognoselinie.

**Literatur:** Standard in der Zeitreihenprognose.
                """,
                "related": ["Prognosehorizont", "Prophet", "ARIMA"],
            },
            
            "Changepoints": {
                "category": "Konzepte",
                "full_name": "Trendwechselpunkte",
                "description": """
**Changepoints** sind Zeitpunkte, an denen sich der Trend ändert (z.B. von steigend zu fallend).

**Ursachen:**
- Marktveränderungen  
- Produkteinführungen/-auslauf  
- Strategiewechsel  
- Externe Schocks (Pandemie, Lieferengpässe)

**Prophet** erkennt Changepoints automatisch.  
**ARIMA/SARIMA** haben damit Schwierigkeiten → manuelle Intervention nötig.

**In PrABCast:** Prophet zeigt erkannte Changepoints im Plot.

**Literatur:** Taylor & Letham (2018).
                """,
                "related": ["Prophet", "Trend", "Strukturbrüche"],
            },
            
            "Detrending": {
                "category": "Preprocessing",
                "full_name": "Trendbereinigung",
                "description": """
**Detrending** entfernt langfristige Trends aus Zeitreihen, um Saisonalität und zyklische Muster besser zu erkennen.

**Methoden:**
- Lineare Regression (bei linearem Trend)  
- Differenzenbildung (ARIMA I-Komponente)  
- Moving Average Subtraktion  
- STL-Zerlegung (Trend-Komponente)

**Wann nötig:** Wenn Trend die Saisonalitätserkennung überlagert.

**In PrABCast:** STL-Zerlegung zeigt Trend separat.

**Literatur:** Cleveland et al. (1990).
                """,
                "related": ["STL-Zerlegung", "Stationarität", "Trend"],
            },
            
            "Trend": {
                "category": "Konzepte",
                "full_name": "Langfristige Entwicklung",
                "description": """
**Trend** ist die langfristige Richtung einer Zeitreihe:
- **Aufwärtstrend:** steigend (Wachstum)  
- **Abwärtstrend:** fallend (Rückgang)  
- **Konstant:** kein Trend (stationär bzgl. Mittelwert)

**Trend-Typen:**
- **Linear:** gleichmäßige Steigung  
- **Exponentiell:** beschleunigte/verlangsamte Änderung  
- **Polynomial:** komplexere Kurven

**In PrABCast:** STL-Zerlegung isoliert Trend-Komponente.

**Literatur:** Fundamentales Zeitreihen-Konzept.
                """,
                "related": ["Detrending", "STL-Zerlegung", "Changepoints"],
            },
            
            "Lag-Variablen": {
                "category": "Preprocessing",
                "full_name": "Verzögerte Werte als Features",
                "description": """
**Lag-Variablen** sind vergangene Werte der Zeitreihe als Features:
- **Lag 1:** Wert von gestern/letzter Periode  
- **Lag 12:** Wert von vor 12 Monaten (bei Monatsdaten)

**Verwendung:**
- XGBoost/LSTM: Explizite Lags als Input-Features  
- ARIMA: Implizit in AR-Komponente

**Wichtig:** Datenlecks vermeiden! Nur vergangene Werte verwenden.

**In PrABCast:** Automatisch für XGBoost generiert (Smart Defaults).

**Literatur:** Standard im Feature Engineering für Zeitreihen.
                """,
                "related": ["Feature Engineering", "XGBoost", "ARIMA"],
            },
            
            "Naive Forecast": {
                "category": "Modelle",
                "full_name": "Naive Prognose (Benchmark)",
                "description": """
**Naive Forecast** ist die einfachste Prognose: Letzter Wert wird fortgeschrieben.

`F(t+1) = y(t)`

**Varianten:**
- **Seasonal Naive:** Wert von vor s Perioden (z.B. vor 12 Monaten)  
- **Drift Naive:** Linearer Trend aus letzten Werten

**Verwendung:**
- Benchmark für Modellvergleich  
- Basis für MASE-Berechnung  
- Oft überraschend gut bei random walk-artigen Daten

**In PrABCast:** Als Baseline im Modellvergleich verfügbar.

**Literatur:** Hyndman & Athanasopoulos (2021).
                """,
                "related": ["MASE", "Random Forest", "Benchmark"],
            },
            
            "Random Forest": {
                "category": "ML-Methoden",
                "full_name": "Random Forest für Zeitreihen",
                "description": """
**Random Forest** ist ein Ensemble aus vielen Entscheidungsbäumen (Bagging).

**Für Zeitreihen:**
- Benötigt Feature Engineering (Lags, Rolling Stats, Kalender)  
- Ähnlich wie XGBoost, aber weniger präzise  
- Robuster gegen Overfitting als einzelne Bäume

**Vorteile:** einfacher zu tunen als XGBoost  
**Nachteile:** meist etwas schlechter als XGBoost

**In PrABCast:** XGBoost wird bevorzugt (state-of-the-art).

**Literatur:** Breiman (2001).
                """,
                "related": ["XGBoost", "Ensemble-Methoden", "Feature Engineering"],
            },
            
            "Regularisierung": {
                "category": "ML-Methoden",
                "full_name": "Regularisierung gegen Overfitting",
                "description": """
**Regularisierung** fügt Strafterme hinzu, um Overfitting zu vermeiden:

**L1 (Lasso):** Setzt unwichtige Features auf 0 (Feature Selection)  
**L2 (Ridge):** Verkleinert alle Gewichte (smoother)  
**Elastic Net:** Kombination aus L1 + L2

**In Zeitreihen:**
- LSTM: Dropout, L2 auf Gewichte  
- XGBoost: max_depth, min_child_weight, gamma  
- ARIMA: AIC/BIC bevorzugen einfachere Modelle

+ reduziert Overfitting, verbessert Generalisierung

**Literatur:** Tibshirani (1996) für Lasso, Hoerl & Kennard (1970) für Ridge.
                """,
                "related": ["Overfitting", "XGBoost", "LSTM"],
            },
            
            "Residualanalyse": {
                "category": "Statistische Tests",
                "full_name": "Analyse der Modellresiduen",
                "description": """
**Residuen** sind die Differenzen zwischen Prognose und Ist-Werten.

**Gute Residuen sollten:**
- Mittelwert ≈ 0 (keine systematische Unter-/Überschätzung)  
- Konstante Varianz (Homoskedastizität)  
- Keine Autokorrelation (ACF-Plot flach)  
- Normalverteilt (QQ-Plot)

**Tests:**
- **Ljung-Box-Test:** prüft auf Autokorrelation  
- **Shapiro-Wilk:** prüft auf Normalverteilung

**In PrABCast:** Automatisch im "Statistische Tests"-Tab.

**Literatur:** Box & Jenkins (1970).
                """,
                "related": ["ACF/PACF", "Ljung-Box-Test", "ARIMA"],
            },
            
            "Ljung-Box-Test": {
                "category": "Statistische Tests",
                "full_name": "Test auf Autokorrelation in Residuen",
                "description": """
**Ljung-Box-Test** prüft, ob Residuen unkorreliert sind (White Noise).

**H₀:** Keine Autokorrelation in Residuen  
**p > 0.05:** Residuen sind White Noise + (Modell gut)  
**p < 0.05:** Signifikante Autokorrelation − (Modell verbesserbar)

**Verwendung:** Validierung von ARIMA-Modellen nach dem Fitting.

**In PrABCast:** Teil der automatischen Modelldiagnose.

**Literatur:** Ljung & Box (1978).
                """,
                "related": ["Residualanalyse", "ACF/PACF", "ARIMA"],
            },
            
            "Heteroskedastizität": {
                "category": "Konzepte",
                "full_name": "Nicht-konstante Varianz",
                "description": """
**Heteroskedastizität** bedeutet, dass die Varianz der Zeitreihe über die Zeit variiert.

**Symptom:** Schwankungen werden größer/kleiner über Zeit (Trichterform im Plot).

**Problem:** Verletzt Annahmen von ARIMA, führt zu unrealistischen Konfidenzintervallen.

**Lösung:**
- Log-Transformation (bei proportionaler Zunahme)  
- Box-Cox-Transformation (flexibler)  
- ARCH/GARCH-Modelle (modellieren Varianz explizit)

**In PrABCast:** Smart Defaults erkennen und schlagen Log-Transform vor.

**Literatur:** Engle (1982) für ARCH.
                """,
                "related": ["Box-Cox-Transformation", "Log-Transformation"],
            },
            
            # ========== WEITERE PRAXISBEGRIFFE ==========
            "Box-Cox-Transformation": {
                "category": "Preprocessing",
                "full_name": "Varianz-stabilisierende Transformation",
                "description": """
**Box-Cox-Transformation** stabilisiert die Varianz und macht Daten näherungsweise normalverteilt.

**Formel:** `y' = (y^λ - 1) / λ` für λ ≠ 0

**Spezialfälle:**
- λ = 1: keine Transformation  
- λ = 0.5: Wurzel-Transformation  
- λ = 0: Log-Transformation  
- λ = -1: inverse Transformation

+ automatische Optimierung von λ  
− funktioniert nur bei positiven Werten

**In PrABCast:** Automatisch bei stark heteroskedastischen Daten.

**Literatur:** Box & Cox (1964).
                """,
                "related": ["Log-Transformation", "Heteroskedastizität", "Stationarität"],
            },
            
            "Log-Transformation": {
                "category": "Preprocessing",
                "full_name": "Logarithmische Transformation",
                "description": """
**Log-Transformation** `y' = log(y)` oder `y' = log(y + 1)` für Werte mit 0.

**Vorteile:**
- Stabilisiert Varianz bei proportionalem Wachstum  
- Macht multiplikative Effekte additiv  
- Reduziert Einfluss von Ausreißern

**Verwendung:**
- Bei exponentiellem Wachstum  
- Wenn Varianz mit Level zunimmt  
- Bei stark rechtschiefen Verteilungen

**Wichtig:** Nach Prognose zurücktransformieren (`exp(y')`)!

**In PrABCast:** Smart Defaults schlagen Log-Transform bei Bedarf vor.

**Literatur:** Standard-Technik in der Zeitreihenanalyse.
                """,
                "related": ["Box-Cox-Transformation", "Heteroskedastizität"],
            },
            
            "Exogene Variablen": {
                "category": "Konzepte",
                "full_name": "Externe Einflussfaktoren",
                "description": """
**Exogene Variablen** sind externe Faktoren, die die Zielvariable beeinflussen.

**Beispiele:**
- **Wirtschaft:** BIP, Arbeitslosenquote, Zinsen  
- **Marketing:** Werbeausgaben, Promotions, Preisänderungen  
- **Kalender:** Feiertage, Schulferien, Events  
- **Wetter:** Temperatur, Niederschlag  
- **Konkurrenz:** Marktanteile, neue Produkte

**Modelle mit exogenen Variablen:**
- ARIMAX (ARIMA mit exogenen Variablen)  
- Prophet (Regressoren)  
- XGBoost/LSTM (Features)

**In PrABCast:** Tab "Datenanreicherung" für multivariate Modelle.

**Literatur:** Fundamental für multivariate Prognosen.
                """,
                "related": ["ARIMAX", "XGBoost", "Feature Engineering", "Multivariate Prognose"],
            },
            
            "ARIMAX": {
                "category": "Modelle",
                "full_name": "ARIMA with eXogenous variables",
                "description": """
**ARIMAX** erweitert ARIMA um exogene Variablen (Regressoren).

**Formel:** `y(t) = ARIMA(p,d,q) + β₁×X₁(t) + β₂×X₂(t) + ...`

**Anwendung:**
- Wenn externe Faktoren bekannten Einfluss haben  
- Werbekampagnen, Preisänderungen, Feiertage  
- Wetterdaten, Wirtschaftsindikatoren

+ kombiniert ARIMA-Stärken mit Regression  
− benötigt auch für Prognose zukünftige X-Werte!

**In PrABCast:** Als erweiterte Option in multivariater Prognose.

**Literatur:** Erweiterung von Box & Jenkins (1970).
                """,
                "related": ["ARIMA", "Exogene Variablen", "Regression"],
            },
            
            "GRU": {
                "category": "ML-Methoden",
                "full_name": "Gated Recurrent Unit",
                "description": """
**GRU** ist eine vereinfachte LSTM-Variante mit weniger Parametern.

**Unterschied zu LSTM:**
- Nur 2 Gates statt 3 (Update & Reset vs. Input, Forget, Output)  
- Schneller zu trainieren  
- Oft ähnliche Performance wie LSTM

+ effizienter als LSTM  
+ weniger Overfitting-Gefahr  
− bei sehr langen Sequenzen evtl. schlechter

**Wann verwenden:** Als Alternative zu LSTM bei begrenzten Daten oder Rechenressourcen.

**Literatur:** Cho et al. (2014).
                """,
                "related": ["LSTM", "RNN", "Feature Engineering"],
            },
            
            "RNN": {
                "category": "ML-Methoden",
                "full_name": "Recurrent Neural Network",
                "description": """
**RNN** ist ein neuronales Netz mit Rückkopplungen für Sequenzen.

**Problem:** Vanilla RNN leidet unter **Vanishing/Exploding Gradients** bei langen Sequenzen.

**Lösung:** LSTM und GRU beheben dieses Problem durch Gate-Mechanismen.

**Verwendung heute:** Hauptsächlich als Konzept; praktisch werden LSTM/GRU genutzt.

**Literatur:** Rumelhart et al. (1986).
                """,
                "related": ["LSTM", "GRU", "Vanishing Gradients"],
            },
            
            "Multivariate Prognose": {
                "category": "Konzepte",
                "full_name": "Prognose mit mehreren Variablen",
                "description": """
**Multivariate Prognose** nutzt mehrere Zeitreihen/Features gleichzeitig.

**Ansätze:**
1. **Exogene Variablen:** Y wird durch X₁, X₂, ... erklärt (ARIMAX, Prophet)  
2. **Vector Autoregression (VAR):** Alle Variablen beeinflussen sich gegenseitig  
3. **ML-Methoden:** XGBoost/LSTM mit vielen Features

**Vorteile:**
+ höhere Genauigkeit durch zusätzliche Information  
+ erfasst kausale Zusammenhänge

**Nachteile:**
− benötigt auch für Prognose zukünftige X-Werte  
− komplexer, mehr Daten erforderlich

**In PrABCast:** Tab 3 "Modellvergleich" → Datenanreicherung → Multivariate Modelle

**Literatur:** Lütkepohl (2005) für VAR.
                """,
                "related": ["Exogene Variablen", "ARIMAX", "XGBoost", "VAR"],
            },
            
            "VAR": {
                "category": "Modelle",
                "full_name": "Vector Autoregression",
                "description": """
**VAR** modelliert mehrere Zeitreihen, die sich gegenseitig beeinflussen.

**Beispiel:** Absatz beeinfluss Preis, Preis beeinflusst Absatz (bidirektional).

**Modell:** Jede Variable wird durch ihre eigenen Lags UND die Lags aller anderen Variablen vorhergesagt.

+ erfasst wechselseitige Beziehungen  
− viele Parameter (p × k² bei k Variablen)  
− benötigt lange Zeitreihen

**Verwendung:** Makroökonomie, Finanzmärkte, Supply Chain Netzwerke.

**Literatur:** Sims (1980), Lütkepohl (2005).
                """,
                "related": ["Multivariate Prognose", "ARIMAX", "Granger-Kausalität"],
            },
            
            "Granger-Kausalität": {
                "category": "Statistische Tests",
                "full_name": "Granger Causality Test",
                "description": """
**Granger-Kausalität** testet, ob Zeitreihe X hilfreich ist, um Y vorherzusagen.

**Achtung:** "Kausalität" bedeutet hier nur **prädiktive Verbesserung**, nicht echte Kausalität!

**Interpretation:**
- X "Granger-caused" Y: Vergangene X-Werte helfen bei Y-Prognose  
- Nicht X → Y: X bringt keine zusätzliche Information für Y

**Verwendung:** Feature-Selektion für multivariate Modelle (VAR, ARIMAX).

**In PrABCast:** Automatisch bei Datenanreicherung zur Feature-Auswahl.

**Literatur:** Granger (1969).
                """,
                "related": ["VAR", "Exogene Variablen", "Feature Engineering"],
            },
            
            "Differenzen": {
                "category": "Preprocessing",
                "full_name": "Differenzenbildung",
                "description": """
**Differenzenbildung** transformiert Zeitreihe durch `y'(t) = y(t) - y(t-1)`.

**Zweck:** Entfernen von Trends → Stationarität erreichen.

**Ordnungen:**
- **1. Differenz:** d=1, entfernt linearen Trend  
- **2. Differenz:** d=2, entfernt quadratischen Trend  
- **Saisonale Differenz:** `y(t) - y(t-s)`, entfernt Saisonalität

**Wichtig:** Nach Prognose zurückransformieren (kumulative Summe)!

**In ARIMA:** Der I-Parameter (d) gibt Differenzordnung an.

**Literatur:** Box & Jenkins (1970).
                """,
                "related": ["Stationarität", "ARIMA", "Detrending"],
            },
            
            "Rolling Statistics": {
                "category": "Preprocessing",
                "full_name": "Gleitende Statistiken",
                "description": """
**Rolling Statistics** sind Features aus gleitenden Fenstern:

**Beispiele:**
- **Rolling Mean:** Durchschnitt der letzten k Werte  
- **Rolling Std:** Standardabweichung der letzten k Werte  
- **Rolling Min/Max:** Extremwerte im Fenster  
- **Exponential Moving Average (EMA):** gewichteter Durchschnitt

**Verwendung:**
- Glättung von Zeitreihen (Noise-Reduktion)  
- Features für XGBoost/LSTM  
- Trendindikator

**Typische Fenstergrößen:** 3, 7, 14, 30 Perioden (je nach Datenfrequenz).

**In PrABCast:** Automatisch als Features für XGBoost generiert.

**Literatur:** Standard im Feature Engineering.
                """,
                "related": ["Feature Engineering", "EMA", "XGBoost"],
            },
            
            "EMA": {
                "category": "Preprocessing",
                "full_name": "Exponential Moving Average",
                "description": """
**EMA** ist ein gewichteter gleitender Durchschnitt, der neuere Werte stärker gewichtet.

**Formel:** `EMA(t) = α × y(t) + (1-α) × EMA(t-1)`  
wobei α = 2/(n+1) für n-Perioden-EMA.

**Vorteile vs. Simple Moving Average:**
+ reagiert schneller auf Änderungen  
+ alle historischen Daten fließen ein (exponentiell abnehmend)  
− schwerer zu interpretieren

**Verwendung:**
- Trendglättung (z.B. EMA-12 und EMA-26)  
- Feature für ML-Modelle  
- Basis für MACD-Indikator

**In PrABCast:** Als Rolling-Feature-Variante verfügbar.

**Literatur:** Standard in der technischen Analyse.
                """,
                "related": ["Rolling Statistics", "Feature Engineering", "Glättung"],
            },
            
            "Ausreißer": {
                "category": "Konzepte",
                "full_name": "Outliers / Anomalien",
                "description": """
**Ausreißer** sind ungewöhnlich hohe/niedrige Werte, die vom normalen Muster abweichen.

**Typen:**
- **Additive:** Einzelner extremer Wert (z.B. Messfehler, Sonderaktion)  
- **Level Shift:** Dauerhafter Niveauwechsel  
- **Temporäre Änderung:** Vorübergehende Verschiebung

**Erkennung:**
- Statistische Tests (Z-Score > 3, IQR-Methode)  
- Visuelle Inspektion  
- Isolation Forest, DBSCAN

**Behandlung:**
- **Behalten:** wenn echte Ereignisse (Black Friday)  
- **Entfernen:** bei Messfehlern  
- **Ersetzen:** durch Mittelwert, Median, Interpolation  
- **Modellieren:** Prophet kann Ausreißer robust behandeln

**In PrABCast:** Smart Defaults warnen bei erkannten Ausreißern.

**Literatur:** Hawkins (1980).
                """,
                "related": ["Prophet", "Robustheit", "Residualanalyse"],
            },
            
            "Glättung": {
                "category": "Preprocessing",
                "full_name": "Exponential Smoothing",
                "description": """
**Exponential Smoothing** ist eine Familie von Prognosemethoden basierend auf gewichteten Durchschnitten.

**Varianten:**
- **Simple (SES):** nur Level, keine Trend/Saisonalität  
- **Holt (DES):** Level + Trend  
- **Holt-Winters (TES):** Level + Trend + Saisonalität

**Formel (SES):** `Forecast(t+1) = α × y(t) + (1-α) × Forecast(t)`

+ einfach, schnell, wenig Parameter  
− weniger flexibel als ARIMA/Prophet

**Relation zu ETS:** State Space Framework für Smoothing (Error, Trend, Seasonal).

**Literatur:** Holt (1957), Winters (1960).
                """,
                "related": ["EMA", "ETS", "Holt-Winters"],
            },
            
            "ETS": {
                "category": "Modelle",
                "full_name": "Error, Trend, Seasonal (State Space Models)",
                "description": """
**ETS** ist ein State-Space-Framework für Exponential Smoothing Methoden.

**Komponenten:**
- **E:** Error (additive A oder multiplicative M)  
- **T:** Trend (none N, additive A, damped Ad, multiplicative M)  
- **S:** Seasonal (none N, additive A, multiplicative M)

**Beispiel:** ETS(A,A,N) = additive errors, additive trend, no seasonality.

+ automatische Modellauswahl (AIC)  
+ Konfidenzintervalle verfügbar

**In R:** `forecast::ets()` ist sehr populär.

**Literatur:** Hyndman et al. (2008).
                """,
                "related": ["Glättung", "Holt-Winters", "ARIMA"],
            },
            
            "Benchmark": {
                "category": "Konzepte",
                "full_name": "Baseline-Modelle zum Vergleich",
                "description": """
**Benchmark-Modelle** sind einfache Referenzmodelle, die jedes fortgeschrittene Modell schlagen sollte.

**Typische Benchmarks:**
- **Naive Forecast:** letzter Wert wird fortgeschrieben  
- **Seasonal Naive:** Wert von vor s Perioden  
- **Mean Forecast:** historischer Durchschnitt  
- **Drift:** linearer Trend aus ersten/letzten Werten

**Verwendung:**
- Modellvalidierung (ist ARIMA besser als Naive?)  
- MASE-Berechnung (nutzt Naive als Nenner)  
- Realitätscheck vor Deployment

**In PrABCast:** Naive Forecast als Baseline im Modellvergleich.

**Literatur:** Best Practice in der Prognosevalidierung.
                """,
                "related": ["Naive Forecast", "MASE", "Modellvergleich"],
            },
            
            "Modellvergleich": {
                "category": "Konzepte",
                "full_name": "Model Selection & Comparison",
                "description": """
**Modellvergleich** evaluiert mehrere Modelle auf gleichen Daten, um das beste auszuwählen.

**Kriterien:**
- **Accuracy:** sMAPE, MAE, RMSE auf Test-Set  
- **Komplexität:** AIC, BIC (bevorzugen einfachere Modelle)  
- **Stabilität:** Performance über mehrere CV-Folds  
- **Interpretierbarkeit:** ARIMA vs. LSTM  
- **Rechenzeit:** wichtig für Re-Training

**In PrABCast:** Tab 3 "Modellvergleich" mit automatischer Ranking-Tabelle.

**Best Practice:** Nicht nur Accuracy, auch Robustheit und Interpretierbarkeit berücksichtigen!

**Literatur:** Fundamental in der Modellentwicklung.
                """,
                "related": ["sMAPE", "AIC", "BIC", "Cross-Validation"],
            },
            
            "AIC": {
                "category": "Metriken",
                "full_name": "Akaike Information Criterion",
                "description": """
**AIC** balanciert Modellgüte und Komplexität:

`AIC = 2k - 2×ln(L)`

wobei k = Anzahl Parameter, L = Likelihood.

**Interpretation:** **Kleineres AIC = besseres Modell**

+ bevorzugt einfachere Modelle (bestraft zu viele Parameter)  
+ vergleichbar zwischen verschiedenen Modelltypen  
− sagt nichts über absolute Güte aus (nur relativ)

**Verwendung:** ARIMA-Parameterauswahl (p, d, q).

**Literatur:** Akaike (1974).
                """,
                "related": ["BIC", "Modellvergleich", "Overfitting"],
            },
            
            "BIC": {
                "category": "Metriken",
                "full_name": "Bayesian Information Criterion",
                "description": """
**BIC** ist ähnlich wie AIC, aber bestraft Komplexität stärker:

`BIC = k×ln(n) - 2×ln(L)`

wobei k = Parameter, n = Datenpunkte, L = Likelihood.

**Unterschied zu AIC:**
- BIC bevorzugt einfachere Modelle (stärkere Strafe)  
- AIC besser für Prognose  
- BIC besser für Modellselektion (wahres Modell finden)

**In PrABCast:** Automatisch bei ARIMA-Modellauswahl berechnet.

**Literatur:** Schwarz (1978).
                """,
                "related": ["AIC", "Modellvergleich", "Overfitting"],
            },
        }

        # ---- Filterung ----
        def _passes_category(cat: str) -> bool:
            return (not category_filter) or (cat in category_filter)

        filtered = {}
        for term, data in glossary.items():
            if not _passes_category(data.get("category", "")):
                continue
            if search_term:
                s = search_term.lower()
                blob = " ".join(
                    [
                        term,
                        data.get("full_name", ""),
                        data.get("description", ""),
                        " ".join(data.get("related", [])),
                    ]
                ).lower()
                if s not in blob:
                    continue
            filtered[term] = data

        # ---- Anzeige ----
        if not filtered:
            cat_txt = ", ".join(category_filter) if category_filter else "alle Kategorien"
            wanted = f"'{search_term}' " if search_term else ""
            st.info(f"Keine Begriffe gefunden für {wanted}in {cat_txt}.")
        else:
            st.markdown(f"**{len(filtered)} Begriff(e) gefunden**")
            st.markdown("---")
            for term in sorted(filtered.keys()):
                _render_glossary_term(term, filtered[term])

    # =========================
    # TAB 2: SERVICES
    # =========================
    with main_tabs[1]:
        st.header("RIF – Institut für Forschung und Transfer")
        st.markdown(
            """
Das **RIF Institut für Forschung und Transfer** unterstützt Unternehmen bei der
**Implementierung und Optimierung von Prognosesystemen**. PrABCast wurde in
Kooperation mit dem RIF entwickelt und bietet einen **praxisnahen Einstieg**.
"""
        )
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Schulungen & Workshops")
            st.markdown(
                """
**Inhalte:**
- Grundlagen Zeitreihenanalyse & Modellwahl
- Parameter-Tuning & Interpretation
- Integration in bestehende IT-Systeme

**Format:** 1–2 Tage, online oder in Präsenz  
**Zielgruppe:** Produktionsplanung, SCM, Data/BI-Teams
"""
            )

            st.subheader("Implementierungsunterstützung")
            st.markdown(
                """
- Integration von PrABCast in IT-Landschaften
- ERP/MES-Anbindung & Datenpipelines
- Automatisierte Workflows, Performance-Optimierung
"""
            )

        with col2:
            st.subheader("Erweiterungen & Custom Features")
            st.markdown(
                """
- Branchenspezifische Modelle
- Datenintegration (APIs, DWH)
- Custom Dashboards & Reports
- Multi-User & Rollen/Rechte
"""
            )

            st.subheader("Forschungskooperationen")
            st.markdown(
                """
**Themen:** KI-gestützte Prognosen, nachhaltige Supply Chains, Resilienz.  
**Förderung:** z. B. BMBF, BMWK, EU – wir unterstützen bei Antrag & Konsortium.
"""
            )

        st.markdown("---")
        st.subheader("Kontakt")
        contact_col1, contact_col2 = st.columns([2, 1])
        with contact_col1:
            st.markdown(
                """
**RIF Institut für Forschung und Transfer e. V.**  
Joseph-von-Fraunhofer-Str. 20, 44227 Dortmund  
**Telefon:** +49 (0)231 75896-0  
**E-Mail:** info@rif-ev.de  
**Website:** https://www.rif-ev.de
"""
            )
        with contact_col2:
            if st.button("Anfrage senden", type="primary", use_container_width=True):
                st.info(
                    "Weiterleitung zum Kontaktformular. Bitte Anliegen, Branche und Zeithorizont angeben."
                )
                st.markdown("[→ E-Mail versenden](mailto:info@rif-ev.de)")
