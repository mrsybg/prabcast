"""
Contextual Help System - Intelligente Hilfe und Erklärungen für Produktionsplaner

Dieses Modul bietet:
- Smart Warnings: Kontextuelle Warnungen basierend auf Daten-Charakteristik
- Inline Tooltips: Erklärungen für jeden Parameter
- Glossar-Popups: Klickbare Definitionen für Fachbegriffe
- Parameter-Guidance: "Was bedeutet das und wie wähle ich?"

Zielgruppe: Produktionsplaner ohne tiefe ML/Statistik-Kenntnisse
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class HelpSystem:
    """
    Intelligentes Hilfe-System für kontextuelle Unterstützung
    """
    
    # ============================================================================
    # GLOSSAR - Fachbegriffe einfach erklärt
    # ============================================================================
    
    GLOSSAR = {
        'ARIMA': {
            'name': 'ARIMA',
            'full_name': 'AutoRegressive Integrated Moving Average',
            'simple': 'Ein klassisches statistisches Modell für Zeitreihenprognosen.',
            'detailed': '''
**ARIMA** ist eines der am häufigsten verwendeten Modelle für Zeitreihenprognosen.

**Wie funktioniert es?**
- **AR (AutoRegressive)**: Nutzt vergangene Werte zur Vorhersage
- **I (Integrated)**: Macht die Daten stationär (konstanter Mittelwert)
- **MA (Moving Average)**: Berücksichtigt vergangene Prognosefehler

**Wann verwenden?**
- Für stabile Zeitreihen ohne starke Saisonalität
- Wenn Sie historische Muster fortsetzen möchten
- Bei kleinen bis mittleren Datensätzen (50-500 Punkte)

**Beispiel:** Monatliche Verkaufszahlen eines Produkts ohne saisonale Schwankungen
            ''',
            'category': 'Modell'
        },
        
        'SARIMA': {
            'name': 'SARIMA',
            'full_name': 'Seasonal ARIMA',
            'simple': 'ARIMA erweitert um saisonale Muster (z.B. Weihnachtsgeschäft).',
            'detailed': '''
**SARIMA** ist eine Erweiterung von ARIMA für saisonale Daten.

**Wie funktioniert es?**
- Alles von ARIMA +
- **Saisonale Komponente**: Erkennt wiederkehrende Muster (z.B. jeden Dezember)

**Wann verwenden?**
- Bei klaren saisonalen Mustern (monatlich, quartalsweise, jährlich)
- Wenn Verkäufe zu bestimmten Zeiten regelmäßig steigen/fallen
- Für Branchen mit Saisongeschäft (Einzelhandel, Tourismus)

**Beispiel:** Eisverkauf (hoch im Sommer, niedrig im Winter)
            ''',
            'category': 'Modell'
        },
        
        'Prophet': {
            'name': 'Prophet',
            'full_name': 'Facebook Prophet',
            'simple': 'Ein robustes Modell von Meta, das Trends und Saisonalität automatisch erkennt.',
            'detailed': '''
**Prophet** wurde von Meta (Facebook) entwickelt und ist besonders benutzerfreundlich.

**Wie funktioniert es?**
- Erkennt Trends automatisch (Wachstum/Rückgang)
- Findet saisonale Muster selbstständig
- Robust gegen fehlende Daten und Ausreißer

**Wann verwenden?**
- Bei komplexen saisonalen Mustern
- Wenn Sie wenig Zeit für Parameter-Tuning haben
- Bei Geschäftsdaten mit unregelmäßigen Ereignissen

**Beispiel:** Website-Traffic mit Wochenend-Mustern und Feiertagsspitzen
            ''',
            'category': 'Modell'
        },
        
        'LSTM': {
            'name': 'LSTM',
            'full_name': 'Long Short-Term Memory',
            'simple': 'Ein neuronales Netz für komplexe Zeitreihenmuster.',
            'detailed': '''
**LSTM** ist ein Deep Learning Modell für Zeitreihen.

**Wie funktioniert es?**
- Neuronales Netzwerk das "lernt"
- Erkennt komplexe, nicht-lineare Muster
- Benötigt viele Daten zum Trainieren

**Wann verwenden?**
- Nur bei großen Datensätzen (>100 Punkte)
- Bei sehr komplexen Mustern
- Wenn klassische Modelle versagen

**Wann NICHT verwenden?**
- Bei wenig Daten (<100 Punkte)
- Wenn Sie Ergebnisse erklären müssen
- Bei einfachen Mustern (zu komplex)

**Beispiel:** Stromverbrauch mit komplexen Abhängigkeiten von Wetter, Tageszeit, Feiertagen
            ''',
            'category': 'Modell'
        },
        
        'sMAPE': {
            'name': 'sMAPE',
            'full_name': 'Symmetric Mean Absolute Percentage Error',
            'simple': 'Misst wie genau eine Prognose ist (0% = perfekt, 100% = sehr schlecht).',
            'detailed': '''
**sMAPE** ist eine Fehlermetrik für Prognosen.

**Wie interpretieren?**
- **0-10%**: Ausgezeichnete Prognose
- **10-20%**: Gute Prognose
- **20-50%**: Akzeptable Prognose
- **>50%**: Schlechte Prognose

**Vorteile:**
- Einfach zu verstehen (Prozent)
- Vergleichbar zwischen verschiedenen Produkten
- Symmetrisch (über- und unterschätzen gleichwertig)

**Beispiel:** sMAPE von 15% bedeutet die Prognose weicht im Schnitt um 15% vom echten Wert ab
            ''',
            'category': 'Metrik'
        },
        
        'MAE': {
            'name': 'MAE',
            'full_name': 'Mean Absolute Error',
            'simple': 'Durchschnittliche absolute Abweichung der Prognose.',
            'detailed': '''
**MAE** misst die durchschnittliche Abweichung in absoluten Zahlen.

**Wie interpretieren?**
- Kleinere Werte = bessere Prognose
- In gleicher Einheit wie Ihre Daten (z.B. Stück, Euro)

**Vorteile:**
- Direkt interpretierbar in Ihrer Maßeinheit
- Alle Fehler gleich gewichtet

**Beispiel:** MAE von 100 Stück bedeutet die Prognose weicht im Schnitt um 100 Stück ab
            ''',
            'category': 'Metrik'
        },
        
        'Prognosehorizont': {
            'name': 'Prognosehorizont',
            'full_name': 'Forecast Horizon',
            'simple': 'Wie weit in die Zukunft Sie prognostizieren möchten.',
            'detailed': '''
**Prognosehorizont** bestimmt die Länge Ihrer Prognose.

**Faustregel:**
- Maximal 20-30% Ihrer Datenlänge
- Bei 50 Monaten Daten → max. 10-15 Monate Prognose

**Warum begrenzt?**
- Längere Prognosen = ungenauer
- Unsicherheit wächst mit jedem Schritt

**Empfehlung:**
- **Kurzfristig (1-3 Monate)**: Sehr genau, für Produktionsplanung
- **Mittelfristig (3-12 Monate)**: Gut, für Budgetplanung
- **Langfristig (>12 Monate)**: Unsicher, nur für strategische Richtung

**Beispiel:** 6 Monate Prognose für nächstes Halbjahr-Budget
            ''',
            'category': 'Parameter'
        },
        
        'Saisonalität': {
            'name': 'Saisonalität',
            'full_name': 'Seasonality',
            'simple': 'Wiederkehrende Muster zu bestimmten Zeiten (z.B. jeder Dezember).',
            'detailed': '''
**Saisonalität** sind regelmäßig wiederkehrende Muster.

**Typen:**
- **Monatlich**: Jeden Monat das gleiche Muster
- **Quartalsweise**: Jeden Quartal ähnlich
- **Jährlich**: Gleiche Monate jedes Jahr (z.B. Weihnachten)

**Erkennung:**
- Hohe ACF-Werte bei Lag 12 (monatlich) oder 4 (quartalsweise)
- Visuell: Wiederholendes Zickzack-Muster

**Warum wichtig?**
- Saisonale Modelle (SARIMA, Prophet) können diese Muster nutzen
- Bessere Prognosen für saisonale Produkte

**Beispiel:** Spielzeugverkäufe steigen jeden November/Dezember
            ''',
            'category': 'Konzept'
        }
    }
    
    
    @staticmethod
    def show_glossar_popup(term: str, use_simple: bool = True) -> None:
        """
        Zeigt Glossar-Popup für einen Begriff
        
        Args:
            term: Begriff aus GLOSSAR
            use_simple: True = nur einfache Definition, False = detailliert
        """
        if term not in HelpSystem.GLOSSAR:
            st.warning(f"Begriff '{term}' nicht im Glossar gefunden.")
            return
        
        entry = HelpSystem.GLOSSAR[term]
        
        with st.popover(f"📖 {entry['name']}", use_container_width=True):
            if 'full_name' in entry:
                st.caption(entry['full_name'])
            
            if use_simple:
                st.info(entry['simple'])
            else:
                st.markdown(entry['detailed'])
    
    
    @staticmethod
    def inline_help(text: str, help_text: str, icon: str = "ℹ️") -> None:
        """
        Zeigt inline Help-Icon mit Tooltip
        
        Args:
            text: Angezeigter Text
            help_text: Tooltip-Text
            icon: Icon (default: ℹ️)
        """
        st.markdown(
            f"{text} [{icon}]({help_text})",
            help=help_text
        )
    
    
    # ============================================================================
    # SMART WARNINGS - Kontextuelle Warnungen
    # ============================================================================
    
    @staticmethod
    def check_forecast_horizon(data_length: int, horizon: int) -> Optional[str]:
        """
        Prüft ob Prognosehorizont sinnvoll ist
        
        Args:
            data_length: Anzahl Datenpunkte
            horizon: Gewählter Prognosehorizont
            
        Returns:
            Warning-Text oder None
        """
        ratio = horizon / data_length
        
        if ratio > 0.5:
            return f"⚠️ **Warnung:** Ihr Prognosehorizont ({horizon} Perioden) ist sehr lang im Vergleich zu Ihren Daten ({data_length} Perioden). Empfehlung: Maximal {int(data_length * 0.3)} Perioden für zuverlässige Prognosen."
        
        elif ratio > 0.3:
            return f"💡 **Hinweis:** Ihr Prognosehorizont ({horizon} Perioden) ist relativ lang. Die Prognose wird unsicherer je weiter Sie in die Zukunft schauen."
        
        return None
    
    
    @staticmethod
    def check_model_suitability(model_name: str, 
                                data_length: int, 
                                has_seasonality: bool) -> Optional[str]:
        """
        Prüft ob Modell für die Daten geeignet ist
        
        Args:
            model_name: Name des Modells
            data_length: Anzahl Datenpunkte
            has_seasonality: Ob Saisonalität vorhanden
            
        Returns:
            Warning-Text oder None
        """
        warnings = []
        
        # LSTM/GRU benötigen viele Daten
        if model_name in ['LSTM', 'GRU'] and data_length < 100:
            warnings.append(
                f"⚠️ **{model_name}:** Deep Learning Modelle benötigen normalerweise "
                f"mindestens 100 Datenpunkte. Sie haben {data_length}. "
                f"Das Modell könnte overfitting zeigen oder schlecht performen."
            )
        
        # ARIMA bei Saisonalität
        if model_name == 'ARIMA' and has_seasonality:
            warnings.append(
                f"💡 **ARIMA:** Ihre Daten zeigen Saisonalität, aber ARIMA berücksichtigt "
                f"diese nicht. Erwägen Sie **SARIMA** oder **Prophet** für bessere Ergebnisse."
            )
        
        # Naive bei langen Horizonten
        if 'Naive' in model_name and data_length > 50:
            warnings.append(
                f"💡 **{model_name}:** Naive-Modelle sind nur Baselines für Vergleiche. "
                f"Für echte Prognosen empfehlen wir ARIMA, Prophet oder SARIMA."
            )
        
        return "\n\n".join(warnings) if warnings else None
    
    
    @staticmethod
    def check_data_quality(data: pd.Series) -> List[str]:
        """
        Prüft Datenqualität und gibt Warnungen
        
        Args:
            data: Zeitreihendaten
            
        Returns:
            Liste von Warning-Strings
        """
        warnings = []
        
        # Zu wenig Daten
        if len(data) < 24:
            warnings.append(
                f"⚠️ **Wenig Daten:** Sie haben nur {len(data)} Datenpunkte. "
                f"Für zuverlässige Prognosen empfehlen wir mindestens 24 Monate (2 Jahre) Daten."
            )
        
        # Viele Nullwerte
        zero_pct = (data == 0).sum() / len(data)
        if zero_pct > 0.3:
            warnings.append(
                f"⚠️ **Viele Nullwerte:** {zero_pct*100:.1f}% Ihrer Daten sind Null. "
                f"Das kann Prognosemodelle verwirren. Prüfen Sie ob das korrekt ist."
            )
        
        # Fehlende Werte
        na_pct = data.isna().sum() / len(data)
        if na_pct > 0.1:
            warnings.append(
                f"⚠️ **Fehlende Werte:** {na_pct*100:.1f}% Ihrer Daten fehlen. "
                f"Die Prognose könnte ungenau werden."
            )
        
        # Sehr hohe Variabilität
        cv = data.std() / data.mean() if data.mean() != 0 else 0
        if cv > 1.0:
            warnings.append(
                f"💡 **Hohe Variabilität:** Ihre Daten schwanken stark (CV={cv:.2f}). "
                f"Erwägen Sie XGBoost oder Ensemble-Modelle für volatile Daten."
            )
        
        return warnings
    
    
    # ============================================================================
    # PARAMETER GUIDANCE - Was bedeutet das?
    # ============================================================================
    
    @staticmethod
    def explain_parameter(parameter: str) -> str:
        """
        Erklärt einen Parameter einfach
        
        Args:
            parameter: Name des Parameters
            
        Returns:
            Erklärung
        """
        explanations = {
            'forecast_horizon': '''
**Prognosehorizont** bestimmt wie viele Perioden (z.B. Monate) in die Zukunft Sie prognostizieren.

**Wie wählen?**
- **Kurzfristig (1-3)**: Für nächste Wochen/Monate (sehr genau)
- **Mittelfristig (3-12)**: Für Quartalsplanung (gut)
- **Langfristig (>12)**: Für Jahresplanung (unsicher)

**Faustregel:** Maximal 20% Ihrer Datenlänge
            ''',
            
            'train_test_split': '''
**Train/Test Split** teilt Ihre Daten in zwei Teile:
- **Training (80%)**: Zum "Lernen" des Modells
- **Test (20%)**: Zum Prüfen der Genauigkeit

**Warum wichtig?**
So können wir sehen wie gut das Modell neue Daten vorhersagt.

**Standard:** 80/20 ist bewährte Praxis
            ''',
            
            'model_selection': '''
**Modellauswahl** bestimmt welche Algorithmen verwendet werden.

**Für Einsteiger:**
- Nutzen Sie die empfohlenen Modelle (grün markiert)
- Bei Unsicherheit: ARIMA oder Prophet sind gute Starts

**Für Fortgeschrittene:**
- Wählen Sie mehrere Modelle zum Vergleich
- Das beste Modell wird automatisch ermittelt
            '''
        }
        
        return explanations.get(parameter, "Keine Erklärung verfügbar.")
    
    
    @staticmethod
    def show_help_sidebar(tab_name: str) -> None:
        """
        Zeigt kontextuelle Hilfe im Sidebar basierend auf aktuellem Tab
        
        Args:
            tab_name: Name des aktuellen Tabs
        """
        with st.sidebar:
            st.markdown("---")
            st.subheader("📚 Hilfe & Tipps")
            
            help_texts = {
                'forecast_simple': '''
**Einfache Absatzprognose**

1. Wählen Sie ein Produkt
2. Setzen Sie den Prognosehorizont (nutzen Sie empfohlenen Wert)
3. Wählen Sie ein Modell (grün = empfohlen)
4. Klicken Sie "Prognose erstellen"

**Tipps:**
- Klicken Sie auf 🤖 für intelligente Empfehlungen
- Nutzen Sie ARIMA oder Prophet als Start
- Bei Saisonalität: SARIMA bevorzugen
                ''',
                
                'forecast_multi': '''
**Modellvergleich**

Vergleichen Sie mehrere Modelle gleichzeitig:

1. Wählen Sie 2-4 Modelle
2. Die Modelle werden automatisch verglichen
3. Beste Modell wird empfohlen

**Tipps:**
- Mehr Modelle = längere Rechenzeit
- Nutzen Sie empfohlene Modelle (grün)
- sMAPE zeigt welches Modell am besten ist
                ''',
                
                'upload': '''
**Daten hochladen**

1. CSV-Datei auswählen
2. Datumsspalte bestimmen
3. Produkte auswählen

**Anforderungen:**
- CSV-Format
- Datumsspalte vorhanden
- Numerische Absatzdaten
                '''
            }
            
            if tab_name in help_texts:
                with st.expander("💡 Anleitung für diesen Tab", expanded=False):
                    st.markdown(help_texts[tab_name])


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def show_smart_warning(warning_text: str, warning_type: str = "warning") -> None:
    """
    Zeigt Smart Warning mit passendem Icon
    
    Args:
        warning_text: Warning-Text
        warning_type: 'warning', 'info', 'error'
    """
    if warning_type == "warning":
        st.warning(warning_text)
    elif warning_type == "info":
        st.info(warning_text)
    elif warning_type == "error":
        st.error(warning_text)


def show_parameter_help(parameter_name: str, current_value: any, data_context: Dict = None) -> None:
    """
    Zeigt kontextuelle Hilfe für einen Parameter
    
    Args:
        parameter_name: Name des Parameters
        current_value: Aktueller Wert
        data_context: Optional - Context (data_length, has_seasonality, etc.)
    """
    help_system = HelpSystem()
    
    # Zeige Erklärung
    explanation = help_system.explain_parameter(parameter_name)
    
    with st.expander(f"❓ Was ist '{parameter_name}'?", expanded=False):
        st.markdown(explanation)
        st.caption(f"Aktueller Wert: **{current_value}**")
        
        # Kontextuelle Warnung wenn verfügbar
        if data_context and parameter_name == 'forecast_horizon':
            warning = help_system.check_forecast_horizon(
                data_context.get('data_length', 100),
                current_value
            )
            if warning:
                st.warning(warning)
