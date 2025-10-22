# PrABCast - Schnellstart

## Installation

1. **Python 3.11+ installieren**
2. **Dependencies installieren:**
   ```bash
   pip install -r requirements.txt
   ```

## Starten

```bash
streamlit run app/layout.py
```

**Öffnet automatisch:** http://localhost:8501

## Verwendung

1. **Daten hochladen** (CSV mit Datum + Produktspalten)
2. **Datumsspalte auswählen**
3. **Produkte auswählen** 
4. **Analysen durchführen** in den verschiedenen Tabs

## Tabs-Übersicht

- **Datenansicht**: CSV-Upload und Konfiguration
- **Absatzanalyse**: Rohdaten, Aggregation, ABC-XYZ, Zeitreihenzerlegung
- **Modellvergleich**: Univariate Prognosemodelle vergleichen
- **Absatzprognose**: Einfache und komplexe Prognosen

## Hauptfunktionen

- 📊 **Interaktive Visualisierungen** mit Plotly
- 🤖 **Machine Learning Modelle** (LSTM, XGBoost, Prophet, ARIMA)
- 📈 **Externe Daten** (Yahoo Finance, FRED API)
- 🎯 **ABC-XYZ Klassifikation**
- 📉 **Zeitreihenzerlegung** (STL)
- 🔍 **Statistische Tests** (ADF, KPSS)

Vollständige Dokumentation: `README.md`