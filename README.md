# PrABCast - AI-gestützte Absatz- und Bedarfsprognose

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)

> **Forschungsprojekt** des [RIF Institut für Forschung und Transfer](https://www.rif-ev.de) in Kooperation mit dem [Institut für Produktionssysteme (IPS)](https://ips.mb.tu-dortmund.de) der TU Dortmund.

---

## ⚠️ Demonstrator-Hinweis

**Diese Anwendung ist ein wissenschaftlicher Demonstrator** zur Veranschaulichung moderner Prognoseverfahren und deren praktischer Anwendbarkeit in der Produktion und im Supply Chain Management.

**Vereinfachungen für Benutzerfreundlichkeit:**
- Modellparametrisierung basiert auf **heuristischen Standardwerten**
- Hyperparameter-Optimierung ist bewusst vereinfacht
- Fokus liegt auf intuitiver Bedienbarkeit und schnellen Ergebnissen

**Für produktive Anwendungen** bietet das RIF Institut vollständige Implementierungen mit:
- Umfassender Hyperparameter-Optimierung (Grid Search, Bayesian Optimization)
- Erweiterter Kreuzvalidierung und Modellselektion
- Domänenspezifischem Feature Engineering
- Maßgeschneiderten Prognoselösungen

📧 **Kontakt:** [info@rif-ev.de](mailto:info@rif-ev.de)

---

## 🎯 Überblick

**PrABCast** unterstützt Unternehmen dabei, Maschinelle Lernverfahren in der Absatz- und Bedarfsprognose einzusetzen. Die Anwendung kombiniert klassische statistische Verfahren mit modernen ML-Algorithmen in einer interaktiven Streamlit-Oberfläche.

### Hauptfunktionen

- **📊 ABC/XYZ-Analyse** – Produktklassifikation nach Wert und Variabilität
- **📈 Univariate Prognosen** – ARIMA, SARIMA, Prophet, LSTM, XGBoost
- **🔗 Multivariate Modelle** – Integration externer Einflussfaktoren (Wirtschaftsdaten, Indizes)
- **📉 Statistische Analysen** – Stationaritätstests, Zeitreihenzerlegung, Korrelationen
- **🎨 Interaktive Visualisierungen** – Plotly-basierte Dashboards
- **📁 Datenhandling** – CSV-Import/Export, flexible Aggregation

---

## 🚀 Schnellstart

### Installation

```bash
# Repository klonen
git clone https://github.com/mrsybg/prabcast.git
cd prabcast

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencies installieren
pip install -r requirements.txt

# Anwendung starten
streamlit run app/run.py
```

Die Anwendung öffnet sich automatisch unter `http://localhost:8501`

### Externe APIs (optional)

Für multivariate Prognosen mit Wirtschaftsdaten:

1. Erstelle eine `.env` Datei im Hauptverzeichnis
2. Füge deinen [FRED API-Key](https://fred.stlouisfed.org/docs/api/api_key.html) hinzu:
   ```
   FRED_API_KEY=your_key_here
   ```

Siehe [SETUP.md](SETUP.md) für detaillierte Installationsanweisungen.

---

## 📚 Dokumentation

- **[SETUP.md](SETUP.md)** – Ausführliche Installationsanleitung
- **[QUICKSTART.md](QUICKSTART.md)** – 5-Minuten-Schnelleinstieg
- **[GITHUB_SECURITY_UPDATE.md](GITHUB_SECURITY_UPDATE.md)** – Sicherheitsmaßnahmen und API-Key-Externalisierung

---

## 🏗️ Architektur

```
prabcast/
├── app/
│   ├── layout.py          # Haupt-UI und Tab-Struktur
│   ├── run.py             # Einstiegspunkt
│   ├── models.py          # Univariate Modelle
│   └── models_multi.py    # Multivariate Modelle
├── tabs/
│   ├── upload.py          # Datenimport
│   ├── forecast.py        # Modellvergleich
│   ├── advanced_forecast.py  # Datenanreicherung
│   ├── multivariate_forecast.py  # Multivariate Prognosen
│   └── glossar.py         # Fachbegriffe-Glossar
├── setup_module/
│   ├── helpers.py         # Zentrale Hilfsfunktionen
│   ├── evaluation.py      # Metriken (sMAPE, MAE, RMSE)
│   └── model_registry.py  # Modellverwaltung
└── templates/
    └── custom_data_template.csv  # CSV-Vorlage
```

---

## 📄 Lizenz

Dieses Projekt ist unter der [MIT License](LICENSE) lizenziert.

---

## 🔗 Links

- **Projektwebsite:** [IPS Forschungsprojekte - PrABCast](https://ips.mb.tu-dortmund.de/forschen-beraten/forschungsprojekte/prabcast/)
- **RIF Institut:** [www.rif-ev.de](https://www.rif-ev.de)
- **Kontakt:** [info@rif-ev.de](mailto:info@rif-ev.de)

---

## 🙏 Danksagungen

Entwickelt im Rahmen eines Forschungsprojekts in Zusammenarbeit mit:
- RIF Institut für Forschung und Transfer e.V.
- Institut für Produktionssysteme (IPS), TU Dortmund
- Gefördert durch das BMWE (IGF)

---

*Letzte Aktualisierung: Oktober 2025 | Version 2.0*