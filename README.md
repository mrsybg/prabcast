# PrABCast - Umfassendes Prognose- und ABC-Analyse System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)

> Ein Forschungsprojekt des [RIF Institut für Forschung und Transfer](https://www.rif-ev.de) in Kooperation mit dem [Institut für Produktionssysteme (IPS)](https://ips.mb.tu-dortmund.de) an der TU Dortmund.

## Projektübersicht

**PrABCast** ist ein Forschungsprojekt des RIF - Instituts für Forschung und Transfer in Kooperation mit dem Institut für Produktionssysteme (IPS) an der Technischen Universität Dortmund. Das System ermöglicht es Unternehmen, Maschinelle Lernverfahren in der Absatz- und Bedarfsprognose einzusetzen und dabei sowohl univariate als auch multivariate Prognosemodelle zu nutzen.

### Kernfunktionen
- **ABC/XYZ-Analyse** zur Produktklassifikation
- **Univariate Zeitreihenprognose** mit klassischen und modernen Algorithmen
- **Multivariate Prognosemodelle** mit externen Einflussfaktoren
- **Statistische Analysen** und Zeitreihenzerlegung
- **Interaktive Visualisierungen** für alle Analysen
- **Datenimport/Export** mit CSV-Unterstützung

---

## Anwendungsarchitektur

### Systemübersicht
Die Anwendung basiert auf **Streamlit** und ist als modulare Web-Applikation aufgebaut. Alle Berechnungen erfolgen lokal ohne externe Cloud-Services.

### Hauptkomponenten

#### 1. **Core Application** (`app/`)
- **`layout.py`**: Haupteinstiegspunkt und UI-Layout
- **`run.py`**: Vereinfachter Launcher für die Anwendung
- **`models.py`**: Univariate Prognosemodelle
- **`models_multi.py`**: Multivariate Prognosemodelle

#### 2. **Setup Module** (`setup_module/`)
- **`helpers.py`**: Zentrale Hilfsfunktionen und Session State Management
- **`evaluation.py`**: Metriken-Berechnungen und Performance-Messungen

#### 3. **Tab-Module** (`tabs/`)
- **`upload.py`**: Datenimport und -konfiguration
- **`rohdaten.py`**: Rohdatenanalyse und Kennwerte
- **`aggregation.py`**: Datenaggregatierung
- **`produktverteilung.py`**: Produktverteilungsanalyse
- **`zerlegung.py`**: STL-Zeitreihenzerlegung
- **`abcxyz.py`**: ABC-XYZ-Klassifikation
- **`statistische_tests.py`**: ADF und KPSS Tests
- **`forecast.py`**: Univariate Prognosemodelle
- **`advanced_forecast.py`**: Datenanreicherung und Korrelationsanalyse
- **`multivariate_forecast.py`**: Multivariate Prognosemodelle
- **`forecast_simple.py`**: Vereinfachte Prognosefunktionen
- **`forecast_complex.py`**: Komplexe Prognosen mit gespeicherten Modellen
- **`glossar.py`**: Begriffserklärungen

#### 4. **Advanced Module** (`tabs/advanced/`)
- **`questionnaire.py`**: Benutzereingaben für Datenanreicherung
- **`api_fetch.py`**: Yahoo Finance und FRED API Integration
- **`analysis.py`**: Korrelations- und Kausalitätsanalysen
- **`visualization.py`**: Erweiterte Visualisierungen

---

## Datenfluss und Session State Management

### Session State Variablen

Die Anwendung nutzt Streamlit's Session State für persistente Datenhaltung zwischen Tab-Wechseln:

```python
# Basis-Datenstrukturen
st.session_state.df                     # Hauptdatensatz (DataFrame)
st.session_state.date_column           # Ausgewählte Datumsspalte
st.session_state.selected_products_in_data  # Ausgewählte Produktspalten
st.session_state.start_date            # Globales Startdatum
st.session_state.end_date              # Globales Enddatum

# Verarbeitungsstatus
st.session_state.ready_for_processing  # Boolean: Daten bereit für Analyse

# Erweiterte Daten (für multivariate Prognosen)
st.session_state.multivariate_data     # Angereicherte Daten mit externen Faktoren

# Modell-Persistierung
st.session_state.saved_models_for_complex  # Gespeicherte trainierte Modelle
st.session_state.forecast_complex_results  # Prognoseergebnisse
```

### Datenfluss zwischen Tabs

```
1. UPLOAD TAB
   ├── CSV-Import → st.session_state.df
   ├── Datumsspalte wählen → st.session_state.date_column
   ├── Produkte wählen → st.session_state.selected_products_in_data
   └── ready_for_processing = True

2. ABSATZANALYSE TABS
   ├── Verwenden st.session_state.df
   ├── Nutzen selected_products_in_data
   └── Arbeiten mit gefilterten Datumsbereichen

3. MODELLVERGLEICH (FORECAST TAB)
   ├── Univariate Modelle mit st.session_state.df
   └── Performance-Vergleiche

4. DATENANREICHERUNG (ADVANCED_FORECAST TAB)
   ├── Lädt externe Daten (Yahoo Finance, FRED)
   ├── Führt Korrelationsanalysen durch
   └── Speichert → st.session_state.multivariate_data

5. MULTIVARIATE PROGNOSE TAB
   ├── Nutzt st.session_state.multivariate_data
   ├── Trainiert komplexe Modelle (LSTM, XGBoost)
   └── Speichert Modelle → st.session_state.saved_models_for_complex

6. KOMPLEXE PROGNOSE TAB
   ├── Nutzt gespeicherte Modelle
   └── Generiert finale Prognosen
```

---

## Implementierte Algorithmen und Modelle

### Univariate Zeitreihenmodelle (`app/models.py`)

#### 1. **Statistische Modelle**
```python
class ARIMAModel:
    # Autoregressive Integrated Moving Average
    # Parameter: order=(p,d,q)
    # Geeignet für: Trend- und AR-Komponenten

class SARIMAXModel:
    # Seasonal ARIMA with eXogenous variables
    # Parameter: order=(p,d,q), seasonal_order=(P,D,Q,s)
    # Geeignet für: Saisonale Zeitreihen

class ProphetModel:
    # Facebook Prophet - Additive Zeitreihenmodell
    # Automatische Trend- und Saisonalitätserkennung
    # Geeignet für: Geschäftsdaten mit starker Saisonalität
```

#### 2. **Glättungsverfahren**
```python
class SimpleExpSmoothing:
    # Einfache exponentielle Glättung
    # Geeignet für: Daten ohne Trend/Saisonalität

class ExponentialSmoothing:
    # Holt-Winters Verfahren
    # Berücksichtigt Trend und Saisonalität
    # Geeignet für: Komplexe saisonale Muster
```

#### 3. **Naive Basismethoden**
```python
class SeasonalNaiveModel:
    # Wiederholung der letzten Saison
    # Parameter: season_length=12
    # Geeignet für: Baseline-Vergleiche

class MovingAverageModel:
    # Gleitender Durchschnitt
    # Parameter: window=12
    # Geeignet für: Einfache Trend-Schätzung
```

#### 4. **Machine Learning Modelle**
```python
class LSTMModel:
    # Long Short-Term Memory Neural Network
    # Automatische Feature-Extraktion
    # Geeignet für: Komplexe nichtlineare Muster

class GRUModel:
    # Gated Recurrent Unit
    # Effizientere Alternative zu LSTM
    # Geeignet für: Mittlere Komplexität

class XGBoostModel:
    # Gradient Boosting Decision Trees
    # Feature Engineering erforderlich
    # Geeignet für: Strukturierte Zeitreihendaten

class RandomForestModel:
    # Ensemble von Entscheidungsbäumen
    # Robuste Vorhersagen
    # Geeignet für: Rauschige Daten
```

### Multivariate Modelle (`app/models_multi.py`)

#### 1. **Deep Learning Architekturen**
```python
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    # Geeignet für: Multivariate Zeitreihen mit komplexen Abhängigkeiten
```

#### 2. **Gradient Boosting**
```python
def build_xgboost_model():
    return XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8
    )
    # Geeignet für: Strukturierte multivariate Daten
```

### Evaluationsmetriken (`setup_module/evaluation.py`)

```python
# Prognosegüte-Metriken
- sMAPE (Symmetric Mean Absolute Percentage Error)
- MAE (Mean Absolute Error) 
- RMSE (Root Mean Square Error)
- MASE (Mean Absolute Scaled Error)

# Performance-Metriken
- Trainingszeit (Sekunden)
- Speicherverbrauch (MB)
- CPU-Auslastung
```

---

## Tab-spezifische Funktionalitäten

### 1. **Upload Tab** (`tabs/upload.py`)
```python
Funktionen:
- CSV-Datei-Import mit automatischer Encoding-Erkennung
- Flexible Datumserkennung (DD.MM.YYYY Format)
- Multi-Produkt-Auswahl
- Datumsbereich-Definition
- Datenvalidierung und -preprocessing

Session State Updates:
- st.session_state.df
- st.session_state.date_column
- st.session_state.selected_products_in_data
- st.session_state.ready_for_processing
```

### 2. **Rohdaten Tab** (`tabs/rohdaten.py`)
```python
Funktionen:
- Deskriptive Statistiken (count, mean, std, min, max, quartile)
- Zeitreihenvisualisierung mit Plotly
- Flexibler Datumsfilter
- Aggregationsoptionen (monatlich, quartalsweise, jährlich)
- Deutsche Zahlenformatierung

Visualisierungen:
- Liniendiagramme für Zeitreihen
- Interaktive Zoom- und Pan-Funktionen
```

### 3. **Aggregation Tab** (`tabs/aggregation.py`)
```python
Funktionen:
- Multi-Level-Aggregation (Monat/Quartal/Jahr)
- Gesamtabsatz-Berechnung über alle Produkte
- Vergleichende Visualisierungen
- Flexible Datumsfilterung

Aggregationsmethoden:
- Summe (für Absatzmengen)
- Durchschnitt (für normalisierte Vergleiche)
```

### 4. **ABC-XYZ Analyse** (`tabs/abcxyz.py`)
```python
Klassifikationskriterien:
- ABC: Basierend auf kumulativem Anteil am Gesamtabsatz
  - A: 0-80% (hoher Wert)
  - B: 80-95% (mittlerer Wert)  
  - C: 95-100% (niedriger Wert)

- XYZ: Basierend auf Variationskoeffizient
  - X: CV < 22% (konstante Nachfrage)
  - Y: 22% ≤ CV < 50% (schwankende Nachfrage)
  - Z: CV ≥ 50% (unregelmäßige Nachfrage)

Ausgabe:
- 9-Felder-Matrix Visualisierung
- Produktklassifikation mit Empfehlungen
- Export der Klassifikationsergebnisse
```

### 5. **STL-Zerlegung** (`tabs/zerlegung.py`)
```python
STL-Dekomposition (Seasonal-Trend decomposition using Loess):
- Trendkomponente: Langfristige Entwicklung
- Saisonkomponente: Wiederkehrende Muster
- Restkomponente: Unerklärbarer Rest und Ausreißer

Parameter:
- Automatische Saisonperioden-Erkennung
- Robuste Loess-Glättung
- Interaktive 4-Panel-Visualisierung
```

### 6. **Statistische Tests** (`tabs/statistische_tests.py`)
```python
Implementierte Tests:
- ADF (Augmented Dickey-Fuller): Stationaritätstest
- KPSS (Kwiatkowski-Phillips-Schmidt-Shin): Trendstationarität

Interpretation:
- p-Werte und Teststatistiken
- Kritische Werte für verschiedene Signifikanzniveaus
- Automatische Stationaritätsbewertung
```

### 7. **Datenanreicherung** (`tabs/advanced_forecast.py`)
```python
Externe Datenquellen:
- Yahoo Finance API: Aktienindizes, Rohstoffpreise
- FRED API: Wirtschaftsindikatoren, Zinssätze
- Custom Data Upload: Benutzerdefinierte Zeitreihen

Analysemethoden:
- Pearson-Korrelation
- Granger-Kausalitätstests  
- Lead-Lag-Analyse
- Cross-Korrelationsfunktionen

Datenverarbeitung:
- Automatische Datumsalignment
- Missing-Data-Interpolation
- Normalisierung und Skalierung
```

### 8. **Multivariate Prognose** (`tabs/multivariate_forecast.py`)
```python
Modell-Pipeline:
1. Datenvorverarbeitung:
   - MinMaxScaler für Normalisierung
   - Sequenz-Generierung für LSTM
   - Feature-Engineering für XGBoost

2. Modelltraining:
   - LSTM: Sequential processing mit Dropout
   - XGBoost: Gradient boosting mit Hyperparameter-Tuning
   - Cross-Validation für Robustheit

3. Prognose-Generierung:
   - Multi-Step-Ahead Forecasting
   - Uncertainty Quantification
   - Konfidenzintervalle

4. Modell-Persistierung:
   - Serialisierung trainierter Modelle
   - Scaler-State-Speicherung
   - Metadaten für Reproduzierbarkeit
```

### 9. **Komplexe Prognose** (`tabs/forecast_complex.py`)
```python
Funktionen:
- Verwendung gespeicherter Modelle aus multivariate_forecast
- Batch-Prognosen für mehrere Produkte
- Ensemble-Methoden (Durchschnitt mehrerer Modelle)
- Automatische Modell-Selection basierend auf Validierungsmetriken

Workflow:
1. Lade gespeicherte Modelle und Scaler
2. Bereite neue Eingabedaten vor
3. Generiere Prognosen mit Unsicherheitsschätzung
4. Aggregiere Ensemble-Ergebnisse
5. Visualisiere und exportiere Resultate
```

---

## API-Integrationen

### Yahoo Finance Integration (`tabs/advanced/api_fetch.py`)
```python
Verfügbare Datenquellen:
- Aktienindizes: S&P 500, DAX, Nikkei, etc.
- Rohstoffe: Öl, Gold, Silber, Kupfer
- Währungen: EUR/USD, GBP/USD, JPY/USD
- Kryptowährungen: Bitcoin, Ethereum

Datenformate:
- OHLC (Open, High, Low, Close)
- Volumen
- Adjustierte Schlusskurse
- Historische Daten bis zu 10 Jahre
```

### FRED API Integration
```python
Wirtschaftsindikatoren:
- GDP Growth Rate
- Inflation Rate (CPI)
- Unemployment Rate
- Interest Rates (Federal Funds Rate)
- Industrial Production Index
- Consumer Confidence Index

Abruf-Parameter:
- Automatische Datumsbereich-Anpassung
- Verschiedene Frequenzen (täglich, monatlich, quartalsweise)
- Missing-Value-Behandlung
```

---

## Performance-Optimierungen

### Caching-Strategien
```python
@st.cache_data
def load_and_process_data():
    # Caching für schwere Datenoperationen
    
@st.cache_resource  
def train_model():
    # Modell-Training nur bei Datenänderungen
```

### Memory Management
```python
# Automatische Garbage Collection nach schweren Berechnungen
import gc
gc.collect()

# Selective data loading für große Datensätze
chunk_size = 10000
for chunk in pd.read_csv(file, chunksize=chunk_size):
    process_chunk(chunk)
```

### Computational Efficiency
```python
# Vectorized operations mit NumPy/Pandas
data.rolling(window=12).mean()  # statt expliziter Schleifen

# Parallelisierung bei Modell-Training
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    results = executor.map(train_model, model_configs)
```

---

## Fehlerbehandlung und Robustheit

### Input Validation
```python
def validate_data(df):
    """Comprehensive data validation"""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isnull().sum().sum() > len(df) * 0.3:
        st.warning("Data contains >30% missing values")
    
    return df
```

### Graceful Degradation
```python
try:
    advanced_model_result = train_lstm_model(data)
except (MemoryError, TimeoutError):
    st.warning("Falling back to simpler model")
    fallback_result = train_simple_model(data)
```

### User Feedback
```python
# Progress bars für lange Operationen
progress_bar = st.progress(0)
for i, step in enumerate(training_steps):
    execute_step(step)
    progress_bar.progress((i + 1) / len(training_steps))

# Informative Fehlermeldungen
except ValueError as e:
    st.error(f"Data validation failed: {str(e)}")
    st.info("Please check your data format and try again")
```

---

## Entwicklungsrichtlinien

### Code-Organisation
```
Modularer Aufbau:
- Ein Tab = Ein Modul
- Getrennte Modell-Definitionen
- Zentrale Helper-Funktionen
- Klare Abhängigkeitsstruktur
```

### Testing-Strategie
```python
# Unit Tests für Modelle
def test_arima_model():
    model = ARIMAModel(order=(1,1,1))
    data = generate_test_data()
    model.fit(data)
    forecast = model.predict(steps=12)
    assert len(forecast) == 12

# Integration Tests für Datenfluss
def test_upload_to_forecast_pipeline():
    # Vollständiger Workflow-Test
    pass
```

### Deployment-Konfiguration
```python
# Streamlit Config (.streamlit/config.toml)
[server]
maxUploadSize = 200
enableCORS = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

---

## Installation und Entwicklungsumgebung

### Systemanforderungen
```
- Python 3.11+
- RAM: 4GB minimum, 8GB empfohlen
- Festplatte: 500MB für Abhängigkeiten
- Internet: Für API-Calls und Package-Installation
```

### Development Setup
```bash
# 1. Repository klonen
git clone <repository-url>
cd PrABCast

# 2. Virtual Environment erstellen
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# 3. Dependencies installieren
pip install -r requirements.txt

# 4. Anwendung starten
streamlit run app/layout.py
```

### Produktions-Deployment
```bash
# Docker-basierte Bereitstellung
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8501
CMD ["streamlit", "run", "app/layout.py"]
```

---

## Troubleshooting

### Häufige Probleme

#### 1. **Python/Streamlit Installation**
```bash
# Problem: streamlit command not found
# Lösung: Direkter Python-Aufruf
python -m streamlit run app/layout.py

# Problem: Package-Konflikte
# Lösung: Clean installation
pip uninstall streamlit
pip install streamlit==1.40.1
```

#### 2. **Datenimport-Probleme**
```python
# Problem: CSV encoding errors
# Lösung: Automatische Encoding-Erkennung in upload.py

# Problem: Datumserkennung fehlschlägt
# Lösung: Manuelle Format-Spezifikation
pd.to_datetime(df['date'], format='%d.%m.%Y')
```

#### 3. **Memory-Issues bei großen Datensätzen**
```python
# Problem: MemoryError bei LSTM-Training
# Lösung: Batch-Processing und Sampling
sample_data = data.sample(n=min(10000, len(data)))
```

#### 4. **API-Verbindungsfehler**
```python
# Problem: Yahoo Finance Timeout
# Lösung: Retry-Mechanismus mit Backoff
import time
for attempt in range(3):
    try:
        data = yf.download(ticker)
        break
    except:
        time.sleep(2 ** attempt)
```

---

## Ausblick und Erweiterungen

### Geplante Features
1. **Erweiterte ML-Modelle**: Transformer, LSTM-Attention
2. **Automatisches Feature Engineering**: Lag-Features, Rolling Statistics
3. **Model Explainability**: SHAP Values, Feature Importance
4. **Real-time Monitoring**: Live-Datenfeeds, Alerting
5. **Multi-User Support**: Benutzerverwaltung, Projekt-Spaces

### Technische Verbesserungen
1. **Performance**: Distributed Computing, GPU-Acceleration  
2. **Scalability**: Database Integration, Cloud Deployment
3. **Robustness**: Enhanced Error Handling, Automated Testing
4. **Usability**: Guided Workflows, Interactive Tutorials

---

## Kontakt und Support

**Entwicklungsteam:**
- RIF - Institut für Forschung und Transfer
- Institut für Produktionssysteme (IPS), TU Dortmund

**Technische Unterstützung:**
- GitHub Issues für Bug-Reports
- Dokumentation und Wiki für Benutzeranleitungen
- Community Forum für Diskussionen

**Projektwebsite:** [IPS Forschungsprojekte](https://ips.mb.tu-dortmund.de/forschen-beraten/forschungsprojekte/prabcast/)

---

*Letzte Aktualisierung: Oktober 2024*
*Version: 2.0*