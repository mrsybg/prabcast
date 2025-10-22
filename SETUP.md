# 🚀 PrABCast Setup-Anleitung

## Schnellstart (5 Minuten)

```bash
# 1. Repository klonen
git clone <repository-url>
cd PrABCast

# 2. Python Virtual Environment erstellen
python -m venv venv

# 3. Environment aktivieren
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Dependencies installieren
pip install -r requirements.txt

# 5. Environment-Konfiguration erstellen
copy .env.example .env  # Windows
cp .env.example .env    # Linux/macOS

# 6. FRED API-Key eintragen (siehe unten)
# Editiere .env und füge deinen Key ein

# 7. App starten
streamlit run app/run.py
```

Die App öffnet sich automatisch im Browser unter `http://localhost:8501`

---

## 📋 Detaillierte Installations-Schritte

### 1. System-Voraussetzungen

**Erforderlich:**
- **Python 3.11 oder höher** ([Download](https://www.python.org/downloads/))
- **4GB RAM** (8GB empfohlen für große Datensätze)
- **500MB freier Festplattenspeicher** (für Dependencies)
- **Internetverbindung** (für API-Zugriff und Package-Installation)

**Optional:**
- **Git** ([Download](https://git-scm.com/downloads))
- **VS Code** oder anderer Code-Editor

### 2. Repository herunterladen

**Mit Git:**
```bash
git clone <repository-url>
cd PrABCast
```

**Ohne Git:**
1. Lade ZIP von GitHub herunter
2. Entpacke das Archiv
3. Öffne Terminal/CMD im entpackten Ordner

### 3. Virtual Environment einrichten

**Warum Virtual Environment?**
- Isoliert Dependencies von anderen Python-Projekten
- Verhindert Versions-Konflikte
- Ermöglicht saubere Deinstallation

**Erstellen:**
```bash
python -m venv venv
```

**Aktivieren:**
```bash
# Windows PowerShell:
venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# Linux/macOS:
source venv/bin/activate
```

**Erfolgreich?** Dein Terminal sollte jetzt `(venv)` am Anfang zeigen:
```
(venv) C:\Users\...\PrABCast>
```

### 4. Dependencies installieren

```bash
pip install -r requirements.txt
```

**⏱️ Dauer:** 5-10 Minuten (je nach Internetgeschwindigkeit)

**Bei Fehlern:**
```bash
# Option 1: pip aktualisieren
python -m pip install --upgrade pip

# Option 2: Einzelne Packages manuell installieren
pip install streamlit pandas numpy scikit-learn prophet xgboost

# Option 3: Ohne Caching (bei Netzwerkproblemen)
pip install --no-cache-dir -r requirements.txt
```

### 5. Environment-Konfiguration (.env)

**Erstelle .env aus Template:**
```bash
# Windows:
copy .env.example .env

# Linux/macOS:
cp .env.example .env
```

**Editiere .env:**
Öffne `.env` mit einem Text-Editor und füge deinen FRED API-Key ein:

```bash
# .env Datei
FRED_API_KEY=dein_api_key_hier
```

---

## 🔑 FRED API-Key besorgen (KOSTENLOS)

Die FRED API (Federal Reserve Economic Data) wird für multivariate Prognosen mit Wirtschaftsdaten benötigt.

### Schritt-für-Schritt:

1. **Registrierung:**
   - Gehe zu: https://fred.stlouisfed.org/
   - Klicke auf **"Sign In"** (oben rechts)
   - Wähle **"Create Account"**

2. **Account erstellen:**
   - E-Mail-Adresse eingeben
   - Passwort festlegen (mindestens 8 Zeichen)
   - Bestätigungs-E-Mail bestätigen

3. **API-Key anfordern:**
   - Nach Login: Gehe zu **"My Account"** (oben rechts)
   - Klicke auf **"API Keys"** im Seitenmenü
   - Klicke **"Request API Key"**
   - Akzeptiere die Terms of Use
   - Klicke **"Request API Key"** nochmal

4. **Key kopieren:**
   - Dein API-Key wird angezeigt (Format: 32-stelliger Hex-String)
   - **Wichtig:** Kopiere den Key sofort (wird nur einmal angezeigt!)

5. **In .env einfügen:**
   ```bash
   FRED_API_KEY=your_actual_key_here
   ```

**💡 Hinweis:** Der API-Key ist kostenlos und hat ein tägliches Limit von 120 Requests/Minute.

---

## 🎯 App starten

### Standard-Methode:
```bash
streamlit run app/run.py
```

### Alternative Methoden:

**Mit benutzerdefiniertem Port:**
```bash
streamlit run app/run.py --server.port 8502
```

**Im Entwicklungsmodus (Auto-Reload):**
```bash
streamlit run app/run.py --server.runOnSave true
```

**Nur Netzwerkzugriff (kein Browser-Auto-Open):**
```bash
streamlit run app/run.py --server.headless true
```

### App öffnen:

Nach dem Start öffnet sich automatisch dein Browser unter:
- **Lokal:** http://localhost:8501
- **Netzwerk:** http://192.168.x.x:8501 (für andere Geräte im gleichen Netzwerk)

---

## ✅ Installations-Check

### Testen Sie die Installation:

**1. Python-Version prüfen:**
```bash
python --version
# Sollte zeigen: Python 3.11.x oder höher
```

**2. Streamlit-Version prüfen:**
```bash
streamlit version
# Sollte zeigen: Streamlit, version 1.40.1
```

**3. Environment-Variablen prüfen:**
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('FRED_API_KEY:', 'OK' if os.getenv('FRED_API_KEY') else 'FEHLT')"
```

**4. Test-Daten laden:**
- Starte die App
- Gehe zum **"📊 Daten"** Tab
- Lade `taegliche_absatzdaten.csv` (im Projektordner)
- Prüfe ob Daten korrekt angezeigt werden

---

## 🐛 Häufige Probleme & Lösungen

### Problem: "streamlit: command not found"

**Ursache:** Streamlit nicht im PATH oder Virtual Environment nicht aktiviert

**Lösung:**
```bash
# 1. Virtual Environment aktivieren (siehe oben)
# 2. Direkt mit Python aufrufen:
python -m streamlit run app/run.py
```

---

### Problem: "ImportError: No module named 'dotenv'"

**Ursache:** python-dotenv nicht installiert

**Lösung:**
```bash
pip install python-dotenv
```

---

### Problem: "ValueError: FRED_API_KEY nicht gefunden"

**Ursache:** .env Datei existiert nicht oder ist leer

**Lösung:**
```bash
# 1. Prüfe ob .env existiert:
ls -la .env  # Linux/macOS
dir .env     # Windows

# 2. Erstelle aus Template:
cp .env.example .env

# 3. Editiere .env und füge API-Key ein
```

---

### Problem: "ModuleNotFoundError: No module named 'tensorflow'"

**Ursache:** TensorFlow nicht korrekt installiert (häufig auf macOS M1/M2)

**Lösung:**
```bash
# Windows/Linux:
pip install tensorflow==2.16.2

# macOS M1/M2:
pip install tensorflow-macos==2.16.2
```

---

### Problem: App lädt sehr langsam oder friert ein

**Ursachen & Lösungen:**

**1. Zu großer Datensatz:**
```python
# Reduziere Datenmenge in Upload-Tab:
# Wähle kürzeren Datumsbereich oder weniger Produkte
```

**2. Modell-Training dauert zu lange:**
```python
# Nutze einfachere Modelle (ARIMA statt LSTM)
# Reduziere Epochen für Neural Networks
```

**3. Zu wenig RAM:**
```bash
# Schließe andere Programme
# Nutze kleinere Batch-Größen
```

---

### Problem: "PermissionError" beim Installieren

**Ursache:** Fehlende Schreibrechte

**Lösung:**
```bash
# Nicht empfohlen aber funktioniert:
pip install --user -r requirements.txt

# Besser: Virtual Environment nutzen (siehe oben)
```

---

## 🔧 Entwicklungs-Setup

### Zusätzliche Dev-Dependencies:

```bash
# Code-Formatierung
pip install black isort

# Linting
pip install pylint flake8

# Testing
pip install pytest pytest-cov

# Type Checking
pip install mypy
```

### VS Code Empfohlene Extensions:

- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Streamlit** (streamlit.streamlit)
- **GitLens** (eamodio.gitlens)

### Pre-Commit Hooks (optional):

```bash
pip install pre-commit
pre-commit install
```

---

## 🌍 Deployment

### Lokales Netzwerk (andere Geräte):

```bash
streamlit run app/run.py --server.address 0.0.0.0
```

Zugriff von anderen Geräten:
```
http://<deine-ip-adresse>:8501
```

### Streamlit Cloud (kostenlos):

1. Push Code zu GitHub
2. Gehe zu https://share.streamlit.io/
3. Verbinde GitHub-Repository
4. Füge Secrets hinzu (FRED_API_KEY)
5. Deploy!

**Secrets in Streamlit Cloud:**
```toml
# .streamlit/secrets.toml
FRED_API_KEY = "dein_key_hier"
```

---

## 📚 Weitere Ressourcen

- **Streamlit Dokumentation:** https://docs.streamlit.io/
- **FRED API Docs:** https://fred.stlouisfed.org/docs/api/
- **Prophet Guide:** https://facebook.github.io/prophet/
- **PrABCast GitHub:** <repository-url>

---

## 🤝 Support

Bei Problemen:

1. **Check FAQ oben** (häufigste Probleme)
2. **GitHub Issues:** <repository-url>/issues
3. **E-Mail:** info@rif-ev.de

---

**🎉 Viel Erfolg mit PrABCast!**

*Wenn alles funktioniert, können Sie mit dem Tutorial im "📖 Was ist PrABCast?"-Tab starten.*
