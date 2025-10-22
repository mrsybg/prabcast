# Demo-Datensatz: TechParts GmbH

## Übersicht

Dieser Ordner enthält einen **fiktiven Demonstrationsdatensatz** der "TechParts GmbH", eines imaginären Herstellers von technischen Komponenten.

## Datensatz-Details

**Datei:** `demo_data_techparts.csv`

- **Zeitraum:** Januar 2021 - Dezember 2023 (36 Monate)
- **Produkte:** 10 verschiedene technische Komponenten
- **Frequenz:** Monatliche Absatzdaten
- **Werte:** Verkaufsmengen im Bereich 100-15.000 Einheiten

## Produkt-Charakteristika

Die 10 Produkte wurden mit unterschiedlichen Mustern generiert, um verschiedene ABC/XYZ-Klassifikationen zu demonstrieren:

| Produkt | Muster | ABC | XYZ | Beschreibung |
|---------|--------|-----|-----|--------------|
| **Dichtungsset DS-150** | Growing Seasonal | A | Y | Höchster Umsatz, wachsender Trend, saisonal |
| **Präzisionslager PL-200** | Declining Seasonal | A | Y | Hoher Umsatz, rückläufiger Trend, saisonal |
| **Dichtring DR-25** | Medium Seasonal | B | Y | Mittlerer Umsatz, moderate Saisonalität |
| **Hydraulikpumpe HP-3000** | Stable Seasonal | A | X | Hoher Umsatz, stabil, vorhersagbar |
| **Kupplungselement KE-120** | Moderate | B | Y | Mittlerer Umsatz, moderate Schwankungen |
| **Servomotor SM-450** | Stable Growth | B | X | Mittlerer Umsatz, stabiles Wachstum |
| **Steuerventil SV-75** | High Variance | B | Z | Mittlerer Umsatz, hohe Variabilität |
| **Sensoreinheit SE-400** | Emerging | C | X | Niedriger Umsatz, aber wachsend |
| **Schmiermittel SM-5L** | Low Stable | C | X | Niedriger Umsatz, stabil |
| **Schraubenset SS-M8** | Low Volatile | C | Z | Niedriger Umsatz, sehr volatil |

## Verwendung

### In PrABCast laden

1. Starte PrABCast: `streamlit run app/run.py`
2. Gehe zum **Upload Tab**
3. Lade `demo_data_techparts.csv` hoch
4. Wähle **Datum** als Datumsspalte
5. Wähle beliebige Produkte aus

### Empfohlene Analysen

- **ABC/XYZ-Analyse:** Zeigt alle 9 Klassifikationen
- **STL-Zerlegung:** Besonders interessant bei Dichtungsset DS-150 (starke Saisonalität)
- **Statistische Tests:** Vergleiche Stationarität zwischen stabilen und volatilen Produkten
- **Prognose-Modellvergleich:** Teste verschiedene Modelle auf unterschiedlichen Produkten
- **Multivariate Prognose:** Reichere mit Wirtschaftsdaten an (FRED API)

## Datengenerierung

Die Daten wurden mit dem Script `generate_demo_data.py` erstellt:

```bash
python sample_data/generate_demo_data.py
```

Jedes Produkt folgt einem mathematischen Modell:

```
Absatz(t) = Basis + Trend × t + Saisonalität × sin(2πt/12) + Rauschen
```

**Parameter-Beispiel (Hydraulikpumpe HP-3000):**
- Basis: 5.000 Einheiten
- Trend: +50 Einheiten/Monat
- Saisonalität: ±800 Einheiten (Amplitude)
- Rauschen: ±300 Einheiten (Standardabweichung)

## Hinweis

⚠️ **Dies ist ein fiktiver Datensatz** für Demonstrations- und Testzwecke. Die Daten sind synthetisch generiert und stellen keine realen Verkaufszahlen dar.

---

**Generiert am:** Oktober 2024  
**Seed:** 42 (für Reproduzierbarkeit)
