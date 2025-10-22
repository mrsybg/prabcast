#!/usr/bin/env python3
"""
Generiert Demo-Datensatz für TechParts GmbH
10 Produkte mit unterschiedlichen Absatzmustern für ABC/XYZ-Analyse
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Seed für Reproduzierbarkeit
np.random.seed(42)

# Datumsbereich: 3 Jahre Monatsdaten (Jan 2021 - Dez 2023)
start_date = datetime(2021, 1, 1)
dates = pd.date_range(start=start_date, periods=36, freq='MS')

# 10 Produkte mit unterschiedlichen Mustern
products = {
    'Hydraulikpumpe HP-3000': {
        'base': 5000, 'trend': 50, 'seasonality': 800, 'noise': 300,
        'pattern': 'stable_seasonal'  # ABC-A, XYZ-X: Hoher Wert, stabil
    },
    'Präzisionslager PL-200': {
        'base': 8500, 'trend': -30, 'seasonality': 1200, 'noise': 400,
        'pattern': 'declining_seasonal'  # ABC-A, XYZ-Y: Hoher Wert, schwankend
    },
    'Dichtungsset DS-150': {
        'base': 12000, 'trend': 100, 'seasonality': 2000, 'noise': 500,
        'pattern': 'growing_seasonal'  # ABC-A, XYZ-Y: Höchster Wert, wachsend
    },
    'Servomotor SM-450': {
        'base': 3200, 'trend': 80, 'seasonality': 500, 'noise': 250,
        'pattern': 'stable_growth'  # ABC-B, XYZ-X: Mittlerer Wert, stabil wachsend
    },
    'Steuerventil SV-75': {
        'base': 2800, 'trend': 0, 'seasonality': 900, 'noise': 600,
        'pattern': 'high_variance'  # ABC-B, XYZ-Z: Mittlerer Wert, sehr variabel
    },
    'Kupplungselement KE-120': {
        'base': 4500, 'trend': 20, 'seasonality': 600, 'noise': 350,
        'pattern': 'moderate'  # ABC-B, XYZ-Y: Mittlerer Wert, moderat schwankend
    },
    'Schmiermittel SM-5L': {
        'base': 1500, 'trend': 15, 'seasonality': 200, 'noise': 150,
        'pattern': 'low_stable'  # ABC-C, XYZ-X: Niedriger Wert, stabil
    },
    'Schraubenset SS-M8': {
        'base': 800, 'trend': -10, 'seasonality': 150, 'noise': 200,
        'pattern': 'low_volatile'  # ABC-C, XYZ-Z: Niedriger Wert, volatil
    },
    'Dichtring DR-25': {
        'base': 6000, 'trend': 40, 'seasonality': 800, 'noise': 400,
        'pattern': 'medium_seasonal'  # ABC-B, XYZ-Y: Mittlerer Wert, saisonal
    },
    'Sensoreinheit SE-400': {
        'base': 2000, 'trend': 60, 'seasonality': 400, 'noise': 250,
        'pattern': 'emerging'  # ABC-C, XYZ-X: Niedriger Wert, aber wachsend
    }
}

# Datensatz generieren
data = {'Datum': dates}

for product, params in products.items():
    values = []
    for i, date in enumerate(dates):
        # Trend
        trend_component = params['trend'] * i
        
        # Saisonalität (12-Monats-Zyklus)
        seasonal_component = params['seasonality'] * np.sin(2 * np.pi * i / 12)
        
        # Zufälliges Rauschen
        noise_component = np.random.normal(0, params['noise'])
        
        # Gesamtwert (immer positiv)
        value = max(100, params['base'] + trend_component + seasonal_component + noise_component)
        values.append(int(value))
    
    data[product] = values

# DataFrame erstellen
df = pd.DataFrame(data)

# Datum formatieren
df['Datum'] = df['Datum'].dt.strftime('%d.%m.%Y')

# Als CSV speichern
output_file = 'demo_data_techparts.csv'
df.to_csv(output_file, index=False, sep=',', encoding='utf-8')

print(f'Demo-Datensatz erstellt: {output_file}')
print(f'Produkte: {len(products)}')
print(f'Zeitraum: 36 Monate (Jan 2021 - Dez 2023)')
print(f'\nProdukt-Charakteristika:')
for product, params in products.items():
    print(f'  - {product}: {params["pattern"]}')
