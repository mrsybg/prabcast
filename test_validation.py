"""
Test-Skript für Datenvalidierung.

Dieses Skript testet die DataValidator-Klasse mit verschiedenen Test-Cases:
- Valide Daten
- Leere Dateien
- Fehlende Werte
- Ungültige Datumsangaben
- Produktspalten mit Problemen
"""

import pandas as pd
import numpy as np
from setup_module.data_validator import DataValidator, ValidationResult, DataQualityReport


def test_valid_data():
    """Test mit validen Daten."""
    print("\n" + "=" * 80)
    print("TEST 1: Valide Daten")
    print("=" * 80)
    
    df = pd.DataFrame({
        'Datum': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Produkt_A': np.random.randint(10, 100, 100),
        'Produkt_B': np.random.randint(20, 150, 100)
    })
    
    # Upload-Validierung
    result = DataValidator.validate_uploaded_file(df, "valid_data.csv")
    print(f"\n✅ Upload-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    
    # Datumsspalten-Validierung
    result = DataValidator.validate_date_column(df, 'Datum')
    print(f"\n✅ Datums-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    if result.metadata.get('date_range'):
        print(f"   Zeitraum: {result.metadata['date_range']}")
    
    # Produktspalten-Validierung
    result = DataValidator.validate_product_columns(df, ['Produkt_A', 'Produkt_B'])
    print(f"\n✅ Produkt-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    
    # Quality Report
    report = DataValidator.assess_data_quality(df, 'Datum', ['Produkt_A', 'Produkt_B'])
    print(f"\n📊 Quality Score: {report.quality_score:.1f}%")
    print(f"   Zeilen: {report.total_rows}")
    print(f"   Spalten: {report.total_columns}")


def test_empty_file():
    """Test mit leerer Datei."""
    print("\n" + "=" * 80)
    print("TEST 2: Leere Datei")
    print("=" * 80)
    
    df = pd.DataFrame()
    
    result = DataValidator.validate_uploaded_file(df, "empty.csv")
    print(f"\n❌ Upload-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    print(f"   Errors: {result.errors}")


def test_insufficient_data():
    """Test mit zu wenig Daten."""
    print("\n" + "=" * 80)
    print("TEST 3: Zu wenig Daten (< 10 Zeilen)")
    print("=" * 80)
    
    df = pd.DataFrame({
        'Datum': pd.date_range('2023-01-01', periods=5, freq='D'),
        'Produkt_A': [10, 20, 30, 40, 50]
    })
    
    result = DataValidator.validate_uploaded_file(df, "small.csv")
    print(f"\n❌ Upload-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    print(f"   Errors: {result.errors}")


def test_missing_values():
    """Test mit vielen fehlenden Werten."""
    print("\n" + "=" * 80)
    print("TEST 4: Viele fehlende Werte (50% NaN)")
    print("=" * 80)
    
    data = np.random.randint(10, 100, 100).astype(float)  # Float für NaN
    data[::2] = np.nan  # Jeder zweite Wert = NaN
    
    df = pd.DataFrame({
        'Datum': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Produkt_A': data
    })
    
    result = DataValidator.validate_uploaded_file(df, "many_nans.csv")
    print(f"\n⚠️  Upload-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    print(f"   Errors: {result.errors}")
    print(f"   Warnings: {result.warnings}")
    
    # Produkt-Validierung
    result = DataValidator.validate_product_columns(df, ['Produkt_A'])
    print(f"\n⚠️  Produkt-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    print(f"   Warnings: {result.warnings}")
    print(f"   NaN%: {result.metadata['nan_percentages']}")


def test_invalid_dates():
    """Test mit ungültigen Datumsangaben."""
    print("\n" + "=" * 80)
    print("TEST 5: Ungültige Datumsangaben")
    print("=" * 80)
    
    df = pd.DataFrame({
        'Datum': ['2023-01-01', 'invalid', '2023-01-03', 'not-a-date', '2023-01-05'] * 20,
        'Produkt_A': np.random.randint(10, 100, 100)
    })
    
    result = DataValidator.validate_date_column(df, 'Datum')
    print(f"\n⚠️  Datums-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    print(f"   Warnings: {result.warnings}")
    print(f"   NaT Count: {result.metadata.get('nat_count', 0)}")


def test_negative_values():
    """Test mit negativen Werten."""
    print("\n" + "=" * 80)
    print("TEST 6: Negative Werte (Retouren?)")
    print("=" * 80)
    
    df = pd.DataFrame({
        'Datum': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Produkt_A': np.random.randint(-20, 100, 100)
    })
    
    result = DataValidator.validate_product_columns(df, ['Produkt_A'], allow_negatives=False)
    print(f"\n⚠️  Produkt-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    print(f"   Warnings: {result.warnings}")
    print(f"   Negative Werte: {result.metadata['negative_values']}")


def test_no_variance():
    """Test mit konstanten Werten."""
    print("\n" + "=" * 80)
    print("TEST 7: Keine Varianz (konstante Werte)")
    print("=" * 80)
    
    df = pd.DataFrame({
        'Datum': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Produkt_A': [50] * 100  # Alle Werte gleich
    })
    
    result = DataValidator.validate_product_columns(df, ['Produkt_A'])
    print(f"\n⚠️  Produkt-Validierung: {'OK' if result.is_valid else 'FEHLER'}")
    print(f"   Warnings: {result.warnings}")


def test_forecast_data():
    """Test Forecast-Daten-Validierung."""
    print("\n" + "=" * 80)
    print("TEST 8: Forecast-Daten-Validierung")
    print("=" * 80)
    
    # Valide Daten
    data = pd.Series(np.random.randint(10, 100, 50), index=pd.date_range('2023-01-01', periods=50))
    
    result = DataValidator.validate_forecast_data(data, forecast_horizon=12)
    print(f"\n✅ Forecast-Validierung (H=12): {'OK' if result.is_valid else 'FEHLER'}")
    print(f"   Datenpunkte: {result.metadata['data_points']}")
    
    # Horizont zu groß
    try:
        result = DataValidator.validate_forecast_data(data, forecast_horizon=60)
        print(f"\n❌ Forecast-Validierung (H=60): {'OK' if result.is_valid else 'FEHLER'}")
        print(f"   Errors: {result.errors}")
    except Exception as e:
        print(f"\n❌ Forecast-Validierung (H=60): Exception wie erwartet")
        print(f"   {type(e).__name__}: {str(e)}")


def test_quality_report():
    """Test umfassender Quality Report."""
    print("\n" + "=" * 80)
    print("TEST 9: Umfassender Quality Report")
    print("=" * 80)
    
    # Erstelle realistische Daten mit verschiedenen Problemen
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Produkt A: Gut
    product_a = np.random.randint(50, 150, 365)
    
    # Produkt B: Mit NaN
    product_b = np.random.randint(30, 100, 365).astype(float)
    product_b[::10] = np.nan  # 10% NaN
    
    # Produkt C: Mit Outliers
    product_c = np.random.randint(40, 80, 365)
    product_c[100] = 1000  # Extremer Outlier
    product_c[200] = 2000
    
    df = pd.DataFrame({
        'Datum': dates,
        'Produkt_A': product_a,
        'Produkt_B': product_b,
        'Produkt_C': product_c
    })
    
    report = DataValidator.assess_data_quality(
        df, 
        date_col='Datum', 
        product_cols=['Produkt_A', 'Produkt_B', 'Produkt_C']
    )
    
    print(f"\n📊 Quality Report:")
    print(f"   Score: {report.quality_score:.1f}%")
    print(f"   Zeilen: {report.total_rows}")
    print(f"   Zeitraum: {report.date_range[0].date()} bis {report.date_range[1].date()}")
    print(f"\n   Missing Values:")
    for col, pct in report.missing_values_pct.items():
        print(f"      {col}: {pct:.2f}%")
    print(f"\n   Outliers:")
    for col, count in report.outliers_count.items():
        print(f"      {col}: {count}")
    print(f"\n   Varianz:")
    for col, has_var in report.variance_check.items():
        print(f"      {col}: {'✅ Ja' if has_var else '❌ Nein'}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" DATENVALIDIERUNG - TEST SUITE")
    print("=" * 80)
    
    test_valid_data()
    test_empty_file()
    test_insufficient_data()
    test_missing_values()
    test_invalid_dates()
    test_negative_values()
    test_no_variance()
    test_forecast_data()
    test_quality_report()
    
    print("\n" + "=" * 80)
    print(" ALLE TESTS ABGESCHLOSSEN")
    print("=" * 80 + "\n")
