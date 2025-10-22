"""
Zentrale Datenvalidierung für PrABCast.

Dieses Modul stellt Validierungsfunktionen für Upload-Daten, Datumsspalten,
Produktspalten und Forecast-Daten bereit. Ziel ist es, Fehler früh zu erkennen
und dem User klare, actionable Fehlermeldungen zu geben.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
from setup_module.exceptions import DataValidationError, InsufficientDataError


# ============================================================================
# DATEN-KLASSEN
# ============================================================================

@dataclass
class ValidationResult:
    """
    Ergebnis einer Validierung.
    
    Attributes:
        is_valid: True wenn alle kritischen Checks bestanden
        errors: Liste von blocking errors
        warnings: Liste von non-blocking issues
        info: Liste von informativen Nachrichten
        metadata: Zusätzliche Informationen (z.B. erkannte Zeiträume)
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def raise_if_invalid(self):
        """Wirft DataValidationError wenn nicht valid."""
        if not self.is_valid:
            raise DataValidationError(
                message="Datenvalidierung fehlgeschlagen",
                validation_errors={
                    "errors": self.errors,
                    "warnings": self.warnings
                }
            )
    
    def display_in_streamlit(self):
        """Zeigt Validierungs-Ergebnisse in Streamlit UI."""
        if self.errors:
            for error in self.errors:
                st.error(f"❌ {error}")
        if self.warnings:
            for warning in self.warnings:
                st.warning(f"⚠️ {warning}")
        if self.info:
            for info_msg in self.info:
                st.info(f"ℹ️ {info_msg}")


@dataclass
class DataQualityReport:
    """
    Daten-Qualitäts-Bericht.
    
    Attributes:
        total_rows: Anzahl Zeilen
        total_columns: Anzahl Spalten
        date_range: Zeitraum (start, end)
        missing_values_pct: NaN-Prozentsatz pro Spalte
        outliers_count: Anzahl Ausreißer pro Spalte
        variance_check: Varianz vorhanden pro Spalte
        time_gaps: Liste von Zeitlücken
        seasonality_detected: Erkannte Saisonalität (Periode) oder None
        quality_score: Gesamt-Qualitätsscore 0-100
    """
    total_rows: int
    total_columns: int
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    missing_values_pct: Dict[str, float] = field(default_factory=dict)
    outliers_count: Dict[str, int] = field(default_factory=dict)
    variance_check: Dict[str, bool] = field(default_factory=dict)
    time_gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    seasonality_detected: Optional[int] = None
    quality_score: float = 0.0
    
    def display_in_streamlit(self):
        """Zeigt Report in Streamlit mit Metrics."""
        st.subheader("📊 Daten-Qualität")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Zeilen", self.total_rows)
        with col2:
            st.metric("Spalten", self.total_columns)
        with col3:
            st.metric("Qualität", f"{self.quality_score:.1f}%")
        with col4:
            if self.missing_values_pct:
                missing_avg = sum(self.missing_values_pct.values()) / len(self.missing_values_pct)
                st.metric("NaN Ø", f"{missing_avg:.1f}%")
            else:
                st.metric("NaN Ø", "0%")
        
        # Details in Expander
        with st.expander("📋 Qualitäts-Details"):
            # Zeitraum
            if self.date_range:
                st.write(f"**Zeitraum:** {self.date_range[0].date()} bis {self.date_range[1].date()}")
                days = (self.date_range[1] - self.date_range[0]).days
                st.write(f"**Tage:** {days}")
            
            # Fehlende Werte
            if self.missing_values_pct:
                st.write("**Fehlende Werte:**")
                missing_df = pd.DataFrame({
                    "Spalte": list(self.missing_values_pct.keys()),
                    "NaN %": [f"{v:.2f}%" for v in self.missing_values_pct.values()]
                })
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            
            # Ausreißer
            if self.outliers_count:
                st.write("**Ausreißer (>3 Std.Dev.):**")
                outlier_df = pd.DataFrame({
                    "Spalte": list(self.outliers_count.keys()),
                    "Anzahl": list(self.outliers_count.values())
                })
                st.dataframe(outlier_df, use_container_width=True, hide_index=True)
            
            # Varianz
            if self.variance_check:
                no_variance = [col for col, has_var in self.variance_check.items() if not has_var]
                if no_variance:
                    st.write("**Keine Varianz:**")
                    st.write(", ".join(no_variance))
            
            # Zeitlücken
            if self.time_gaps:
                st.write(f"**Zeitlücken:** {len(self.time_gaps)} erkannt")
                for start, end in self.time_gaps[:5]:  # Max 5 anzeigen
                    st.write(f"  - {start.date()} bis {end.date()}")
            
            # Saisonalität
            if self.seasonality_detected:
                st.write(f"**Saisonalität:** Periode {self.seasonality_detected} erkannt")


# ============================================================================
# VALIDATOR KLASSE
# ============================================================================

class DataValidator:
    """Zentrale Datenvalidierung für PrABCast."""
    
    # Konstanten
    MIN_ROWS = 10
    MIN_COLUMNS = 2
    MAX_NAN_PERCENT = 90
    MAX_WARNING_NAN_PERCENT = 50
    MIN_FORECAST_DATA_POINTS = 10
    MAX_OUTLIER_THRESHOLD = 3  # Std.Dev.
    MIN_VARIANCE_THRESHOLD = 0.01
    
    @staticmethod
    def validate_uploaded_file(df: pd.DataFrame, filename: str) -> ValidationResult:
        """
        Validiert hochgeladene CSV-Dateien.
        
        Args:
            df: Der hochgeladene DataFrame
            filename: Name der Datei
            
        Returns:
            ValidationResult mit Fehler/Warnungen
            
        Checks:
            - Nicht leer
            - Mindestens MIN_ROWS Zeilen
            - Mindestens MIN_COLUMNS Spalten
            - Keine komplett leeren Spalten
            - Nicht alle Werte NaN
        """
        errors = []
        warnings = []
        info = []
        metadata = {}
        
        # Check: DataFrame leer
        if df.empty:
            errors.append(f"Datei '{filename}' ist leer - keine Daten vorhanden")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check: Zu wenig Zeilen
        if len(df) < DataValidator.MIN_ROWS:
            errors.append(
                f"Zu wenig Zeilen: {len(df)} Zeilen vorhanden, "
                f"mindestens {DataValidator.MIN_ROWS} erforderlich"
            )
        
        # Check: Zu wenig Spalten
        if len(df.columns) < DataValidator.MIN_COLUMNS:
            errors.append(
                f"Zu wenig Spalten: {len(df.columns)} Spalten vorhanden, "
                f"mindestens {DataValidator.MIN_COLUMNS} erforderlich (Datum + min. 1 Produkt)"
            )
        
        # Check: Komplett leere Spalten
        completely_empty = []
        for col in df.columns:
            if df[col].isna().all():
                completely_empty.append(col)
        
        if completely_empty:
            warnings.append(
                f"Komplett leere Spalten gefunden: {', '.join(completely_empty)}"
            )
        
        # Check: Alle Werte NaN
        total_cells = len(df) * len(df.columns)
        nan_cells = df.isna().sum().sum()
        nan_percent = (nan_cells / total_cells * 100) if total_cells > 0 else 100
        
        if nan_percent >= DataValidator.MAX_NAN_PERCENT:
            errors.append(
                f"Fast keine validen Daten: {nan_percent:.1f}% fehlende Werte"
            )
        elif nan_percent >= DataValidator.MAX_WARNING_NAN_PERCENT:
            warnings.append(
                f"Viele fehlende Werte: {nan_percent:.1f}% NaN"
            )
        
        # Info
        info.append(f"Datei '{filename}' geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
        
        # Metadata
        metadata['filename'] = filename
        metadata['rows'] = len(df)
        metadata['columns'] = len(df.columns)
        metadata['nan_percent'] = nan_percent
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            metadata=metadata
        )
    
    @staticmethod
    def validate_date_column(df: pd.DataFrame, date_col: str) -> ValidationResult:
        """
        Validiert Datumsspalte.
        
        Args:
            df: Der DataFrame
            date_col: Name der Datumsspalte
            
        Returns:
            ValidationResult mit Fehler/Warnungen
            
        Checks:
            - Spalte existiert
            - Kann zu datetime geparst werden
            - Nicht alle NaT
            - Plausible Range (1900-2100)
            - Chronologisch sortiert (Warning)
            - Duplikate (Warning)
        """
        errors = []
        warnings = []
        info = []
        metadata = {}
        
        # Check: Spalte existiert
        if date_col not in df.columns:
            errors.append(f"Datumsspalte '{date_col}' existiert nicht")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check: Kann zu datetime konvertiert werden
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce')
        except Exception as e:
            errors.append(f"Kann Spalte '{date_col}' nicht zu Datum konvertieren: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check: Alle NaT
        if dates.isna().all():
            errors.append(f"Keine validen Datumsangaben in Spalte '{date_col}'")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check: NaT Prozentsatz
        nat_count = dates.isna().sum()
        nat_percent = (nat_count / len(dates) * 100) if len(dates) > 0 else 0
        
        if nat_percent > 10:
            warnings.append(
                f"{nat_percent:.1f}% ungültige Datumsangaben in '{date_col}' ({nat_count} von {len(dates)})"
            )
        
        # Nur valide Daten für weitere Checks
        valid_dates = dates.dropna()
        
        if len(valid_dates) == 0:
            errors.append(f"Keine validen Datumsangaben nach Bereinigung")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check: Plausible Range
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        
        if min_date.year < 1900 or max_date.year > 2100:
            warnings.append(
                f"Unplausible Datumsangaben: {min_date.date()} bis {max_date.date()}"
            )
        
        # Check: Chronologisch sortiert
        if not valid_dates.is_monotonic_increasing:
            warnings.append(
                f"Datumsangaben nicht chronologisch sortiert - bitte sortieren"
            )
        
        # Check: Duplikate
        duplicates = valid_dates.duplicated().sum()
        if duplicates > 0:
            warnings.append(
                f"{duplicates} doppelte Datumsangaben gefunden"
            )
        
        # Info
        info.append(
            f"Datumsspalte '{date_col}' validiert: {min_date.date()} bis {max_date.date()}"
        )
        
        # Metadata
        metadata['date_column'] = date_col
        metadata['date_range'] = (min_date, max_date)
        metadata['valid_dates'] = len(valid_dates)
        metadata['nat_count'] = nat_count
        metadata['duplicates'] = duplicates
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            metadata=metadata
        )
    
    @staticmethod
    def validate_product_columns(
        df: pd.DataFrame,
        product_cols: List[str],
        allow_negatives: bool = False
    ) -> ValidationResult:
        """
        Validiert Produktspalten.
        
        Args:
            df: Der DataFrame
            product_cols: Liste der Produktspalten
            allow_negatives: Negative Werte erlauben
            
        Returns:
            ValidationResult mit Fehler/Warnungen
            
        Checks:
            - Spalten existieren
            - Numerische Werte
            - Nicht alle NaN
            - Nicht alle Null
            - Keine negativen Werte (optional)
            - Max NaN% pro Spalte
            - Varianz vorhanden
        """
        errors = []
        warnings = []
        info = []
        metadata = {
            'nan_percentages': {},
            'zero_columns': [],
            'negative_values': {},
            'no_variance': []
        }
        
        if not product_cols:
            errors.append("Keine Produktspalten ausgewählt")
            return ValidationResult(is_valid=False, errors=errors)
        
        for col in product_cols:
            # Check: Spalte existiert
            if col not in df.columns:
                errors.append(f"Produktspalte '{col}' existiert nicht")
                continue
            
            # Check: Numerisch
            try:
                values = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                errors.append(f"Spalte '{col}' kann nicht zu numerischen Werten konvertiert werden: {str(e)}")
                continue
            
            # Check: Alle NaN
            if values.isna().all():
                errors.append(f"Spalte '{col}' ist komplett leer (nur NaN)")
                continue
            
            # NaN Prozentsatz
            nan_count = values.isna().sum()
            nan_percent = (nan_count / len(values) * 100) if len(values) > 0 else 100
            metadata['nan_percentages'][col] = nan_percent
            
            if nan_percent >= DataValidator.MAX_NAN_PERCENT:
                errors.append(f"Spalte '{col}': Fast keine Daten ({nan_percent:.1f}% NaN)")
            elif nan_percent >= DataValidator.MAX_WARNING_NAN_PERCENT:
                warnings.append(f"Spalte '{col}': Viele fehlende Werte ({nan_percent:.1f}% NaN)")
            
            # Nur valide Werte für weitere Checks
            valid_values = values.dropna()
            
            if len(valid_values) == 0:
                continue
            
            # Check: Alle Null
            if (valid_values == 0).all():
                warnings.append(f"Spalte '{col}': Nur Nullen - keine Verkäufe")
                metadata['zero_columns'].append(col)
            
            # Check: Negative Werte
            if not allow_negatives:
                negative_count = (valid_values < 0).sum()
                if negative_count > 0:
                    warnings.append(
                        f"Spalte '{col}': {negative_count} negative Werte gefunden (Retouren?)"
                    )
                    metadata['negative_values'][col] = negative_count
            
            # Check: Varianz
            std = valid_values.std()
            mean = valid_values.mean()
            
            if std < DataValidator.MIN_VARIANCE_THRESHOLD * abs(mean) if mean != 0 else std == 0:
                warnings.append(f"Spalte '{col}': Sehr geringe Varianz - fast konstante Werte")
                metadata['no_variance'].append(col)
        
        # Info
        if not errors:
            info.append(f"{len(product_cols)} Produktspalten validiert")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            metadata=metadata
        )
    
    @staticmethod
    def validate_forecast_data(
        data: pd.Series,
        forecast_horizon: int,
        min_data_points: Optional[int] = None
    ) -> ValidationResult:
        """
        Validiert Daten für Forecasting.
        
        Args:
            data: Die Zeitreihendaten
            forecast_horizon: Forecast-Horizont
            min_data_points: Minimum Datenpunkte (default: MIN_FORECAST_DATA_POINTS)
            
        Returns:
            ValidationResult mit Fehler/Warnungen
            
        Checks:
            - Genug Datenpunkte
            - Forecast-Horizont valide
            - Keine lange NaN-Sequenzen
            - Genug Varianz
            - Extreme Outliers
        """
        errors = []
        warnings = []
        info = []
        metadata = {}
        
        if min_data_points is None:
            min_data_points = DataValidator.MIN_FORECAST_DATA_POINTS
        
        # Check: Forecast-Horizont
        if forecast_horizon < 1:
            errors.append(f"Forecast-Horizont muss mindestens 1 sein (aktuell: {forecast_horizon})")
        
        if forecast_horizon >= len(data):
            errors.append(
                f"Forecast-Horizont ({forecast_horizon}) ist zu groß für Datenmenge ({len(data)} Punkte)"
            )
        
        # Check: Genug Datenpunkte
        if len(data) < min_data_points:
            raise InsufficientDataError(
                f"Nicht genug Daten für Forecast: {len(data)} vorhanden, "
                f"mindestens {min_data_points} erforderlich"
            )
        
        # Check: NaN-Werte
        nan_count = data.isna().sum()
        nan_percent = (nan_count / len(data) * 100) if len(data) > 0 else 0
        
        if nan_percent > 30:
            warnings.append(f"Viele fehlende Werte: {nan_percent:.1f}% NaN - ggf. interpolieren")
        
        # Check: Lange NaN-Sequenzen
        is_nan = data.isna()
        nan_sequences = []
        current_sequence = 0
        
        for val in is_nan:
            if val:
                current_sequence += 1
            else:
                if current_sequence > 0:
                    nan_sequences.append(current_sequence)
                current_sequence = 0
        
        if current_sequence > 0:
            nan_sequences.append(current_sequence)
        
        if nan_sequences:
            max_nan_sequence = max(nan_sequences)
            if max_nan_sequence > len(data) * 0.3:
                warnings.append(
                    f"Große Datenlücke: {max_nan_sequence} aufeinanderfolgende NaN-Werte"
                )
        
        # Nur valide Werte für weitere Checks
        valid_data = data.dropna()
        
        if len(valid_data) < min_data_points:
            errors.append(
                f"Nach NaN-Entfernung nur noch {len(valid_data)} Datenpunkte "
                f"(mindestens {min_data_points} erforderlich)"
            )
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Check: Varianz
        std = valid_data.std()
        mean = valid_data.mean()
        
        if std < DataValidator.MIN_VARIANCE_THRESHOLD * abs(mean) if mean != 0 else std == 0:
            warnings.append("Sehr geringe Varianz - Daten fast konstant, Forecast möglicherweise wenig aussagekräftig")
        
        # Check: Extreme Outliers
        if len(valid_data) > 10:  # Nur wenn genug Daten
            q1 = valid_data.quantile(0.25)
            q3 = valid_data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 5 * iqr  # 5*IQR für extreme Outliers
            upper_bound = q3 + 5 * iqr
            
            extreme_outliers = ((valid_data < lower_bound) | (valid_data > upper_bound)).sum()
            
            if extreme_outliers > 0:
                info.append(f"{extreme_outliers} extreme Ausreißer erkannt - könnten Forecast beeinflussen")
                metadata['extreme_outliers'] = extreme_outliers
        
        # Info
        info.append(f"Forecast-Daten validiert: {len(valid_data)} Datenpunkte, Horizont {forecast_horizon}")
        
        # Metadata
        metadata['data_points'] = len(valid_data)
        metadata['forecast_horizon'] = forecast_horizon
        metadata['nan_percent'] = nan_percent
        metadata['std'] = std
        metadata['mean'] = mean
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            metadata=metadata
        )
    
    @staticmethod
    def assess_data_quality(
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        product_cols: Optional[List[str]] = None
    ) -> DataQualityReport:
        """
        Erstellt umfassenden Daten-Qualitäts-Report.
        
        Args:
            df: Der DataFrame
            date_col: Optional - Datumsspalte für Zeitreihen-Analyse
            product_cols: Optional - Produktspalten für Analyse
            
        Returns:
            DataQualityReport mit allen Metriken
        """
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Date Range
        date_range = None
        time_gaps = []
        
        if date_col and date_col in df.columns:
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                if len(dates) > 0:
                    date_range = (dates.min(), dates.max())
                    
                    # Zeitlücken erkennen (nur wenn sortiert und genug Daten)
                    if len(dates) > 2:
                        dates_sorted = dates.sort_values()
                        diff = dates_sorted.diff()[1:]  # Erste NaT überspringen
                        median_diff = diff.median()
                        
                        # Lücken sind > 2x median diff
                        gaps = diff[diff > 2 * median_diff]
                        
                        if len(gaps) > 0:
                            gap_indices = gaps.index
                            for idx in gap_indices[:10]:  # Max 10 Gaps
                                gap_start = dates_sorted.loc[dates_sorted.index < idx].iloc[-1]
                                gap_end = dates_sorted.loc[idx]
                                time_gaps.append((gap_start, gap_end))
            except:
                pass
        
        # Missing Values
        missing_values_pct = {}
        
        cols_to_check = product_cols if product_cols else df.columns
        for col in cols_to_check:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                nan_pct = (nan_count / len(df) * 100) if len(df) > 0 else 0
                missing_values_pct[col] = nan_pct
        
        # Outliers (nur numerische Spalten)
        outliers_count = {}
        variance_check = {}
        
        for col in cols_to_check:
            if col in df.columns:
                try:
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    
                    if len(values) > 10:
                        # Outliers via IQR
                        q1 = values.quantile(0.25)
                        q3 = values.quantile(0.75)
                        iqr = q3 - q1
                        
                        lower = q1 - DataValidator.MAX_OUTLIER_THRESHOLD * iqr
                        upper = q3 + DataValidator.MAX_OUTLIER_THRESHOLD * iqr
                        
                        outlier_count = ((values < lower) | (values > upper)).sum()
                        outliers_count[col] = outlier_count
                        
                        # Varianz
                        std = values.std()
                        mean = values.mean()
                        has_variance = std >= DataValidator.MIN_VARIANCE_THRESHOLD * abs(mean) if mean != 0 else std > 0
                        variance_check[col] = has_variance
                except:
                    pass
        
        # Quality Score berechnen (0-100)
        # Faktoren: NaN%, Outliers, Varianz, Zeitlücken
        score = 100.0
        
        # NaN penalty
        if missing_values_pct:
            avg_nan = sum(missing_values_pct.values()) / len(missing_values_pct)
            score -= min(avg_nan * 0.5, 30)  # Max 30 Punkte Abzug
        
        # Outlier penalty
        if outliers_count:
            total_outliers = sum(outliers_count.values())
            outlier_pct = (total_outliers / total_rows * 100) if total_rows > 0 else 0
            score -= min(outlier_pct * 2, 20)  # Max 20 Punkte Abzug
        
        # Varianz penalty
        if variance_check:
            no_var_count = sum(1 for has_var in variance_check.values() if not has_var)
            no_var_pct = (no_var_count / len(variance_check) * 100) if len(variance_check) > 0 else 0
            score -= min(no_var_pct * 0.3, 15)  # Max 15 Punkte Abzug
        
        # Zeitlücken penalty
        if time_gaps:
            score -= min(len(time_gaps) * 5, 15)  # Max 15 Punkte Abzug
        
        score = max(0, min(100, score))  # Clamp auf 0-100
        
        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            date_range=date_range,
            missing_values_pct=missing_values_pct,
            outliers_count=outliers_count,
            variance_check=variance_check,
            time_gaps=time_gaps,
            seasonality_detected=None,  # TODO: Implement seasonality detection
            quality_score=score
        )
