"""
Smart Defaults System - Automatische Erkennung und Empfehlung von Forecast-Parametern

Dieses Modul analysiert Zeitreihendaten und gibt intelligente Empfehlungen für:
- Forecast Horizon (Prognosezeitraum)
- Saisonalität (Seasonal Period)
- ARIMA Parameter (p, d, q)
- Beste Modelle basierend auf Daten-Charakteristik

Zielgruppe: Produktionsplaner ohne tiefe ML-Kenntnisse
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings('ignore')


class SmartDefaults:
    """
    Intelligente Empfehlungen für Forecast-Parameter
    """
    
    @staticmethod
    def detect_seasonality(data: pd.Series, max_lag: int = 50) -> Dict[str, any]:
        """
        Erkennt automatisch Saisonalität in Zeitreihendaten
        
        Args:
            data: Zeitreihendaten (pd.Series mit DatetimeIndex)
            max_lag: Maximum Lag für ACF-Analyse
            
        Returns:
            Dict mit:
                - has_seasonality: bool
                - period: int (erkannte Periode, z.B. 12 für monatlich)
                - strength: float (0-1, Stärke der Saisonalität)
                - confidence: str ('high', 'medium', 'low')
                - reason: str (Erklärung für User)
        """
        result = {
            'has_seasonality': False,
            'period': None,
            'strength': 0.0,
            'confidence': 'low',
            'reason': ''
        }
        
        # Mindestens 2 Perioden benötigt
        if len(data) < 24:
            result['reason'] = "Zu wenig Daten für Saisonalitätserkennung (min. 24 Datenpunkte)"
            return result
        
        try:
            # 1. ACF-basierte Erkennung
            acf_values = acf(data.dropna(), nlags=min(max_lag, len(data)//2 - 1))
            
            # Suche nach signifikanten Peaks (außer Lag 0)
            peaks = []
            for lag in range(2, len(acf_values)):
                if lag > 1 and lag < len(acf_values) - 1:
                    # Peak wenn Wert größer als Nachbarn
                    if acf_values[lag] > acf_values[lag-1] and acf_values[lag] > acf_values[lag+1]:
                        # Nur signifikante Peaks (> 0.3)
                        if abs(acf_values[lag]) > 0.3:
                            peaks.append((lag, abs(acf_values[lag])))
            
            if peaks:
                # Stärkster Peak ist wahrscheinlich Saisonalität
                peaks.sort(key=lambda x: x[1], reverse=True)
                period, strength = peaks[0]
                
                result['has_seasonality'] = True
                result['period'] = period
                result['strength'] = strength
                
                # Confidence basierend auf Stärke
                if strength > 0.6:
                    result['confidence'] = 'high'
                    result['reason'] = f"Starke Saisonalität alle {period} Perioden erkannt (ACF: {strength:.2f})"
                elif strength > 0.4:
                    result['confidence'] = 'medium'
                    result['reason'] = f"Moderate Saisonalität alle {period} Perioden erkannt (ACF: {strength:.2f})"
                else:
                    result['confidence'] = 'low'
                    result['reason'] = f"Schwache Saisonalität alle {period} Perioden möglich (ACF: {strength:.2f})"
                
            else:
                result['reason'] = "Keine klare Saisonalität erkannt - Daten scheinen nicht saisonal zu sein"
                
        except Exception as e:
            result['reason'] = f"Saisonalitätserkennung fehlgeschlagen: {str(e)}"
        
        return result
    
    
    @staticmethod
    def recommend_horizon(data: pd.Series, 
                         has_seasonality: bool = False,
                         seasonal_period: Optional[int] = None) -> Dict[str, any]:
        """
        Empfiehlt intelligenten Forecast Horizon
        
        Args:
            data: Zeitreihendaten
            has_seasonality: Ob Saisonalität erkannt wurde
            seasonal_period: Saisonale Periode (falls vorhanden)
            
        Returns:
            Dict mit:
                - recommended: int (empfohlener Horizon)
                - min: int (Minimum sinnvoll)
                - max: int (Maximum sinnvoll)
                - reason: str (Erklärung)
        """
        n = len(data)
        
        # Regel: 20% der Datenlänge, aber nicht mehr als 12 (1 Jahr bei Monatsdaten)
        # Mindestens 3 Perioden
        recommended = max(3, min(int(n * 0.2), 12))
        
        # Bei Saisonalität: Mindestens 1 Saison
        if has_seasonality and seasonal_period:
            recommended = max(recommended, seasonal_period)
        
        # Minimum: 3 Perioden
        min_horizon = 3
        
        # Maximum: 30% der Daten oder 24 (2 Jahre)
        max_horizon = min(int(n * 0.3), 24)
        
        # Reason
        if has_seasonality and seasonal_period:
            reason = f"Empfohlen: {recommended} Perioden (≥1 Saison von {seasonal_period} Perioden)"
        else:
            reason = f"Empfohlen: {recommended} Perioden (≈20% der Datenlänge von {n})"
        
        return {
            'recommended': recommended,
            'min': min_horizon,
            'max': max_horizon,
            'reason': reason
        }
    
    
    @staticmethod
    def suggest_arima_params(data: pd.Series) -> Dict[str, any]:
        """
        Schlägt ARIMA (p,d,q) Parameter vor
        
        Args:
            data: Zeitreihendaten
            
        Returns:
            Dict mit:
                - p: int (AR order)
                - d: int (Differencing order)
                - q: int (MA order)
                - reason: str (Erklärung)
                - method: str (Methode der Erkennung)
        """
        result = {
            'p': 1,
            'd': 1,
            'q': 1,
            'reason': '',
            'method': 'default'
        }
        
        try:
            # 1. Stationarity Test (für d)
            adf_result = adfuller(data.dropna())
            p_value = adf_result[1]
            
            if p_value < 0.05:
                # Stationär
                result['d'] = 0
                result['reason'] = "Daten sind bereits stationär (d=0). "
            else:
                # Nicht stationär - eine Differenzierung
                result['d'] = 1
                result['reason'] = "Daten sind nicht stationär - eine Differenzierung empfohlen (d=1). "
            
            # 2. ACF/PACF für p und q
            # PACF für p (AR order)
            pacf_values = pacf(data.dropna(), nlags=min(20, len(data)//2))
            
            # Finde wo PACF insignifikant wird (< 0.2)
            p = 1
            for i in range(1, len(pacf_values)):
                if abs(pacf_values[i]) > 0.2:
                    p = i
                else:
                    break
            p = min(p, 3)  # Max 3
            result['p'] = p
            
            # ACF für q (MA order)  
            acf_values = acf(data.dropna(), nlags=min(20, len(data)//2))
            
            q = 1
            for i in range(1, len(acf_values)):
                if abs(acf_values[i]) > 0.2:
                    q = i
                else:
                    break
            q = min(q, 3)  # Max 3
            result['q'] = q
            
            result['reason'] += f"AR={p}, MA={q} basierend auf ACF/PACF-Analyse."
            result['method'] = 'acf_pacf'
            
        except Exception as e:
            result['reason'] = f"Automatische Erkennung fehlgeschlagen, nutze Defaults (1,1,1): {str(e)}"
        
        return result
    
    
    @staticmethod
    def recommend_models(data: pd.Series,
                        has_seasonality: bool = False,
                        seasonal_period: Optional[int] = None,
                        data_length: int = 0) -> Dict[str, any]:
        """
        Empfiehlt beste Modelle basierend auf Daten-Charakteristik
        
        Args:
            data: Zeitreihendaten
            has_seasonality: Ob Saisonalität vorhanden
            seasonal_period: Saisonale Periode
            data_length: Länge der Daten
            
        Returns:
            Dict mit:
                - primary: List[str] (Hauptempfehlungen - 2-3 Modelle)
                - secondary: List[str] (Alternative Modelle)
                - avoid: List[str] (Nicht empfohlene Modelle)
                - reasons: Dict[str, str] (Begründung pro Modell)
        """
        primary = []
        secondary = []
        avoid = []
        reasons = {}
        
        # Berechne Variabilität (CV = std/mean)
        cv = data.std() / data.mean() if data.mean() != 0 else 0
        
        # Trend Detection
        try:
            decompose_result = seasonal_decompose(data.dropna(), model='additive', period=min(seasonal_period or 12, len(data)//2))
            trend = decompose_result.trend.dropna()
            has_trend = abs(trend.iloc[-1] - trend.iloc[0]) / trend.mean() > 0.1 if len(trend) > 0 else False
        except:
            has_trend = False
        
        # === MODELL-EMPFEHLUNGEN ===
        
        # 1. ARIMA/SARIMA - Immer eine gute Basis
        if has_seasonality and seasonal_period:
            primary.append("SARIMA")
            reasons["SARIMA"] = f"Beste Wahl: Saisonalität erkannt (Periode={seasonal_period})"
            secondary.append("ARIMA")
            reasons["ARIMA"] = "Alternative: Ignoriert Saisonalität"
        else:
            primary.append("ARIMA")
            reasons["ARIMA"] = "Gute Wahl: Klassisches Modell für nicht-saisonale Daten"
        
        # 2. Prophet - Gut für Saisonalität und Trends
        if has_seasonality or has_trend:
            primary.append("Prophet")
            if has_seasonality and has_trend:
                reasons["Prophet"] = "Sehr gut: Erkennt Saisonalität UND Trend automatisch"
            elif has_seasonality:
                reasons["Prophet"] = "Gut: Handhabt Saisonalität robust"
            else:
                reasons["Prophet"] = "Gut: Erkennt Trends automatisch"
        else:
            secondary.append("Prophet")
            reasons["Prophet"] = "Alternative: Auch ohne Saisonalität verwendbar"
        
        # 3. Exponential Smoothing - Einfach und schnell
        if has_seasonality:
            secondary.append("Holt-Winters")
            reasons["Holt-Winters"] = "Alternative: Einfaches saisonales Modell"
        else:
            secondary.append("ExponentialSmoothing")
            reasons["ExponentialSmoothing"] = "Alternative: Einfaches Trend-Modell"
        
        # 4. LSTM/GRU - Nur bei vielen Daten
        if data_length >= 100:
            secondary.append("LSTM")
            reasons["LSTM"] = f"Möglich: Ausreichend Daten ({data_length} Punkte) für Deep Learning"
        else:
            avoid.append("LSTM")
            reasons["LSTM"] = f"Nicht empfohlen: Zu wenig Daten ({data_length} < 100) für Deep Learning"
        
        # 5. XGBoost - Bei komplexen Mustern
        if cv > 0.3 and data_length >= 50:
            secondary.append("XGBoost")
            reasons["XGBoost"] = "Möglich: Hohe Variabilität in Daten (CV={cv:.2f})"
        
        # 6. Naive/Drift - Nur als Baseline
        avoid.append("Naive")
        avoid.append("SeasonalNaive")
        reasons["Naive"] = "Baseline: Nur für Vergleich, keine echte Prognose"
        reasons["SeasonalNaive"] = "Baseline: Nur für Vergleich, keine echte Prognose"
        
        return {
            'primary': primary[:3],  # Max 3 Hauptempfehlungen
            'secondary': secondary,
            'avoid': avoid,
            'reasons': reasons
        }
    
    
    @staticmethod
    def get_smart_recommendations(data: pd.Series) -> Dict[str, any]:
        """
        Komplett-Analyse: Alle Smart Defaults auf einmal
        
        Args:
            data: Zeitreihendaten (pd.Series mit DatetimeIndex)
            
        Returns:
            Dict mit allen Empfehlungen:
                - seasonality: Dict
                - horizon: Dict
                - arima_params: Dict
                - models: Dict
        """
        # 1. Saisonalität erkennen
        seasonality = SmartDefaults.detect_seasonality(data)
        
        # 2. Horizon empfehlen
        horizon = SmartDefaults.recommend_horizon(
            data,
            has_seasonality=seasonality['has_seasonality'],
            seasonal_period=seasonality['period']
        )
        
        # 3. ARIMA Parameter
        arima_params = SmartDefaults.suggest_arima_params(data)
        
        # 4. Modelle empfehlen
        models = SmartDefaults.recommend_models(
            data,
            has_seasonality=seasonality['has_seasonality'],
            seasonal_period=seasonality['period'],
            data_length=len(data)
        )
        
        return {
            'seasonality': seasonality,
            'horizon': horizon,
            'arima_params': arima_params,
            'models': models
        }


def format_recommendation_message(recommendations: Dict) -> str:
    """
    Formatiert Empfehlungen als benutzerfreundliche Nachricht
    
    Args:
        recommendations: Output von get_smart_recommendations()
        
    Returns:
        Formatierter String für UI-Anzeige
    """
    msg = "## 🎯 Intelligente Empfehlungen\n\n"
    
    # Saisonalität
    season = recommendations['seasonality']
    if season['has_seasonality']:
        msg += f"**Saisonalität**: {season['reason']}\n\n"
    else:
        msg += f"**Saisonalität**: Keine erkannt - Daten scheinen nicht saisonal zu sein\n\n"
    
    # Horizon
    horizon = recommendations['horizon']
    msg += f"**Prognosehorizont**: {horizon['recommended']} Perioden empfohlen\n"
    msg += f"  _{horizon['reason']}_\n\n"
    
    # Modelle
    models = recommendations['models']
    msg += f"**Empfohlene Modelle**: {', '.join(models['primary'])}\n"
    for model in models['primary']:
        if model in models['reasons']:
            msg += f"  • {model}: {models['reasons'][model]}\n"
    
    return msg
