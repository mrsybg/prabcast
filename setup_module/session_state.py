# setup_module/session_state.py
"""
Zentrale Session State Management Klasse für PrABCast.
Arbeitet direkt mit st.session_state ohne Wrapper.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, date


class DataState:
    """Proxy für Data-bezogene Session State Variablen."""
    
    @property
    def df(self) -> Optional[pd.DataFrame]:
        return st.session_state.get('df', None)
    
    @df.setter
    def df(self, value: Optional[pd.DataFrame]):
        st.session_state.df = value
    
    @property
    def date_column(self) -> Optional[str]:
        return st.session_state.get('date_column', None)
    
    @date_column.setter
    def date_column(self, value: Optional[str]):
        st.session_state.date_column = value
    
    @property
    def selected_products(self) -> List[str]:
        # Unterstütze sowohl neuen als auch alten Key-Namen
        products = st.session_state.get('selected_products_in_data')
        if products is None:
            products = st.session_state.get('selected_products', [])
        return products
    
    @selected_products.setter
    def selected_products(self, value: List[str]):
        st.session_state.selected_products_in_data = value
        # Halte alten Key für Legacy-Code in Sync
        st.session_state.selected_products = value
    
    @property
    def start_date(self) -> Optional[date]:
        return st.session_state.get('start_date', None)
    
    @start_date.setter
    def start_date(self, value: Optional[date]):
        st.session_state.start_date = value
    
    @property
    def end_date(self) -> Optional[date]:
        return st.session_state.get('end_date', None)
    
    @end_date.setter
    def end_date(self, value: Optional[date]):
        st.session_state.end_date = value
    
    @property
    def start_date_selected(self) -> Optional[date]:
        return st.session_state.get('start_date_selected', None)
    
    @start_date_selected.setter
    def start_date_selected(self, value: Optional[date]):
        st.session_state.start_date_selected = value
    
    @property
    def end_date_selected(self) -> Optional[date]:
        return st.session_state.get('end_date_selected', None)
    
    @end_date_selected.setter
    def end_date_selected(self, value: Optional[date]):
        st.session_state.end_date_selected = value
    
    @property
    def is_loaded(self) -> bool:
        """Prüft, ob Basisdaten geladen sind."""
        return st.session_state.get('df') is not None
    
    @property
    def is_configured(self) -> bool:
        """Prüft, ob Daten vollständig konfiguriert sind."""
        return (
            st.session_state.get('df') is not None and
            st.session_state.get('date_column') is not None and
            len(self.selected_products) > 0
        )
    

class ForecastState:
    """Proxy für Forecast-bezogene Session State Variablen."""
    
    @property
    def forecast_results(self) -> Dict[str, Any]:
        return st.session_state.get('forecast_results', {})
    
    @property
    def selected_models(self) -> List[str]:
        return st.session_state.get('selected_models', [])
    
    @selected_models.setter
    def selected_models(self, value: List[str]):
        st.session_state.selected_models = value
    
    @property
    def forecast_horizon(self) -> int:
        return st.session_state.get('forecast_horizon', 12)
    
    @forecast_horizon.setter
    def forecast_horizon(self, value: int):
        st.session_state.forecast_horizon = value
    
    @property
    def multivariate_data(self) -> Optional[pd.DataFrame]:
        return st.session_state.get('multivariate_data', None)
    
    @multivariate_data.setter
    def multivariate_data(self, value: Optional[pd.DataFrame]):
        st.session_state.multivariate_data = value
    
    @property
    def saved_models(self) -> Dict[str, Any]:
        return st.session_state.get('saved_models', {})
    
    @property
    def forecast_complex_results(self) -> Optional[Dict[str, Any]]:
        return st.session_state.get('forecast_complex_results', None)
    
    @forecast_complex_results.setter
    def forecast_complex_results(self, value: Optional[Dict[str, Any]]):
        st.session_state.forecast_complex_results = value
    
    @property
    def forecast_complex_target(self) -> Optional[str]:
        return st.session_state.get('forecast_complex_target', None)
    
    @forecast_complex_target.setter
    def forecast_complex_target(self, value: Optional[str]):
        st.session_state.forecast_complex_target = value
    
    @property
    def forecast_status(self) -> Optional[str]:
        return st.session_state.get('forecast_status', None)
    
    @forecast_status.setter
    def forecast_status(self, value: Optional[str]):
        st.session_state.forecast_status = value
    
    @property
    def saved_model_status(self) -> Optional[str]:
        return st.session_state.get('saved_model_status', None)
    
    @saved_model_status.setter
    def saved_model_status(self, value: Optional[str]):
        st.session_state.saved_model_status = value
    
    @property
    def has_multivariate_data(self) -> bool:
        """Prüft, ob multivariate Daten verfügbar sind."""
        data = st.session_state.get('multivariate_data', None)
        return data is not None and not data.empty
    
    @property
    def has_saved_models(self) -> bool:
        """Prüft, ob gespeicherte Modelle vorhanden sind."""
        return len(st.session_state.get('saved_models', {})) > 0
    
    def clear_forecast_results(self):
        """Löscht alle Prognoseergebnisse."""
        if 'forecast_results' in st.session_state:
            st.session_state.forecast_results.clear()
        st.session_state.forecast_complex_results = None
        st.session_state.forecast_status = None


class UIState:
    """Proxy für UI-bezogene Session State Variablen."""
    
    @property
    def current_tab(self) -> str:
        return st.session_state.get('current_tab', "upload")
    
    @current_tab.setter
    def current_tab(self, value: str):
        st.session_state.current_tab = value
    
    @property
    def show_advanced_options(self) -> bool:
        return st.session_state.get('show_advanced_options', False)
    
    @show_advanced_options.setter
    def show_advanced_options(self, value: bool):
        st.session_state.show_advanced_options = value
    
    @property
    def date_filter_active(self) -> bool:
        return st.session_state.get('date_filter_active', True)
    
    @date_filter_active.setter
    def date_filter_active(self, value: bool):
        st.session_state.date_filter_active = value


class AppState:
    """Zentrale Klasse für strukturierten Zugriff auf Session State."""
    
    _instance = None
    _data = None
    _forecast = None
    _ui = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def data(self) -> DataState:
        if AppState._data is None:
            AppState._data = DataState()
        return AppState._data
    
    @property
    def forecast(self) -> ForecastState:
        if AppState._forecast is None:
            AppState._forecast = ForecastState()
        return AppState._forecast
    
    @property
    def ui(self) -> UIState:
        if AppState._ui is None:
            AppState._ui = UIState()
        return AppState._ui
    
    @property
    def ready_for_processing(self) -> bool:
        """Prüft, ob die Anwendung bereit für die Verarbeitung ist."""
        return self.data.is_configured
    
    # Backward Compatibility Properties für sanfte Migration
    @property
    def df(self) -> Optional[pd.DataFrame]:
        """Legacy: Zugriff auf DataFrame."""
        return self.data.df
    
    @df.setter
    def df(self, value: pd.DataFrame):
        """Legacy: Setzen des DataFrame."""
        self.data.df = value
    
    @property
    def date_column(self) -> Optional[str]:
        """Legacy: Zugriff auf Datumsspalte."""
        return self.data.date_column
    
    @date_column.setter
    def date_column(self, value: str):
        """Legacy: Setzen der Datumsspalte."""
        self.data.date_column = value
    
    @property
    def selected_products_in_data(self) -> List[str]:
        """Legacy: Zugriff auf ausgewählte Produkte."""
        return self.data.selected_products
    
    @selected_products_in_data.setter
    def selected_products_in_data(self, value: List[str]):
        """Legacy: Setzen der ausgewählten Produkte."""
        self.data.selected_products = value


# Globale Instanz für einfachen Zugriff
def get_app_state() -> AppState:
    """
    Gibt die globale AppState-Instanz zurück.
    
    Usage:
        state = get_app_state()
        if state.ready_for_processing:
            data = state.data.get_filtered_data()
    """
    return AppState()


# Convenience Functions für häufige Operationen
def is_ready_for_processing() -> bool:
    """Prüft, ob Daten bereit zur Verarbeitung sind."""
    return get_app_state().ready_for_processing


def get_current_data() -> Optional[pd.DataFrame]:
    """Gibt die aktuell geladenen Daten zurück."""
    return get_app_state().data.df


def get_filtered_data(start_date: Optional[date] = None, end_date: Optional[date] = None) -> pd.DataFrame:
    """Gibt gefilterte Daten zurück."""
    return get_app_state().data.get_filtered_data(start_date, end_date)


def get_selected_products() -> List[str]:
    """Gibt die Liste der ausgewählten Produkte zurück."""
    return get_app_state().data.selected_products
