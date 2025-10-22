# setup_module/memory_manager.py
"""
Memory Management Utilities für PrABCast
Verhindert Memory Leaks bei ML-Modellen und großen DataFrames
"""

import streamlit as st
import gc
from typing import Optional, List
from setup_module.logging_config import logger

try:
    from tensorflow.keras import backend as K
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class MemoryManager:
    """Verwaltet Speicher-Cleanup für ML-Modelle und große Objekte"""
    
    @staticmethod
    def clear_keras_session():
        """Löscht Keras/TensorFlow Session"""
        if TENSORFLOW_AVAILABLE:
            try:
                K.clear_session()
                logger.info("Keras session cleared")
            except Exception as e:
                logger.warning(f"Failed to clear Keras session: {e}")
    
    @staticmethod
    def clear_old_models(max_models: int = 10):
        """
        Löscht alte Modelle aus session_state
        
        Args:
            max_models: Maximale Anzahl gespeicherter Modelle
        """
        if 'saved_models' in st.session_state:
            models = st.session_state.saved_models
            if len(models) > max_models:
                # Lösche älteste Modelle (FIFO)
                models_to_remove = list(models.keys())[:-max_models]
                for key in models_to_remove:
                    del models[key]
                    logger.info(f"Removed old model: {key}")
                gc.collect()
    
    @staticmethod
    def clear_old_forecasts(max_forecasts: int = 20):
        """
        Löscht alte Forecast-Ergebnisse aus session_state
        
        Args:
            max_forecasts: Maximale Anzahl gespeicherter Forecasts
        """
        if 'forecast_results' in st.session_state:
            results = st.session_state.forecast_results
            if len(results) > max_forecasts:
                # Lösche älteste Results (FIFO)
                results_to_remove = list(results.keys())[:-max_forecasts]
                for key in results_to_remove:
                    del results[key]
                    logger.info(f"Removed old forecast: {key}")
                gc.collect()
    
    @staticmethod
    def get_session_state_size() -> dict:
        """
        Schätzt Speicherverbrauch von session_state Objekten
        
        Returns:
            Dictionary mit Größen-Informationen
        """
        import sys
        
        sizes = {}
        total_size = 0
        
        for key, value in st.session_state.items():
            try:
                size = sys.getsizeof(value)
                sizes[key] = size
                total_size += size
            except Exception as e:
                sizes[key] = f"Error: {e}"
        
        sizes['_total_bytes'] = total_size
        sizes['_total_mb'] = round(total_size / (1024 * 1024), 2)
        
        return sizes
    
    @staticmethod
    def cleanup_all(force: bool = False):
        """
        Führt vollständigen Memory-Cleanup durch
        
        Args:
            force: Wenn True, löscht ALLE gespeicherten Modelle/Forecasts
        """
        if force:
            if 'saved_models' in st.session_state:
                st.session_state.saved_models.clear()
                logger.info("Cleared all saved models")
            
            if 'forecast_results' in st.session_state:
                st.session_state.forecast_results.clear()
                logger.info("Cleared all forecast results")
        else:
            MemoryManager.clear_old_models()
            MemoryManager.clear_old_forecasts()
        
        MemoryManager.clear_keras_session()
        gc.collect()
        logger.info("Memory cleanup completed")
    
    @staticmethod
    def add_cleanup_button(location: str = "sidebar"):
        """
        Fügt einen Memory-Cleanup Button zur UI hinzu
        
        Args:
            location: "sidebar" oder "main"
        """
        container = st.sidebar if location == "sidebar" else st
        
        with container.expander("🗑️ Speicher-Management"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Cache leeren", use_container_width=True):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Cache geleert")
            
            with col2:
                if st.button("Memory cleanup", use_container_width=True):
                    MemoryManager.cleanup_all(force=False)
                    st.success("Cleanup durchgeführt")
            
            # Zeige Speicher-Info
            sizes = MemoryManager.get_session_state_size()
            st.metric("Session State", f"{sizes.get('_total_mb', 0)} MB")
            
            if st.checkbox("Details anzeigen"):
                st.json({k: v for k, v in sizes.items() if not k.startswith('_')})


def auto_cleanup_decorator(max_models: int = 10, max_forecasts: int = 20):
    """
    Decorator für automatischen Memory-Cleanup nach Forecast-Funktionen
    
    Usage:
        @auto_cleanup_decorator(max_models=5)
        def my_forecast_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            MemoryManager.clear_old_models(max_models)
            MemoryManager.clear_old_forecasts(max_forecasts)
            return result
        return wrapper
    return decorator
