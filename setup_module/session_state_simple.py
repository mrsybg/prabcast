# setup_module/session_state_simple.py
"""
EINFACHSTE Session State Lösung - Direkt mit st.session_state arbeiten.
KEINE Wrapper, KEINE Klassen, NUR direkte Zugriffe.
"""

import streamlit as st


def init_state():
    """Initialisiert Session State beim ersten Aufruf."""
    # Data
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'date_column' not in st.session_state:
        st.session_state.date_column = None
    if 'selected_products_in_data' not in st.session_state:
        st.session_state.selected_products_in_data = []
    if 'start_date' not in st.session_state:
        st.session_state.start_date = None
    if 'end_date' not in st.session_state:
        st.session_state.end_date = None
    if 'start_date_selected' not in st.session_state:
        st.session_state.start_date_selected = None
    if 'end_date_selected' not in st.session_state:
        st.session_state.end_date_selected = None
    
    # Forecast
    if 'multivariate_data' not in st.session_state:
        st.session_state.multivariate_data = None
    if 'saved_models' not in st.session_state:
        st.session_state.saved_models = {}
    if 'forecast_complex_results' not in st.session_state:
        st.session_state.forecast_complex_results = None
    if 'forecast_status' not in st.session_state:
        st.session_state.forecast_status = None


def ready_for_processing():
    """Prüft, ob Daten bereit sind."""
    return (st.session_state.get('df') is not None and
            st.session_state.get('date_column') is not None and
            len(st.session_state.get('selected_products_in_data', [])) > 0)


# Initialisiere beim Import
init_state()
