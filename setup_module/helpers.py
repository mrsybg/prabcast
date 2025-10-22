# setup_module/helpers.py
"""
Helper-Funktionen für PrABCast.
Nutzt DIREKTEN Session State Zugriff.
"""

import streamlit as st
import pandas as pd
import numpy as np
from .session_state_simple import init_state, ready_for_processing

# Initialisiere Session State
init_state()


def create_date_filter(key_prefix="", use_selected_keys=True):
    """
    Zentrale Funktion für Datumsfiltererstellung mit eindeutigen Schlüsseln.

    Args:
        key_prefix (str): Prefix für die Schlüssel der Datumseingaben
        use_selected_keys (bool): Steuerung, ob 'start_date_selected' und 'end_date_selected'
                                  oder 'start_date' und 'end_date' verwendet werden.
    """
    
    # Wähle die richtigen Datumswerte aus Session State
    if use_selected_keys:
        start_value = st.session_state.get('start_date_selected') or st.session_state.get('start_date')
        end_value = st.session_state.get('end_date_selected') or st.session_state.get('end_date')
        min_value = st.session_state.get('start_date_selected') or st.session_state.get('start_date')
        max_value = st.session_state.get('end_date_selected') or st.session_state.get('end_date')
    else:
        start_value = st.session_state.get('start_date')
        end_value = st.session_state.get('end_date')
        min_value = st.session_state.get('start_date')
        max_value = st.session_state.get('end_date')

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Startdatum",
            value=start_value,
            min_value=min_value,
            max_value=max_value,
            key=f"{key_prefix}_start_date",
            format="DD/MM/YYYY"
        )

    with col2:
        end_date = st.date_input(
            "Enddatum",
            value=end_value,
            min_value=min_value,
            max_value=max_value,
            key=f"{key_prefix}_end_date",
            format="DD/MM/YYYY"
        )

    return start_date, end_date


def filter_df(start_date, end_date):
    """
    Filtert DataFrame nach Datum.
    Nutzt direkten Session State Zugriff.
    """
    
    if st.session_state.get('df') is None:
        return pd.DataFrame()
    
    df = st.session_state.df.copy()
    date_column = st.session_state.get('date_column')
    
    # Sicherstellen, dass die Datumsspalte als Index gesetzt ist
    if date_column in df.columns:
        # Deutsches Datumsformat (DD.MM.YYYY)
        df[date_column] = pd.to_datetime(df[date_column], format='%d.%m.%Y')
        df.set_index(date_column, inplace=True)
    
    # Konvertiere Datumswerte
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filtere DataFrame
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df.loc[mask]

def aggregation_select_box(option):
    options = ["Monatlich", "Quartalsweise", "Jährlich"]
    return st.selectbox("Wähle das Aggregationslevel", options, index=0 if option == 1 else 1)

def get_resampling_frequency(aggregation_level):
    freq_map = {
        "Monatlich": "M",
        "Quartalsweise": "Q",
        "Jährlich": "A"
    }
    return freq_map.get(aggregation_level, "M")

def create_summary_table(df, selected_products_in_data=None):
    summary = df.describe().transpose()
    if selected_products_in_data:
        summary = summary.loc[selected_products_in_data]
    return summary

def german_number_format(x, decimals=2):
    try:
        x = float(x)
        # First format with US-style, then swap symbols:
        formatted = f"{x:,.{decimals}f}"
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return x

def format_summary_table(df, columns_to_format, decimal_places=2):
    formatter = {
        col: (lambda x, dec=decimal_places: german_number_format(x, dec))
        for col in df.columns
    }
    return df.style.format(formatter)

def format_summary_table(summary_table_to_format, subset_columns, decimal_places=2):
    """
    Formatiert die Zahlen eines DF von 56,843.5 zu 56.843,5.

    Parameter:
    summary_table_to_format (DataFrame): Eine pandas DataFrame.
    subset_columns (list): Eine Liste von Spaltennamen (als Strings), die formatiert werden
                           sollen (z. B. ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]).
    decimal_places (int): Die Anzahl der Dezimalstellen, die angezeigt werden sollen (Standard: 2).

    Rückgabewert:
    pandas Styler: Ein formatierter pandas Styler, der auf die angegebenen Spalten angewendet wurde.
    """

    def format_value(x):
        # Handle infinity and NaN
        if pd.isna(x) or np.isinf(x):
            return 'N/A'
        # Check if it's a whole number
        try:
            if x == int(x):
                return '{:,.0f}'.format(x)
        except (ValueError, OverflowError):
            return 'N/A'
        return '{:,.{prec}f}'.format(x, prec=decimal_places)  # Dynamische Anzahl Dezimalstellen

    summary_table_formatted = summary_table_to_format.style.format(
        format_value,
        subset=subset_columns,
        thousands='.',
        decimal=',',
    )
    return summary_table_formatted


def check_ready_for_processing():
    """
    Prüft, ob alle erforderlichen Daten konfiguriert sind.
    Nutzt die Session State Validierung.
    """
    return ready_for_processing()


# Backward Compatibility Functions für sanfte Migration
def get_df():
    """Legacy: Gibt DataFrame zurück."""
    return st.session_state.get('df')


def get_date_column():
    """Legacy: Gibt Datumsspalte zurück."""
    return st.session_state.get('date_column')


def get_selected_products():
    """Legacy: Gibt ausgewählte Produkte zurück."""
    return st.session_state.get('selected_products_in_data', [])


def is_ready_for_processing():
    """Legacy: Prüft Verarbeitungsbereitschaft."""
    return ready_for_processing()