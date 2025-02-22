# setup_module/helpers.py
import streamlit as st
import pandas as pd

# Setze die Standardwerte für st.session_state
if "df" not in st.session_state:
    st.session_state.df = None

if "date_column" not in st.session_state:
    st.session_state.date_column = None

if "selected_products_in_data" not in st.session_state:
    st.session_state.selected_products_in_data = []


def create_date_filter(key_prefix="", use_selected_keys=True):
    """
    Zentrale Funktion für Datumsfiltererstellung mit eindeutigen Schlüsseln.

    Args:
        key_prefix (str): Prefix für die Schlüssel der Datumseingaben
        use_selected_keys (bool): Steuerung, ob 'start_date_selected' und 'end_date_selected'
                                  oder 'start_date' und 'end_date' verwendet werden.
    """
    # Schlüssel für das Start- und Enddatum, abhängig von use_selected_keys
    start_date_key = 'start_date_selected' if use_selected_keys else 'start_date'
    end_date_key = 'end_date_selected' if use_selected_keys else 'end_date'

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Startdatum",
            value=st.session_state.get(start_date_key),
            min_value=st.session_state.get(start_date_key),
            max_value=st.session_state.get(end_date_key),
            key=f"{key_prefix}_start_date",
            format="DD/MM/YYYY"
        )

    with col2:
        end_date = st.date_input(
            "Enddatum",
            value=st.session_state.get(end_date_key),
            min_value=st.session_state.get(start_date_key),
            max_value=st.session_state.get(end_date_key),
            key=f"{key_prefix}_end_date",
            format="DD/MM/YYYY"
        )

    return start_date, end_date


def filter_df(start_date, end_date):
    """Filtert DataFrame nach Datum"""
    df = st.session_state.df.copy()
    
    # Sicherstellen, dass die Datumsspalte als Index gesetzt ist
    if st.session_state.date_column in df.columns:
        # Deutsches Datumsformat (DD.MM.YYYY)
        df[st.session_state.date_column] = pd.to_datetime(df[st.session_state.date_column], format='%d.%m.%Y')
        df.set_index(st.session_state.date_column, inplace=True)
    
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
        if x == int(x):  # Prüfe, ob die Zahl eine Ganzzahl ist
            return '{:,.0f}'.format(x)
        else:
            return '{:,.{prec}f}'.format(x, prec=decimal_places)  # Dynamische Anzahl Dezimalstellen

    summary_table_formatted = summary_table_to_format.style.format(
        format_value,
        subset=subset_columns,
        thousands='.',
        decimal=',',
    )
    return summary_table_formatted


def check_ready_for_processing():
    if st.session_state.date_column and st.session_state.selected_products_in_data:
        st.session_state.ready_for_processing = True
    else:
        st.session_state.ready_for_processing = False