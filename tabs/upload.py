from setup_module.helpers import *

# Caching der hochgeladenen Datei
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file, delimiter=';')

def display_tab():
    uploaded_file = st.file_uploader(
        "Hier eine CSV-Datei hochladen/aktualisieren.", type="csv"
    )

    if uploaded_file:
        df = load_csv(uploaded_file)
        st.session_state.df = df
        
        # Dropdown für die Auswahl der Datumsspalte
        date_column = st.selectbox(
            "Wähle die Datumsspalte:",
            st.session_state.df.columns,
            index=None,
            on_change=check_ready_for_processing,
        )
        if date_column is not None:
            st.session_state.date_column = date_column
            # Initialize date range in session state with German date format
            df[st.session_state.date_column] = pd.to_datetime(df[st.session_state.date_column], format='%d.%m.%Y')
            st.session_state.start_date = df[st.session_state.date_column].min()
            st.session_state.end_date = df[st.session_state.date_column].max()
        else:
            st.warning("Bitte wähle eine Datumsspalte aus.")
            return

        # Produktauswahl
        columns = [col for col in st.session_state.df.columns if col != date_column]
        selected_products = st.multiselect(
            "Wähle die Produktspalten:",
            columns,
            on_change=check_ready_for_processing,
        )
        if selected_products:
            st.session_state.selected_products_in_data = selected_products

        # Date filter using new centralized function
        if st.session_state.date_column:

            st.session_state.start_date_selected, st.session_state.end_date_selected= create_date_filter("upload", use_selected_keys=False)

            if st.session_state.start_date_selected and st.session_state.end_date_selected:
                filtered_df = filter_df(st.session_state.start_date_selected, st.session_state.end_date_selected)
                with st.expander("Tabelle anzeigen"):
                    st.dataframe(filtered_df)

    check_ready_for_processing()