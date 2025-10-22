from setup_module.helpers import *
from setup_module.session_state_simple import init_state, ready_for_processing
from setup_module.error_handler import ErrorHandler, UserFeedback
from setup_module.exceptions import DataLoadError, DateParsingError, DataValidationError
from setup_module.logging_config import log_data_operation, log_error
from setup_module.data_validator import DataValidator

# Import UI Components
from setup_module.design_system import UI
from setup_module.ui_helpers import display_validation_results, display_quality_dashboard, safe_execute

# Initialisiere State
init_state()

# Caching der hochgeladenen Datei
@st.cache_data(ttl=3600)  # ✨ CACHE: 1 Stunde Cache
@ErrorHandler.handle_errors(show_details=True)
def load_csv(uploaded_file):
    """
    Lädt eine CSV-Datei mit Fehlerbehandlung.
    
    Raises:
        DataLoadError: Wenn die Datei nicht gelesen werden kann
    """
    try:
        df = pd.read_csv(uploaded_file, delimiter=';')
        log_data_operation("CSV-Datei geladen", data_shape=(len(df), len(df.columns)))
        return df
    except pd.errors.ParserError as e:
        raise DataLoadError(
            message="CSV-Datei konnte nicht geparst werden",
            details=f"Parser-Fehler: {str(e)}",
            help_text="Überprüfen Sie, ob die Datei das richtige Trennzeichen (;) verwendet.",
            original_exception=e
        )
    except Exception as e:
        raise DataLoadError(
            message="Fehler beim Laden der CSV-Datei",
            details=str(e),
            help_text="Stellen Sie sicher, dass die Datei im CSV-Format vorliegt.",
            original_exception=e
        )

@ErrorHandler.handle_errors(show_details=True)
def parse_date_column(df, date_column):
    """
    Parst die Datumsspalte mit Fehlerbehandlung.
    
    Args:
        df: DataFrame mit den Daten
        date_column: Name der Datumsspalte
        
    Returns:
        DataFrame mit geparster Datumsspalte
        
    Raises:
        DateParsingError: Wenn die Datumsspalte nicht geparst werden kann
    """
    try:
        df[date_column] = pd.to_datetime(df[date_column], format='%d.%m.%Y', errors='coerce')
        
        # Prüfe auf ungültige Daten
        invalid_dates = df[date_column].isna().sum()
        if invalid_dates > 0:
            UserFeedback.warning(
                f"{invalid_dates} Datumseinträge konnten nicht geparst werden",
                help_text="Überprüfen Sie, ob alle Daten im Format TT.MM.JJJJ vorliegen."
            )
        
        log_data_operation("Datumsspalte geparst", data_shape=(len(df), len(df.columns)), column=date_column, invalid_count=invalid_dates)
        return df
        
    except Exception as e:
        raise DateParsingError(
            message=f"Datumsspalte '{date_column}' konnte nicht geparst werden",
            details=str(e),
            help_text="Stellen Sie sicher, dass die Datumsspalte das Format TT.MM.JJJJ verwendet.",
            original_exception=e
        )

def display_tab():
    """Upload-Tab mit verbesserter UX und Validation Dashboard."""
    
    UI.section_header("Daten hochladen", help_text="Laden Sie Ihre CSV-Datei hoch und konfigurieren Sie die Datenspalten")
    
    uploaded_file = st.file_uploader(
        "CSV-Datei auswählen",
        type="csv",
        help="Unterstützte Formate: CSV mit Semikolon (;) als Trennzeichen"
    )

    if uploaded_file:
        try:
            # Lade Daten und speichere DIREKT
            df = load_csv(uploaded_file)
            if df is None:  # Error handler returned None
                log_error("UploadTab", "CSV konnte nicht geladen werden", stage="load_csv")
                return
            
            # ✨ NEUE UX: Validation Dashboard verwenden
            upload_validation = DataValidator.validate_uploaded_file(df, uploaded_file.name)
            display_validation_results(upload_validation, show_metadata=False)
            
            if not upload_validation.is_valid:
                UI.error_message(
                    "Upload abgebrochen - bitte beheben Sie die kritischen Fehler",
                    show_recovery=False
                )
                log_error("UploadTab", "Upload-Validierung fehlgeschlagen", stage="validate_upload", errors=upload_validation.errors)
                return
            
            # ✨ SUCCESS FEEDBACK
            UI.success_message(
                f"Datei '{uploaded_file.name}' erfolgreich geladen ({len(df)} Zeilen, {len(df.columns)} Spalten)"
            )
            
            # ✨ NEUE UX: Quality Dashboard verwenden
            st.markdown("---")
            quality_report = DataValidator.assess_data_quality(df)
            display_quality_dashboard(quality_report, show_trends=True)
            
            st.session_state.df = df
            log_data_operation("Upload: DataFrame gespeichert", data_shape=(len(df), len(df.columns)))
        
            # ✨ PROFESSIONELLER: Section Header für Datumsspalte
            st.markdown("---")
            UI.section_header("Datumsspalte konfigurieren", help_text="Wählen Sie die Spalte mit den Zeitstempeln")
            
            # Dropdown für die Auswahl der Datumsspalte
            date_column = st.selectbox(
                "Datumsspalte",
                df.columns,
                index=None,
                help="Wählen Sie die Spalte, die die Zeitinformationen enthält"
            )
            
            if date_column is not None:
                st.session_state.date_column = date_column
                log_data_operation("Upload: Datumsspalte gewählt", data_shape=(len(df), len(df.columns)), column=date_column)
                
                # ✨ NEUE UX: Validation Dashboard
                date_validation = DataValidator.validate_date_column(df, date_column)
                display_validation_results(date_validation, show_metadata=False, expanded=False)
                
                if not date_validation.is_valid:
                    UI.error_message(
                        "Bitte wählen Sie eine andere Datumsspalte",
                        show_recovery=False
                    )
                    log_error("UploadTab", "Datumsspalten-Validierung fehlgeschlagen", stage="validate_date_column", column=date_column, errors=date_validation.errors)
                    return
                
                # Parse date column with error handling
                df = parse_date_column(df, date_column)
                if df is None:  # Error handler returned None
                    log_error("UploadTab", "Datumsspalte konnte nicht geparst werden", stage="parse_date_column", column=date_column)
                    return
                    
                st.session_state.df = df  # Update state with parsed dates
                st.session_state.start_date = df[date_column].min().date()
                st.session_state.end_date = df[date_column].max().date()
                
                # ✨ SUCCESS FEEDBACK
                UI.success_message(
                    f"Datumsspalte '{date_column}' erfolgreich konfiguriert (Zeitraum: {st.session_state.start_date} bis {st.session_state.end_date})"
                )
                
                log_data_operation(
                    "Upload: Datenzeitraum gesetzt",
                    data_shape=(len(df), len(df.columns)),
                    start_date=st.session_state.start_date,
                    end_date=st.session_state.end_date,
                )
            else:
                UI.info_message("Bitte wählen Sie eine Datumsspalte aus.")
                log_error("UploadTab", "Keine Datumsspalte gewählt", stage="select_date_column")
                return

            # ✨ PROFESSIONELLER: Section Header für Produktspalten
            st.markdown("---")
            UI.section_header("Produktspalten auswählen", help_text="Wählen Sie die Spalten mit den Absatzdaten")
            
            # Produktauswahl
            columns = [col for col in df.columns if col != date_column]
            selected_products = st.multiselect(
                "Produktspalten",
                columns,
                help="Wählen Sie eine oder mehrere Produktspalten für die Analyse"
            )
            
            if selected_products:
                # ✨ NEUE UX: Validation Dashboard
                product_validation = DataValidator.validate_product_columns(
                    df, selected_products, allow_negatives=False
                )
                display_validation_results(product_validation, show_metadata=True, expanded=False)
                
                if not product_validation.is_valid:
                    UI.error_message(
                        "Einige Produktspalten haben kritische Probleme - bitte überprüfen Sie die Fehler oben",
                        show_recovery=False
                    )
                    log_error("UploadTab", "Produktspalten-Validierung fehlgeschlagen", stage="validate_products", products=selected_products, errors=product_validation.errors)
                else:
                    # ✨ SUCCESS FEEDBACK
                    UI.success_message(
                        f"{len(selected_products)} Produktspalte(n) erfolgreich konfiguriert"
                    )
                    
                    # ✨ NEUE UX: Quality Dashboard für Produktspalten
                    st.markdown("---")
                    quality_report = DataValidator.assess_data_quality(
                        df, date_col=date_column, product_cols=selected_products
                    )
                    display_quality_dashboard(quality_report, show_trends=False)
                
                st.session_state.selected_products_in_data = selected_products
                log_data_operation(
                    "Upload: Produkte ausgewählt",
                    data_shape=(len(df), len(selected_products)),
                    products=selected_products,
                )
            else:
                log_data_operation("Upload: Keine Produkte ausgewählt", data_shape=(len(df), 0))

            # Date filter
            if st.session_state.date_column:
                start_date, end_date = create_date_filter("upload", use_selected_keys=False)
                
                # Speichere die ausgewählten Daten
                st.session_state.start_date_selected = start_date
                st.session_state.end_date_selected = end_date
                log_data_operation(
                    "Upload: Zeitraum ausgewählt",
                    data_shape=(len(df), len(df.columns)),
                    start_date_selected=start_date,
                    end_date_selected=end_date,
                )

                if start_date and end_date:
                    filtered_df = filter_df(start_date, end_date)
                    log_data_operation(
                        "Upload: Daten gefiltert",
                        data_shape=(len(filtered_df), len(filtered_df.columns)),
                    )
                    with st.expander("Tabelle anzeigen"):
                        st.dataframe(filtered_df)
        except Exception as exc:
            log_error("UploadTab", str(exc), stage="display_tab_exception")
            raise

    # Prüfe und zeige Status
    if ready_for_processing():
        UserFeedback.success(f"Daten konfiguriert: {len(st.session_state.selected_products_in_data)} Produkte ausgewählt")
    else:
        UserFeedback.info("Bitte lade eine CSV-Datei hoch und wähle Datum sowie Produkte aus.")