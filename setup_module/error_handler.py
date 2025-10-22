# setup_module/error_handler.py
"""
Zentrales Error Handling und User Feedback System für PrABCast.
Bietet konsistente Fehlerbehandlung und Benutzer-Feedback.
"""

import streamlit as st
import logging
import traceback
from typing import Optional, Callable, Any, TypeVar
from functools import wraps
from .exceptions import (
    PrABCastException,
    ErrorSeverity,
    wrap_exception
)
from .session_state import get_app_state

# Setup Logger
logger = logging.getLogger('prabcast')

T = TypeVar('T')


class UserFeedback:
    """
    Zentrale Klasse für Benutzer-Feedback.
    Bietet konsistente UI für Fehler, Warnungen und Erfolgsmeldungen.
    """
    
    @staticmethod
    def show_exception(
        exception: PrABCastException,
        show_details: bool = False,
        show_help: bool = True
    ):
        """
        Zeigt eine Exception dem Benutzer an.
        
        Args:
            exception: Die anzuzeigende Exception
            show_details: Ob technische Details angezeigt werden sollen
            show_help: Ob Hilfetext angezeigt werden soll
        """
        # Wähle UI-Komponente basierend auf Severity
        if exception.severity == ErrorSeverity.ERROR or exception.severity == ErrorSeverity.CRITICAL:
            st.error(f"❌ {exception.get_user_message()}")
        elif exception.severity == ErrorSeverity.WARNING:
            st.warning(f"⚠️ {exception.get_user_message()}")
        else:  # INFO
            st.info(f"ℹ️ {exception.get_user_message()}")
        
        # Zeige Hilfetext wenn vorhanden
        if show_help and exception.help_text:
            st.info(f"💡 **Tipp:** {exception.help_text}")
        
        # Zeige technische Details in Expander
        if show_details and (exception.details or exception.original_exception):
            with st.expander("🔧 Technische Details"):
                st.code(exception.get_technical_details())
    
    @staticmethod
    def error(message: str, help_text: Optional[str] = None, details: Optional[str] = None):
        """Zeigt eine Fehlermeldung an."""
        st.error(f"❌ {message}")
        if help_text:
            st.info(f"💡 **Tipp:** {help_text}")
        if details:
            with st.expander("🔧 Details"):
                st.code(details)
    
    @staticmethod
    def warning(message: str, help_text: Optional[str] = None):
        """Zeigt eine Warnung an."""
        st.warning(f"⚠️ {message}")
        if help_text:
            st.info(f"💡 {help_text}")
    
    @staticmethod
    def success(message: str, next_step: Optional[str] = None):
        """Zeigt eine Erfolgsmeldung an."""
        st.success(f"✅ {message}")
        if next_step:
            st.info(f"➡️ **Nächster Schritt:** {next_step}")
    
    @staticmethod
    def info(message: str):
        """Zeigt eine Info-Meldung an."""
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def validation_errors(errors: dict, title: str = "Validierungsfehler"):
        """
        Zeigt Validierungsfehler strukturiert an.
        
        Args:
            errors: Dictionary mit Feld -> Fehlermeldung
            title: Titel der Fehlermeldung
        """
        st.error(f"❌ **{title}**")
        for field, error in errors.items():
            st.markdown(f"- **{field}**: {error}")
    
    @staticmethod
    def progress_with_feedback(
        items: list,
        process_func: Callable,
        success_message: str = "Verarbeitung abgeschlossen",
        error_message: str = "Fehler bei der Verarbeitung"
    ):
        """
        Zeigt Progress Bar mit Fehlerbehandlung.
        
        Args:
            items: Liste der zu verarbeitenden Items
            process_func: Funktion die jedes Item verarbeitet
            success_message: Nachricht bei Erfolg
            error_message: Nachricht bei Fehler
        """
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        errors = []
        
        for i, item in enumerate(items):
            try:
                status_text.text(f"Verarbeite {i+1}/{len(items)}...")
                result = process_func(item)
                results.append(result)
            except Exception as e:
                error = wrap_exception(e, f"Fehler bei Item {i+1}")
                errors.append(error)
                logger.error(f"Error processing item {i+1}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(items))
        
        # Zeige Ergebnisse
        if errors:
            UserFeedback.warning(
                f"{error_message}: {len(errors)} von {len(items)} fehlgeschlagen",
                f"{len(results)} Items erfolgreich verarbeitet"
            )
            with st.expander("Fehlerdetails anzeigen"):
                for error in errors:
                    UserFeedback.show_exception(error, show_details=True)
        else:
            UserFeedback.success(f"{success_message}: {len(results)} Items verarbeitet")
        
        return results, errors


class ErrorHandler:
    """
    Zentrale Error-Handler Klasse.
    Bietet Decorators und Context Manager für konsistente Fehlerbehandlung.
    """
    
    @staticmethod
    def handle_errors(
        show_details: bool = False,
        show_help: bool = True,
        log_error: bool = True,
        reraise: bool = False
    ):
        """
        Decorator für Fehlerbehandlung in Funktionen.
        
        Args:
            show_details: Ob technische Details angezeigt werden
            show_help: Ob Hilfetext angezeigt wird
            log_error: Ob Fehler geloggt werden
            reraise: Ob Exception nach Behandlung neu geworfen wird
        
        Usage:
            @ErrorHandler.handle_errors(show_details=True)
            def my_function():
                # Code that might raise exceptions
                pass
        """
        def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Optional[T]:
                try:
                    return func(*args, **kwargs)
                except PrABCastException as e:
                    # Unsere eigenen Exceptions
                    if log_error:
                        logger.error(f"{func.__name__}: {e.message}", exc_info=True)
                    
                    UserFeedback.show_exception(e, show_details=show_details, show_help=show_help)
                    
                    if reraise:
                        raise
                    return None
                    
                except Exception as e:
                    # Unerwartete Exceptions
                    wrapped = wrap_exception(e, f"Unerwarteter Fehler in {func.__name__}")
                    
                    if log_error:
                        logger.error(f"{func.__name__}: {str(e)}", exc_info=True)
                    
                    UserFeedback.show_exception(wrapped, show_details=True, show_help=show_help)
                    
                    if reraise:
                        raise
                    return None
            
            return wrapper
        return decorator
    
    @staticmethod
    def safe_execute(
        func: Callable[..., T],
        *args,
        fallback_value: Optional[T] = None,
        error_message: Optional[str] = None,
        show_error: bool = True,
        **kwargs
    ) -> Optional[T]:
        """
        Führt eine Funktion sicher aus mit Fehlerbehandlung.
        
        Args:
            func: Auszuführende Funktion
            fallback_value: Rückgabewert bei Fehler
            error_message: Optionale Error-Message
            show_error: Ob Fehler dem Benutzer angezeigt wird
            *args, **kwargs: Argumente für func
            
        Returns:
            Funktionsergebnis oder fallback_value bei Fehler
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wrapped = wrap_exception(e, error_message)
            logger.error(f"safe_execute failed: {str(e)}", exc_info=True)
            
            if show_error:
                UserFeedback.show_exception(wrapped)
            
            return fallback_value
    
    @staticmethod
    def validate_and_execute(
        validation_func: Callable[[], bool],
        action_func: Callable[..., T],
        validation_message: str = "Validierung fehlgeschlagen",
        *args,
        **kwargs
    ) -> Optional[T]:
        """
        Validiert zuerst, führt dann Action aus.
        
        Args:
            validation_func: Validierungsfunktion (gibt bool zurück)
            action_func: Auszuführende Funktion bei erfolgreicher Validierung
            validation_message: Nachricht bei Validierungsfehler
            *args, **kwargs: Argumente für action_func
            
        Returns:
            Ergebnis von action_func oder None
        """
        try:
            if not validation_func():
                UserFeedback.warning(validation_message)
                return None
            
            return ErrorHandler.safe_execute(action_func, *args, **kwargs)
        except Exception as e:
            wrapped = wrap_exception(e)
            UserFeedback.show_exception(wrapped)
            return None


class ErrorContext:
    """
    Context Manager für Error Handling.
    
    Usage:
        with ErrorContext("Datenverarbeitung"):
            # Code that might raise exceptions
            process_data()
    """
    
    def __init__(
        self,
        operation_name: str,
        show_details: bool = False,
        show_help: bool = True
    ):
        self.operation_name = operation_name
        self.show_details = show_details
        self.show_help = show_help
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True
        
        # Exception aufgetreten
        if issubclass(exc_type, PrABCastException):
            error = exc_val
        else:
            error = wrap_exception(exc_val, f"Fehler bei: {self.operation_name}")
        
        logger.error(f"{self.operation_name}: {str(exc_val)}", exc_info=True)
        UserFeedback.show_exception(error, self.show_details, self.show_help)
        
        # Suppress Exception (nicht weiter propagieren)
        return True


# Convenience Functions

def show_error(message: str, **kwargs):
    """Shortcut für Fehleranzeige."""
    UserFeedback.error(message, **kwargs)


def show_warning(message: str, **kwargs):
    """Shortcut für Warnung."""
    UserFeedback.warning(message, **kwargs)


def show_success(message: str, **kwargs):
    """Shortcut für Erfolgsmeldung."""
    UserFeedback.success(message, **kwargs)


def show_info(message: str):
    """Shortcut für Info."""
    UserFeedback.info(message)


def safe_call(func: Callable, *args, fallback=None, **kwargs):
    """Shortcut für safe_execute."""
    return ErrorHandler.safe_execute(func, *args, fallback_value=fallback, **kwargs)
