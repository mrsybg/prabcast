# setup_module/exceptions.py
"""
Zentrale Exception-Klassen und Error Handling für PrABCast.
Bietet strukturierte, typsichere Exception-Hierarchie.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Schweregrad von Fehlern."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PrABCastException(Exception):
    """
    Basis-Exception für alle PrABCast-spezifischen Fehler.
    Bietet strukturierte Fehlerinformationen.
    """
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        help_text: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Args:
            message: Hauptfehlermeldung für den Benutzer
            severity: Schweregrad des Fehlers
            details: Zusätzliche technische Details
            help_text: Hilfetext für den Benutzer
            original_exception: Ursprüngliche Exception (falls Wrapper)
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.details = details or {}
        self.help_text = help_text
        self.original_exception = original_exception
    
    def get_user_message(self) -> str:
        """Gibt die benutzerfreundliche Fehlermeldung zurück."""
        return self.message
    
    def get_technical_details(self) -> str:
        """Gibt technische Details für Debugging zurück."""
        details_str = "\n".join(f"  {k}: {v}" for k, v in self.details.items())
        if self.original_exception:
            details_str += f"\n  Original: {type(self.original_exception).__name__}: {str(self.original_exception)}"
        return details_str if details_str else "Keine Details verfügbar"


# === Daten-bezogene Exceptions ===

class DataException(PrABCastException):
    """Basis-Exception für datenbezogene Fehler."""
    pass


class DataValidationError(DataException):
    """Fehler bei der Datenvalidierung."""
    
    def __init__(self, message: str, validation_errors: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            details={"validation_errors": validation_errors or {}},
            **kwargs
        )
        self.validation_errors = validation_errors or {}


class DataLoadError(DataException):
    """Fehler beim Laden von Daten."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            details={"file_path": file_path} if file_path else {},
            help_text="Bitte überprüfen Sie das Dateiformat und versuchen Sie es erneut.",
            **kwargs
        )


class DataFormatError(DataException):
    """Fehler im Datenformat."""
    
    def __init__(self, message: str, expected_format: Optional[str] = None, **kwargs):
        details = {"expected_format": expected_format} if expected_format else {}
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            details=details,
            help_text="Bitte stellen Sie sicher, dass Ihre Daten das erwartete Format haben.",
            **kwargs
        )


class MissingDataError(DataException):
    """Fehler wenn erforderliche Daten fehlen."""
    
    def __init__(self, message: str, missing_fields: Optional[list] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            details={"missing_fields": missing_fields or []},
            help_text="Bitte laden Sie zuerst die erforderlichen Daten hoch.",
            **kwargs
        )


class DateParsingError(DataException):
    """Fehler beim Parsen von Datumsangaben."""
    
    def __init__(self, message: str, date_value: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            details={"date_value": date_value} if date_value else {},
            help_text="Bitte verwenden Sie das Format DD.MM.YYYY (z.B. 01.01.2024)",
            **kwargs
        )


# === Modell-bezogene Exceptions ===

class ModelException(PrABCastException):
    """Basis-Exception für modellbezogene Fehler."""
    pass


class ModelTrainingError(ModelException):
    """Fehler beim Training eines Modells."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            details={"model_name": model_name} if model_name else {},
            help_text="Das Modell konnte nicht trainiert werden. Versuchen Sie ein anderes Modell oder passen Sie die Parameter an.",
            **kwargs
        )


class ModelPredictionError(ModelException):
    """Fehler bei der Prognose."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            details={"model_name": model_name} if model_name else {},
            help_text="Die Prognose konnte nicht erstellt werden. Bitte überprüfen Sie die Eingabedaten.",
            **kwargs
        )


class InsufficientDataError(ModelException):
    """Fehler wenn nicht genug Daten für Modelltraining vorhanden sind."""
    
    def __init__(
        self, 
        message: str, 
        required_samples: Optional[int] = None,
        available_samples: Optional[int] = None,
        **kwargs
    ):
        details = {}
        if required_samples:
            details["required_samples"] = required_samples
        if available_samples:
            details["available_samples"] = available_samples
            
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            details=details,
            help_text="Bitte laden Sie einen größeren Datensatz hoch oder wählen Sie einen kürzeren Prognosehorizont.",
            **kwargs
        )


# === API-bezogene Exceptions ===

class APIException(PrABCastException):
    """Basis-Exception für API-bezogene Fehler."""
    pass


class APIConnectionError(APIException):
    """Fehler bei der API-Verbindung."""
    
    def __init__(self, message: str, api_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            details={"api_name": api_name} if api_name else {},
            help_text="Bitte überprüfen Sie Ihre Internetverbindung und versuchen Sie es später erneut.",
            **kwargs
        )


class APIRateLimitError(APIException):
    """Fehler bei API Rate Limit."""
    
    def __init__(self, message: str, api_name: Optional[str] = None, retry_after: Optional[int] = None, **kwargs):
        details = {"api_name": api_name} if api_name else {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
            
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            details=details,
            help_text=f"Bitte warten Sie {retry_after} Sekunden und versuchen Sie es erneut." if retry_after else "Bitte versuchen Sie es später erneut.",
            **kwargs
        )


class APIDataError(APIException):
    """Fehler bei API-Daten."""
    
    def __init__(self, message: str, api_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            details={"api_name": api_name} if api_name else {},
            help_text="Die API-Daten konnten nicht abgerufen werden. Versuchen Sie andere Datenquellen.",
            **kwargs
        )


# === Konfiguration-bezogene Exceptions ===

class ConfigurationException(PrABCastException):
    """Basis-Exception für Konfigurationsfehler."""
    pass


class InvalidConfigurationError(ConfigurationException):
    """Fehler bei ungültiger Konfiguration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            details={"config_key": config_key} if config_key else {},
            help_text="Bitte überprüfen Sie Ihre Konfigurationseinstellungen.",
            **kwargs
        )


# === Visualisierung-bezogene Exceptions ===

class VisualizationException(PrABCastException):
    """Basis-Exception für Visualisierungsfehler."""
    pass


class PlotCreationError(VisualizationException):
    """Fehler bei der Erstellung von Plots."""
    
    def __init__(self, message: str, plot_type: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            details={"plot_type": plot_type} if plot_type else {},
            help_text="Die Visualisierung konnte nicht erstellt werden. Die Daten werden dennoch verarbeitet.",
            **kwargs
        )


# === Utility Functions ===

def wrap_exception(e: Exception, message: Optional[str] = None) -> PrABCastException:
    """
    Wraps eine Standard-Exception in eine PrABCastException.
    
    Args:
        e: Die zu wrappende Exception
        message: Optionale benutzerdefinierte Nachricht
        
    Returns:
        PrABCastException mit wrapped Exception
    """
    if isinstance(e, PrABCastException):
        return e
    
    # Mappe bekannte Exception-Typen
    if isinstance(e, (pd.errors.ParserError, pd.errors.EmptyDataError)):
        return DataLoadError(
            message or "Fehler beim Laden der CSV-Datei",
            original_exception=e
        )
    
    if isinstance(e, ValueError):
        return DataValidationError(
            message or "Ungültige Daten",
            original_exception=e
        )
    
    if isinstance(e, FileNotFoundError):
        return DataLoadError(
            message or "Datei nicht gefunden",
            original_exception=e
        )
    
    if isinstance(e, (ConnectionError, TimeoutError)):
        return APIConnectionError(
            message or "Verbindungsfehler",
            original_exception=e
        )
    
    # Fallback: Generische PrABCastException
    return PrABCastException(
        message or str(e),
        severity=ErrorSeverity.ERROR,
        original_exception=e
    )


# Importiere pandas für Exception-Mapping
try:
    import pandas as pd
except ImportError:
    pd = None
