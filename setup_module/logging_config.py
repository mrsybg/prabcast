# setup_module/logging_config.py
"""
Logging-Konfiguration für PrABCast.
Bietet strukturiertes Logging für Debugging und Monitoring.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class PrABCastLogger:
    """Zentrale Logging-Konfiguration für PrABCast."""
    
    _initialized = False
    
    @classmethod
    def setup(
        cls,
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_dir: Optional[Path] = None
    ):
        """
        Konfiguriert das Logging-System.
        
        Args:
            log_level: Logging-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Ob in Datei geloggt werden soll
            log_to_console: Ob in Console geloggt werden soll
            log_dir: Verzeichnis für Log-Dateien
        """
        if cls._initialized:
            return
        
        # Root Logger für PrABCast
        logger = logging.getLogger('prabcast')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Format für Log-Messages
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File Handler
        if log_to_file:
            if log_dir is None:
                log_dir = Path.cwd() / 'logs'
            
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)
            
            # Rotierendes Log-File mit Datum
            log_file = log_dir / f"prabcast_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # File bekommt alle Details
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        cls._initialized = True
        logger.info("=== PrABCast Logging initialized ===")
    
    @classmethod
    def get_logger(cls, name: str = 'prabcast') -> logging.Logger:
        """
        Gibt einen konfigurierten Logger zurück.
        
        Args:
            name: Name des Loggers
            
        Returns:
            Konfigurierter Logger
        """
        if not cls._initialized:
            cls.setup()
        
        return logging.getLogger(name)


# Convenience Functions

def log_function_call(func_name: str, **kwargs):
    """Loggt einen Funktionsaufruf mit Parametern."""
    logger = PrABCastLogger.get_logger()
    params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Called {func_name}({params_str})")


def log_data_operation(operation: str, data_shape: tuple, **details):
    """Loggt eine Datenoperation."""
    logger = PrABCastLogger.get_logger()
    details_str = ", ".join(f"{k}={v}" for k, v in details.items())
    logger.info(f"Data operation: {operation} | Shape: {data_shape} | {details_str}")


def log_model_training(model_name: str, data_shape: tuple, duration: float, **metrics):
    """Loggt Modell-Training."""
    logger = PrABCastLogger.get_logger()
    metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                           for k, v in metrics.items())
    logger.info(
        f"Model trained: {model_name} | "
        f"Data: {data_shape} | "
        f"Duration: {duration:.2f}s | "
        f"Metrics: {metrics_str}"
    )


def log_api_call(api_name: str, endpoint: str, status: str, duration: Optional[float] = None):
    """Loggt API-Aufrufe."""
    logger = PrABCastLogger.get_logger()
    duration_str = f"Duration: {duration:.2f}s" if duration else ""
    logger.info(f"API call: {api_name} | Endpoint: {endpoint} | Status: {status} | {duration_str}")


def log_error(error_type: str, message: str, **context):
    """Loggt einen Fehler mit Kontext."""
    logger = PrABCastLogger.get_logger()
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.error(f"{error_type}: {message} | Context: {context_str}")


def log_performance_metric(metric_name: str, value: float, unit: str = ""):
    """Loggt Performance-Metriken."""
    logger = PrABCastLogger.get_logger()
    logger.info(f"Performance: {metric_name} = {value:.4f} {unit}")


# Initialize logging on import
PrABCastLogger.setup()
