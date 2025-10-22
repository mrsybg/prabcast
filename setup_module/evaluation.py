# evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import psutil
import tracemalloc

def measure_performance(func):
    """Decorator to measure execution time and memory usage"""
    def wrapper(*args, **kwargs):
        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure time
        execution_time = time.time() - start_time
        
        # Measure memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return result, execution_time, peak / 10**6  # Convert to MB
    return wrapper

def calculate_additional_metrics(y_true, y_pred):
    """
    Berechnet zusätzliche Metriken wie Bias und Theil's U.
    """
    bias = np.mean(y_pred - y_true)
    # Verhindern von Division durch Null
    denominator = np.sqrt(mean_squared_error(y_true, np.roll(y_true, 1))) if len(y_true) > 1 else 1
    theils_u = np.sqrt(mean_squared_error(y_true, y_pred)) / denominator
    return bias, theils_u

def calculate_metrics(y_true, y_pred, execution_time=None, memory_used=None):
    """
    Berechnet verschiedene Evaluationsmetriken.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    bias, theils_u = calculate_additional_metrics(y_true, y_pred)
    
    metrics = {
        'MAE': mae, 
        'RMSE': rmse, 
        'sMAPE': smape, 
        'Bias': bias, 
        'Theils_U': theils_u
    }
    
    if execution_time is not None:
        metrics['Training_Time_s'] = execution_time
    if memory_used is not None:
        metrics['Memory_Usage_MB'] = memory_used
        
    return metrics