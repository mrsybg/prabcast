# utils.py
import numpy as np
import tensorflow as tf
import random
import os
import logging

def set_seed(seed=42):
    """
    Setzt die Zufallskeime für Reproduzierbarkeit.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging(log_file='forecasting.log'):
    """
    Richtet das Logging ein.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
