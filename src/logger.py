# src/logging_setup.py

import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name):
    # Ensure the logs directory exists
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        # Create rotating file handler (5MB per file, max 3 files)
        handler = RotatingFileHandler(os.path.join(log_dir, 'eda_log.log'), maxBytes=5*1024*1024, backupCount=3)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Example usage
logger = get_logger('EDA_Logger')
logger.info("Starting the exploratory data analysis...")
