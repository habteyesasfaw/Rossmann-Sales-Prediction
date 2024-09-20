import logging
import os

def get_logger(name):
    # Ensure the logs directory exists
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Add logging handler if not already present
    # if not logger.hasHandlers():
    try:
        handler = logging.FileHandler(os.path.join(log_dir, 'eda_log.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("Logger initialized and ready to record.")
    except Exception as e:
        print(f"Failed to create log file: {e}")

    return logger

# Example usage
logger = get_logger('EDA_Logger')
logger.info("Starting the exploratory data analysis...")
