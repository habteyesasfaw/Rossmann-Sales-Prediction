import logging
import os

def get_logger(name, log_file='app.log', level=logging.INFO):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f"{log_dir}/{log_file}")
    file_handler.setFormatter(formatter)
    
    logger.setLevel(level)
    logger.addHandler(file_handler)
    
    return logger
