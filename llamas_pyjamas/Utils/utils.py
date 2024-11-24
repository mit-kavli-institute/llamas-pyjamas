# llamas_pyjamas/utils.py
import os
import logging
from datetime import datetime

def setup_logger(name, log_filename=None):
    """
    Setup logger with file and console handlers
    Args:
        name: Logger name (usually __name__)
        log_filename: Optional custom log filename
    """
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create file handler
    log_file = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add only file handler
    logger.addHandler(file_handler)
    
    return logger