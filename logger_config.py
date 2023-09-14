import os
import logging
from logging.handlers import RotatingFileHandler

log_dir = "logs"

def setup_logging():
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'advsecurenet.log')
    handler = RotatingFileHandler(log_file, maxBytes=15*1024*1024, backupCount=3)  # Rotate after 15MB, keep 3 backup copies
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
