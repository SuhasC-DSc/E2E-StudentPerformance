import logging
import os
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs")
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] [%(filename)s:%(lineno)d] %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filemode='w'
)

logging.info("Logger initialized")