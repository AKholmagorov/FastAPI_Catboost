import logging
from logging.handlers import TimedRotatingFileHandler
import os


def setup_logger():
    log_dir = "logs"
    log_file = "app.log"

    os.makedirs("logs", exist_ok=True)

    log_path = os.path.join(log_dir, log_file)

    handler = TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=7, encoding='utf-8'
    )
    handler.suffix = "%Y-%m-%d"  # имя файла с датой

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger("api_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger
