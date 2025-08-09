import logging
import os
from datetime import datetime
from pathlib import Path

# logs directory
LOGS_DIR = Path(os.getcwd()) / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# single log file per run
LOG_FILE = LOGS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

LOG_FORMAT = "[ %(asctime)s ] %(levelname)s %(name)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid adding handlers twice if imported multiple times
    if logger.handlers:
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
