# Logger configuration and setup

import logging
import os
from logging.handlers import RotatingFileHandler

# Define the log directory and ensure it exists
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str = __name__, log_file: str = None) -> logging.Logger:
    """
    Creates and returns a logger with the specified name.

    Args:
        name (str): The name of the logger. Defaults to the module's __name__.
        log_file (str): Optional log file name. If None, uses the default 'src.log'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all types of logs

    if not logger.handlers:
        # Define log format
        formatter = logging.Formatter(
            fmt='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler for outputting logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set to INFO for less verbose console output
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Determine the log file to use
        if log_file is None:
            log_file = os.path.join(LOG_DIR, 'src.log')
        else:
            log_file = os.path.join(LOG_DIR, log_file)

        # File handler for writing logs to a file with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3  # Keep up to 3 backup log files
        )
        file_handler.setLevel(logging.DEBUG)  # Capture all logs in the file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Prevent log messages from being propagated to the root logger multiple times
        logger.propagate = False

    return logger
