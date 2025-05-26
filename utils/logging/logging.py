# utils/logging/logger.py

import logging
import os
from datetime import datetime


def setup_logger(level="INFO", log_file=None):
    """
    Set up logging configuration

    Parameters:
    - level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - log_file: Optional log file path

    Returns:
    - Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger('tft_trading_bot')
    logger.setLevel(numeric_level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger