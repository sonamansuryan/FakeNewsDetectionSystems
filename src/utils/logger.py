"""Logging utilities for the project."""

import logging
import sys
from pathlib import Path
from typing import Optional
import yaml


def setup_logger(
        name: str,
        config_path: Optional[str] = None,
        log_file: Optional[str] = None,
        level: str = "INFO",
        console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        config_path: Path to config file
        log_file: Path to log file
        level: Logging level
        console: Whether to log to console

    Returns:
        Configured logger instance
    """
    # Load config if provided
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging_config = config.get('logging', {})
            level = logging_config.get('level', level)
            log_format = logging_config.get('format',
                                            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            if not log_file:
                log_file = logging_config.get('file', 'logs/training.log')
            console = logging_config.get('console', console)
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger