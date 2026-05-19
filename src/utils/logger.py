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
        console: bool = True,
) -> logging.Logger:

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    # Load config overrides if a config file was provided
    if config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logging_cfg = config.get("logging", {})
            level      = logging_cfg.get("level",   level)
            log_format = logging_cfg.get(
                "format",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            if not log_file:
                log_file = logging_cfg.get("file", None)
            console = logging_cfg.get("console", console)
        except (FileNotFoundError, KeyError, TypeError):
            # Config unavailable -- fall through to defaults silently.
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Resolve numeric log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(log_format)

    # Console handler (stdout)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


class LoggerMixin:

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger