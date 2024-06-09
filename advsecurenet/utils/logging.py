# logging_config.py
import logging.config
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingConfig:
    """
    This dataclass is used to store the logging configuration.

    Attributes:
        log_dir (str): The log directory.
        log_file (str): The name of the log file.
        level (str): The log level. The hierarchy of log levels is as follows: DEBUG < INFO < WARNING < ERROR < CRITICAL.
        disable_existing_loggers (bool): Whether to disable existing loggers.
        formatters (dict): The log formatters.
        version (int): The logging configuration version.
    """
    log_dir: str = field(
        default_factory=lambda: os.path.join(os.path.expanduser('~'), 'logs')
    )
    log_file: str = 'advsecurenet.log'
    level: str = 'INFO'
    disable_existing_loggers: bool = False
    formatters: dict = field(default_factory=lambda: {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    })
    version: int = 1


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Setup logging configuration.

    Args:
        config (LoggingConfig): The logging configuration.
    """
    if config is None:
        config = LoggingConfig()
    # Create log directory if it doesn't exist
    os.makedirs(config.log_dir, exist_ok=True)

    # Define the logging configuration
    logging_config = {
        'version': config.version,
        'disable_existing_loggers': config.disable_existing_loggers,
        'formatters': config.formatters,
        'handlers': {
            'file_handler': {
                'class': 'logging.FileHandler',
                'filename': os.path.join(config.log_dir, config.log_file),
                'formatter': 'standard',
            },
        },
        'loggers': {
            '': {
                'handlers': ['file_handler'],
                'level': config.level,
                'propagate': True
            },
        }
    }

    # Apply the logging configuration
    logging.config.dictConfig(logging_config)
