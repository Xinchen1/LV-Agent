"""
Logging initialization and configuration for OpenMythos Agent.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Set up logging configuration based on the provided LoggingConfig.
    
    This should be called early in the application lifecycle to ensure
    all logging uses the configured settings.
    
    Args:
        config: LoggingConfig instance from the application configuration
    """
    # Clear any existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Prepare formatters
    if config.output_mode == "json":
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s, '
            '"module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
        )
    elif config.output_mode == "plain" or not RICH_AVAILABLE:
        # Plain text formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Rich formatter (default when available and output_mode is "auto" or "rich")
        formatter = None  # RichHandler uses its own formatting
    
    # Console handler
    if config.console:
        if config.output_mode == "json":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
        elif RICH_AVAILABLE and config.rich_markup and config.output_mode != "plain":
            # Use RichHandler for nice colored output
            console_handler = RichHandler(
                show_time=True,
                show_level=True,
                show_path=True,
                enable_link_path=True,
                markup=True
            )
            # RichHandler doesn't use a formatter in the traditional sense
            formatter = None
        else:
            # Standard console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
        
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.file:
        # Ensure log directory exists
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to prevent log files from growing too large
        file_handler = logging.handlers.RotatingFileHandler(
            config.file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        if config.output_mode == "json":
            file_handler.setFormatter(formatter)
        else:
            # File logs are usually plain text for easy parsing
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s '
                '[%(module)s.%(funcName)s:%(lineno)d]'
            )
            file_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(file_handler)
    
    # Prevent duplicate logging in Jupyter notebooks and similar environments
    root_logger.propagate = False
    
    # Log that logging has been configured
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={config.level}, "
        f"file={config.file}, console={config.console}, "
        f"output_mode={config.output_mode}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    This is a convenience function that ensures consistent logger naming.
    
    Args:
        name: Logger name (typically __name__ from the calling module)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
