"""
Logging Configuration
=====================

Centralized logging setup for the application.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler

from ..config.settings import get_settings


def setup_logger(
    name: str = "wealth_manager",
    level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Set up application logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_to_file: Whether to log to file
        
    Returns:
        Configured logger
    """
    settings = get_settings()
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with Rich
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False
    )
    console_handler.setLevel(level)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_file = settings.LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
