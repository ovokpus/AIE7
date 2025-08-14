# second_agent/modules/logging_config.py
"""
Enhanced logging configuration for the Expert Agent System
"""
import logging


class ColorFormatter(logging.Formatter):
    """Colored logging formatter for better visibility"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"\033[94m{record.name}\033[0m"  # Blue
        return super().format(record)


def setup_expert_logging():
    """Set up enhanced logging for the Expert Agent System"""
    
    # Basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Create colored console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    ))

    # Set up main agent logger
    logger = logging.getLogger("SecondAgent")
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    # Set up interaction logger
    interaction_logger = logging.getLogger("Interaction")
    interaction_logger.handlers.clear()
    interaction_logger.addHandler(console_handler)
    interaction_logger.setLevel(logging.INFO)

    return logger, interaction_logger


def get_loggers():
    """Get the configured loggers"""
    return logging.getLogger("SecondAgent"), logging.getLogger("Interaction")
