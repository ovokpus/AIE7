"""
Enhanced logging configuration for the main A2A agent server
This provides colored, detailed logging for demo purposes
"""

import logging
import sys
from datetime import datetime


class ColorFormatter(logging.Formatter):
    """Colored logging formatter for better visibility in terminal demos"""
    
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


def setup_enhanced_logging():
    """Set up enhanced logging for the main agent"""
    
    # Create main agent logger
    main_logger = logging.getLogger("MainAgent")
    main_logger.setLevel(logging.INFO)
    
    # Create executor logger 
    executor_logger = logging.getLogger("app.agent_executor")
    executor_logger.setLevel(logging.INFO)
    
    # Create agent logger
    agent_logger = logging.getLogger("app.agent")
    agent_logger.setLevel(logging.INFO)
    
    # Create console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColorFormatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    )
    
    # Add handlers to loggers
    for logger in [main_logger, executor_logger, agent_logger]:
        logger.handlers.clear()
        logger.addHandler(console_handler)
    
    # Log startup message
    main_logger.info("🚀 Enhanced logging enabled for A2A agent server")
    main_logger.info("🎯 Ready to receive requests from second agent")
    main_logger.info("📊 All agent interactions will be logged below")
    
    return main_logger


def log_request_received(query: str, persona_context: str = None):
    """Log when a request is received from the second agent"""
    logger = logging.getLogger("MainAgent")
    logger.info("=" * 80)
    logger.info("📥 NEW REQUEST FROM SECOND AGENT")
    logger.info("=" * 80)
    logger.info(f"📝 Query: {query}")
    if persona_context:
        logger.info(f"🎭 Persona context detected: {persona_context[:100]}...")
    logger.info("🔄 Starting processing...")


def log_tool_execution(tool_name: str, tool_input: str):
    """Log when tools are being executed"""
    logger = logging.getLogger("MainAgent")
    logger.info(f"🛠️  Executing {tool_name}")
    logger.info(f"📊 Tool input: {tool_input}")


def log_response_generated(response: str, processing_time: float = None):
    """Log when response is generated"""
    logger = logging.getLogger("MainAgent")
    logger.info("✅ Response generated successfully")
    logger.info(f"📝 Response length: {len(response)} characters")
    if processing_time:
        logger.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
    preview = response[:100] + "..." if len(response) > 100 else response
    logger.info(f"👀 Response preview: {preview}")
    logger.info("📤 Sending response back to second agent")
    logger.info("=" * 80)
