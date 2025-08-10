#!/usr/bin/env python3
"""
Standalone MCP Server Runner

This script runs the MCP server as a standalone service that other applications
can connect to. The current LangGraph integration uses direct function calls,
but this server can be useful for:
- Testing MCP protocol compliance
- Connecting from other applications
- Running as a microservice
"""

import sys
import signal
import logging
from pathlib import Path

# Add the app directory to path
sys.path.append(str(Path(__file__).parent))

from app.mcp_server import mcp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}. Shutting down MCP server...")
    sys.exit(0)


def main():
    """Run the MCP server."""
    print("🚀 Starting Standalone MCP Server")
    print("=" * 40)
    print("This MCP server provides 8 tools for:")
    print("📁 File operations (read, write, list)")
    print("📊 Data analysis (CSV analysis, statistics)")  
    print("🌐 Web utilities (URL validation)")
    print("⚙️ System utilities (environment, time)")
    print()
    print("Available tools:")
    
    # List available tools
    tool_descriptions = [
        "read_file_content - Read file contents",
        "write_file_content - Write content to file", 
        "list_directory_contents - List directory contents",
        "analyze_csv_data - Analyze CSV files",
        "get_current_time - Get current timestamp",
        "validate_url - Validate URL accessibility",
        "get_environment_info - Get system environment",
        "calculate_statistics - Calculate number statistics"
    ]
    
    for i, desc in enumerate(tool_descriptions, 1):
        print(f"  {i}. {desc}")
    
    print()
    print("Server running on stdio transport...")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run the MCP server
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        logger.info("MCP server stopped")


if __name__ == "__main__":
    main()
