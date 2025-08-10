"""
MCP Client for LangGraph Integration

This module provides a simplified client that converts MCP tool functions
to LangChain-compatible tools using the @tool decorator for proper type generation.
"""

import logging
from typing import List
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Simplified client for converting MCP tools to LangChain-compatible tools.
    """
    
    def __init__(self, server_script_path: str):
        """
        Initialize the MCP client.
        
        Args:
            server_script_path: Path to the MCP server script (used for reference)
        """
        self.server_script_path = Path(server_script_path)
    
    def get_langchain_tools(self) -> List:
        """
        Convert MCP tools to LangChain-compatible tools with proper schemas.
        
        Returns:
            List of LangChain tools
        """
        # Import the tool functions to get proper type hints
        from app import mcp_tools
        from typing import List
        
        # Create properly typed LangChain tools using @tool decorator
        @tool
        def read_file_content(file_path: str) -> str:
            """Read the contents of a file.
            
            Args:
                file_path: Path to the file to read
            """
            return mcp_tools.read_file_content(file_path)
        
        @tool 
        def write_file_content(file_path: str, content: str) -> str:
            """Write content to a file.
            
            Args:
                file_path: Path to the file to write
                content: Content to write to the file
            """
            return mcp_tools.write_file_content(file_path, content)
        
        @tool
        def list_directory_contents(directory_path: str = ".") -> str:
            """List the contents of a directory.
            
            Args:
                directory_path: Path to the directory to list (defaults to current directory)
            """
            return mcp_tools.list_directory_contents(directory_path)
        
        @tool
        def analyze_csv_data(file_path: str, operation: str = "summary") -> str:
            """Analyze CSV data with various operations.
            
            Args:
                file_path: Path to the CSV file
                operation: Type of analysis ('summary', 'columns', 'sample', 'stats')
            """
            return mcp_tools.analyze_csv_data(file_path, operation)
        
        @tool
        def get_current_time(timezone: str = "UTC") -> str:
            """Get the current date and time.
            
            Args:
                timezone: Timezone to use (currently only supports UTC)
            """
            return mcp_tools.get_current_time(timezone)
        
        @tool
        def validate_url(url: str) -> str:
            """Validate if a URL is properly formatted and accessible.
            
            Args:
                url: URL to validate
            """
            return mcp_tools.validate_url(url)
        
        @tool
        def get_environment_info() -> str:
            """Get system environment information."""
            return mcp_tools.get_environment_info()
        
        @tool
        def calculate_statistics(numbers: List[float], stat_type: str = "all") -> str:
            """Calculate statistics for a list of numbers.
            
            Args:
                numbers: List of numbers to analyze
                stat_type: Type of statistics ('mean', 'median', 'std', 'all')
            """
            return mcp_tools.calculate_statistics(numbers, stat_type)
        
        return [
            read_file_content,
            write_file_content, 
            list_directory_contents,
            analyze_csv_data,
            get_current_time,
            validate_url,
            get_environment_info,
            calculate_statistics
        ]


# Global MCP client instance
_mcp_client = None


def get_mcp_client() -> MCPClient:
    """
    Get or create the global MCP client instance.
    
    Returns:
        MCP client instance
    """
    global _mcp_client
    
    if _mcp_client is None:
        # Get the absolute path to the MCP server script
        current_dir = Path(__file__).parent
        server_path = current_dir / "mcp_server.py"
        _mcp_client = MCPClient(str(server_path))
    
    return _mcp_client


def get_mcp_tools() -> List:
    """
    Get MCP tools as LangChain-compatible tools.
    
    Returns:
        List of MCP tools ready for use in LangGraph
    """
    client = get_mcp_client()
    return client.get_langchain_tools()
