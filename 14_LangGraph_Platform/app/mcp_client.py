"""
MCP Client for LangGraph Integration

This module provides a client to connect to the local MCP server and
convert MCP tools to LangChain-compatible tools for use in LangGraph.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Annotated

logger = logging.getLogger(__name__)


# MCPTool class removed - now using @tool decorator for proper type generation


class MCPClient:
    """
    Client for accessing MCP tools directly.
    """
    
    def __init__(self, server_script_path: str):
        """
        Initialize the MCP client.
        
        Args:
            server_script_path: Path to the MCP server script (used for reference)
        """
        self.server_script_path = Path(server_script_path)
        self.tools_cache: Dict[str, Any] = {}
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get the list of available tools from the MCP server.
        
        Returns:
            List of tool definitions
        """
        try:
            # This is a simplified approach - in a real MCP implementation,
            # you would use the proper MCP protocol for tool discovery
            
            # For now, we'll return our known tools from the server
            tools = [
                {
                    "name": "read_file_content",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                },
                {
                    "name": "write_file_content", 
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                },
                {
                    "name": "list_directory_contents",
                    "description": "List the contents of a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory_path": {
                                "type": "string",
                                "description": "Path to the directory to list (defaults to current directory)",
                                "default": "."
                            }
                        }
                    }
                },
                {
                    "name": "analyze_csv_data",
                    "description": "Analyze CSV data with various operations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            },
                            "operation": {
                                "type": "string",
                                "description": "Type of analysis ('summary', 'columns', 'sample', 'stats')",
                                "default": "summary"
                            }
                        },
                        "required": ["file_path"]
                    }
                },
                {
                    "name": "get_current_time",
                    "description": "Get the current date and time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Timezone to use (currently only supports UTC)",
                                "default": "UTC"
                            }
                        }
                    }
                },
                {
                    "name": "validate_url",
                    "description": "Validate if a URL is properly formatted and accessible",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to validate"
                            }
                        },
                        "required": ["url"]
                    }
                },
                {
                    "name": "get_environment_info",
                    "description": "Get system environment information",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "calculate_statistics",
                    "description": "Calculate statistics for a list of numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "numbers": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of numbers to analyze"
                            },
                            "stat_type": {
                                "type": "string",
                                "description": "Type of statistics ('mean', 'median', 'std', 'all')",
                                "default": "all"
                            }
                        },
                        "required": ["numbers"]
                    }
                }
            ]
            
            self.tools_cache = {tool["name"]: tool for tool in tools}
            return tools
            
        except Exception as e:
            logger.error(f"Error getting available tools: {e}")
            return []
    
    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Call a tool directly by importing and executing the function.
        
        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool
            
        Returns:
            Result from the tool execution
        """
        try:
            # Import the tool functions directly from the tools module
            from app import mcp_tools
            
            # Map tool names to their functions
            tool_functions = {
                'read_file_content': mcp_tools.read_file_content,
                'write_file_content': mcp_tools.write_file_content,
                'list_directory_contents': mcp_tools.list_directory_contents,
                'analyze_csv_data': mcp_tools.analyze_csv_data,
                'get_current_time': mcp_tools.get_current_time,
                'validate_url': mcp_tools.validate_url,
                'get_environment_info': mcp_tools.get_environment_info,
                'calculate_statistics': mcp_tools.calculate_statistics,
            }
            
            # Get the tool function
            tool_func = tool_functions.get(tool_name)
            if not tool_func:
                return f"Error: Tool '{tool_name}' not found. Available tools: {list(tool_functions.keys())}"
            
            # Call the function with the provided arguments
            result = tool_func(**args)
            return str(result)
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return f"Error: {str(e)}"
    
    def get_langchain_tools(self) -> List:
        """
        Convert MCP tools to LangChain-compatible tools with proper schemas.
        
        Returns:
            List of LangChain tools
        """
        tools = []
        
        # Import the tool functions to get proper type hints
        from app import mcp_tools
        
        # Create properly typed LangChain tools
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
        
        tools = [
            read_file_content,
            write_file_content, 
            list_directory_contents,
            analyze_csv_data,
            get_current_time,
            validate_url,
            get_environment_info,
            calculate_statistics
        ]
        
        return tools


# Global MCP client instance
_mcp_client: Optional[MCPClient] = None


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
