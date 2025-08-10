#!/usr/bin/env python3
"""
Local MCP Server with FastMCP

This server provides additional tools to complement the existing LangGraph toolbelt:
- File operations (read, write, list directory)
- Data analysis (CSV operations, basic statistics)
- Web utilities (URL validation, basic web scraping)
- System utilities (current time, environment info)
"""

from fastmcp import FastMCP
from app.mcp_tools import (
    read_file_content,
    write_file_content,
    list_directory_contents,
    analyze_csv_data,
    get_current_time,
    validate_url,
    get_environment_info,
    calculate_statistics
)

# Initialize the MCP server
mcp = FastMCP("LangGraphToolServer")

# Register tools with the MCP server
mcp.tool()(read_file_content)
mcp.tool()(write_file_content)
mcp.tool()(list_directory_contents)
mcp.tool()(analyze_csv_data)
mcp.tool()(get_current_time)
mcp.tool()(validate_url)
mcp.tool()(get_environment_info)
mcp.tool()(calculate_statistics)


if __name__ == "__main__":
    print("Starting LangGraph MCP Tool Server...")
    print("Available tools:")
    print("- read_file_content: Read file contents")
    print("- write_file_content: Write content to file")
    print("- list_directory_contents: List directory contents")
    print("- analyze_csv_data: Analyze CSV files")
    print("- get_current_time: Get current timestamp")
    print("- validate_url: Validate and check URL accessibility")
    print("- get_environment_info: Get system environment info")
    print("- calculate_statistics: Calculate statistical measures")
    print("\nServer running on stdio transport...")
    
    mcp.run(transport="stdio")
