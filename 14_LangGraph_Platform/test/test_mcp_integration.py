#!/usr/bin/env python3
"""
Test script for MCP server integration with LangGraph.

This script tests:
1. MCP server startup
2. Tool discovery and loading
3. Basic tool functionality
4. Integration with LangGraph
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.mcp_client import get_mcp_client, get_mcp_tools
from app.tools import get_tool_belt, get_core_tool_belt
from app.state import AgentState
from app.models import get_chat_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mcp_tools_available():
    """Test that MCP tools are available and discoverable."""
    print("\n🔧 Testing MCP Tools Availability...")
    
    try:
        client = get_mcp_client()
        
        # Test tool discovery
        tools = client.get_available_tools()
        if tools:
            print(f"✅ Found {len(tools)} available tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
            return True
        else:
            print("❌ No tools found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MCP tools: {e}")
        return False


def test_tool_integration():
    """Test that MCP tools integrate properly with the tool belt."""
    print("\n🔨 Testing Tool Integration...")
    
    try:
        # Test core tools only
        core_tools = get_core_tool_belt()
        print(f"✅ Core tools loaded: {len(core_tools)} tools")
        
        # Test with MCP tools
        all_tools = get_tool_belt(include_mcp=True)
        print(f"✅ All tools loaded: {len(all_tools)} tools")
        
        mcp_tool_count = len(all_tools) - len(core_tools)
        print(f"✅ MCP tools added: {mcp_tool_count} tools")
        
        # List all tools
        print("\nAll available tools:")
        for i, tool in enumerate(all_tools, 1):
            tool_name = getattr(tool, 'name', 'Unknown')
            tool_desc = getattr(tool, 'description', 'No description')
            print(f"   {i}. {tool_name}: {tool_desc}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tool integration: {e}")
        return False


def test_mcp_tool_execution():
    """Test that MCP tools can be executed."""
    print("\n⚡ Testing MCP Tool Execution...")
    
    try:
        client = get_mcp_client()
        
        # Test get_current_time tool
        print("Testing get_current_time tool...")
        result = client.call_tool("get_current_time", {})
        print(f"✅ get_current_time result: {result}")
        
        # Test get_environment_info tool
        print("\nTesting get_environment_info tool...")
        result = client.call_tool("get_environment_info", {})
        print(f"✅ get_environment_info result: {result[:100]}...")
        
        # Test list_directory_contents tool
        print("\nTesting list_directory_contents tool...")
        result = client.call_tool("list_directory_contents", {"directory_path": "."})
        print(f"✅ list_directory_contents result: Found directory listing")
        
        # Test calculate_statistics tool
        print("\nTesting calculate_statistics tool...")
        result = client.call_tool("calculate_statistics", {
            "numbers": [1, 2, 3, 4, 5], 
            "stat_type": "all"
        })
        print(f"✅ calculate_statistics result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tool execution: {e}")
        return False


def test_langchain_tool_format():
    """Test that MCP tools work in LangChain format."""
    print("\n🔗 Testing LangChain Tool Format...")
    
    try:
        mcp_tools = get_mcp_tools()
        
        if not mcp_tools:
            print("❌ No MCP tools found")
            return False
        
        # Test that each tool has the required LangChain properties
        for tool in mcp_tools:
            if not hasattr(tool, 'name'):
                print(f"❌ Tool missing 'name' attribute: {tool}")
                return False
            if not hasattr(tool, 'description'):
                print(f"❌ Tool missing 'description' attribute: {tool}")
                return False
            if not hasattr(tool, '_run'):
                print(f"❌ Tool missing '_run' method: {tool}")
                return False
        
        print(f"✅ All {len(mcp_tools)} MCP tools have proper LangChain format")
        
        # Test executing one tool
        print("\nTesting tool execution via LangChain interface...")
        time_tool = None
        for tool in mcp_tools:
            if tool.name == "get_current_time":
                time_tool = tool
                break
        
        if time_tool:
            # Use invoke() method instead of _run() for proper LangChain interface
            result = time_tool.invoke({})
            print(f"✅ LangChain tool execution successful: {result}")
        else:
            print("⚠️  get_current_time tool not found for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LangChain format: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting MCP Integration Tests\n")
    
    tests = [
        ("MCP Tools Availability", test_mcp_tools_available),
        ("Tool Integration", test_tool_integration),
        ("MCP Tool Execution", test_mcp_tool_execution),
        ("LangChain Tool Format", test_langchain_tool_format),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! MCP integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
