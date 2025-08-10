#!/usr/bin/env python3
"""
MCP Integration Demo with Existing LangGraph Agents

This example demonstrates how the existing LangGraph agents in app/graphs/
now have MCP tools available by default. No changes needed to the agents!
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.tools import get_tool_belt, get_core_tool_belt
from app.graphs.simple_agent import graph as simple_agent
from app.graphs.agent_with_helpfulness import graph as helpful_agent
from langchain_core.messages import HumanMessage


def demo_simple_agent_with_mcp():
    """Demo: Simple agent now has MCP tools available."""
    print("📋 Demo: Simple Agent with MCP Tools")
    print("=" * 45)
    
    # Create sample data first
    csv_data = """product,category,price,rating
                    Laptop,Electronics,999.99,4.5
                    Book,Education,29.99,4.8
                    Chair,Furniture,199.99,4.2
                    Phone,Electronics,699.99,4.6
                    Desk,Furniture,299.99,4.3
                    Tablet,Electronics,399.99,4.4"""
    
    print("Using the existing simple_agent from app/graphs/simple_agent.py")
    print("This agent now automatically has access to all MCP tools!\n")
    
    # Task: Create and analyze a CSV file using MCP tools
    query = f"""
                I need you to help me analyze some product data using your file and data analysis tools. Please:
                
                1. Create a CSV file called 'examples/products.csv' with this data:
                {csv_data}
                
                2. Analyze the CSV file to get a summary of the data
                
                3. Calculate statistics for the price and rating columns
                
                4. Tell me what insights you can derive from this data
                """
    
    print(f"Query: {query[:100]}...")
    
    try:
        # Use the existing simple agent - it already has MCP tools!
        initial_state = {"messages": [HumanMessage(content=query)]}
        result = simple_agent.invoke(initial_state)
        
        print("\n🤖 Simple Agent Response:")
        print("-" * 30)
        for message in result["messages"]:
            if hasattr(message, 'content') and message.content:
                print(message.content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error running simple agent: {e}")
        return False


def demo_helpful_agent_with_mcp():
    """Demo: Helpful agent (with loops) now has MCP tools available."""
    print("\n🔄 Demo: Helpful Agent with MCP Tools")
    print("=" * 45)
    
    print("Using the existing agent_with_helpfulness from app/graphs/agent_with_helpfulness.py")
    print("This agent also automatically has access to all MCP tools!\n")
    
    query = """
            Please help me with a system analysis task using your available tools:
            
            1. Get the current system environment information
            2. List the contents of the current directory  
            3. Create a system report file called 'examples/system_report.txt' with all findings
            4. Validate if https://httpbin.org/json is accessible
            
            Make sure your response is helpful and complete before finishing.
            """
    
    print(f"Query: {query[:100]}...")
    
    try:
        # Use the existing helpful agent - it already has MCP tools!
        initial_state = {"messages": [HumanMessage(content=query)]}
        result = helpful_agent.invoke(initial_state)
        
        print("\n🤖 Helpful Agent Response:")
        print("-" * 30)
        for message in result["messages"]:
            if hasattr(message, 'content') and message.content:
                print(message.content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error running helpful agent: {e}")
        return False


def demo_tool_comparison():
    """Demo comparing core tools vs MCP-enhanced tools."""
    print("\n🔍 Tool Comparison Demo")
    print("=" * 30)
    
    # Get core tools
    core_tools = get_core_tool_belt()
    print(f"Core tools: {len(core_tools)}")
    for i, tool in enumerate(core_tools, 1):
        name = getattr(tool, 'name', 'Unknown')
        desc = getattr(tool, 'description', 'No description')
        print(f"   {i}. {name}: {desc[:60]}...")
    
    # Get all tools including MCP
    all_tools = get_tool_belt(include_mcp=True)
    mcp_tools = all_tools[len(core_tools):]
    
    print(f"\nMCP tools added: {len(mcp_tools)}")
    for i, tool in enumerate(mcp_tools, 1):
        name = getattr(tool, 'name', 'Unknown')
        desc = getattr(tool, 'description', 'No description')
        print(f"   {i}. {name}: {desc[:60]}...")
    
    print(f"\nTotal tools available: {len(all_tools)}")


def main():
    """Run demos with your existing LangGraph agents that now have MCP tools."""
    print("🚀 Existing LangGraph Agents + MCP Tools Demo")
    print("=" * 55)
    print("This demo shows your existing agents now have MCP tools automatically!\n")
    
    # Check if we have required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found. Agent demos will fail.")
        print("   Set your OpenAI API key to run the agent demos.")
        print("   You can still run the tool comparison demo below.\n")
    
    try:
        # Show tool comparison first
        demo_tool_comparison()
        
        if not os.getenv('OPENAI_API_KEY'):
            print("\n⚠️  Skipping agent demos (no API key)")
            return 0
        
        # Run demos with existing agents
        print("\n" + "="*55)
        print("Testing your existing agents with new MCP capabilities...")
        print("="*55)
        
        success_count = 0
        total_demos = 2
        
        if demo_simple_agent_with_mcp():
            success_count += 1
        
        if demo_helpful_agent_with_mcp():
            success_count += 1
        
        print(f"\n{'='*55}")
        print(f"Demo Summary: {success_count}/{total_demos} agent demos completed successfully")
        
        if success_count == total_demos:
            print("✅ Your existing agents now have MCP superpowers!")
            print("\nGenerated files (you can review them):")
            print("   - examples/products.csv")
            print("   - examples/system_report.txt")
            print("\n🎉 No changes needed to your agents - they automatically got 8 new tools!")
        else:
            print("⚠️  Some demos encountered issues. Check the output above.")
        
        return 0 if success_count == total_demos else 1
        
    except Exception as e:
        print(f"❌ Demo suite failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
