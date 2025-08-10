#!/usr/bin/env python3
"""
Interactive LangGraph + MCP Demo

This interactive demo lets you chat with your existing LangGraph agents
that now have MCP tools available. You can ask them to perform file operations,
data analysis, web validation, and system tasks.
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


def print_banner():
    """Print the welcome banner."""
    print("🚀 Interactive LangGraph + MCP Tools Demo")
    print("=" * 50)
    print("Chat with your existing LangGraph agents that now have MCP superpowers!")
    print("\nAvailable agents:")
    print("1. Simple Agent (quick responses)")
    print("2. Helpful Agent (with quality loops)")
    print("\nMCP Tools available:")
    print("📁 File operations, 📊 Data analysis, 🌐 Web validation, ⚙️ System info")
    print("=" * 50)


def show_tool_capabilities():
    """Show what tools are available."""
    print("\n🔧 Available Tools:")
    print("-" * 20)
    
    core_tools = get_core_tool_belt()
    all_tools = get_tool_belt(include_mcp=True)
    mcp_tools = all_tools[len(core_tools):]
    
    print("Core Tools:")
    for i, tool in enumerate(core_tools, 1):
        name = getattr(tool, 'name', 'Unknown')
        desc = getattr(tool, 'description', 'No description')
        print(f"  {i}. {name}: {desc[:60]}...")
    
    print(f"\nMCP Tools ({len(mcp_tools)} new tools):")
    for i, tool in enumerate(mcp_tools, 1):
        name = getattr(tool, 'name', 'Unknown')
        desc = getattr(tool, 'description', 'No description')
        print(f"  {i}. {name}: {desc[:60]}...")
    
    print(f"\nTotal: {len(all_tools)} tools available to your agents!")


def show_example_prompts():
    """Show example prompts users can try."""
    print("\n💡 Example things you can ask:")
    print("-" * 30)
    
    examples = [
        "📊 Create a CSV file with sales data and analyze it",
        "📁 List the files in the current directory", 
        "🕒 Get the current time and save it to a file",
        "🌐 Check if https://httpbin.org/json is accessible",
        "📈 Calculate statistics for these numbers: [10, 20, 30, 40, 50]",
        "💾 Create a system report with environment information",
        "🔍 Search for information about Python and save key findings",
        "📋 Help me organize project files and create a summary"
    ]
    
    for example in examples:
        print(f"  • {example}")
    
    print("\nType 'help' anytime to see this again!")


def chat_with_agent(agent_graph, agent_name):
    """Interactive chat session with an agent."""
    print(f"\n🤖 Starting chat with {agent_name}")
    print("Type 'quit' to exit, 'switch' to change agents, 'help' for examples")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n💬 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                return 'quit'
            
            if user_input.lower() in ['switch', 's']:
                print("🔄 Switching agents...")
                return 'switch'
                
            if user_input.lower() in ['help', 'h']:
                show_example_prompts()
                continue
                
            if user_input.lower() in ['tools', 't']:
                show_tool_capabilities()
                continue
            
            # Process with the agent
            print(f"\n🤖 {agent_name} is thinking...")
            
            initial_state = {"messages": [HumanMessage(content=user_input)]}
            result = agent_graph.invoke(initial_state)
            
            print(f"\n🤖 {agent_name}:")
            print("-" * 20)
            
            # Display the response
            for message in result["messages"]:
                if hasattr(message, 'content') and message.content:
                    # Don't repeat the user's input
                    if message.content != user_input:
                        print(message.content)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted. Type 'quit' to exit properly.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Try a different question or type 'help' for examples.")


def select_agent():
    """Let user select which agent to chat with."""
    while True:
        print("\n🤖 Select an agent:")
        print("1. Simple Agent (fast, direct responses)")
        print("2. Helpful Agent (thorough, with quality checks)")
        print("3. Show tool capabilities")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            return simple_agent, "Simple Agent"
        elif choice == '2':
            return helpful_agent, "Helpful Agent"
        elif choice == '3':
            show_tool_capabilities()
            continue
        elif choice == '4':
            return None, None
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


def main():
    """Run the interactive demo."""
    print_banner()
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️ WARNING: OPENAI_API_KEY not found!")
        print("Set your OpenAI API key to use the agents:")
        print("export OPENAI_API_KEY='your-key-here'")
        print("\nYou can still see tool capabilities with option 3.")
        print()
    
    show_example_prompts()
    
    while True:
        agent_graph, agent_name = select_agent()
        
        if agent_graph is None:
            print("👋 Thanks for trying the MCP integration demo!")
            break
            
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Cannot start agent without OpenAI API key.")
            continue
            
        result = chat_with_agent(agent_graph, agent_name)
        
        if result == 'quit':
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)
