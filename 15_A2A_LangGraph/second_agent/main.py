"""
Expert Agent System - Modular A2A Client Entry Point.

This module provides the main entry point for the Expert Agent System, a sophisticated
LangGraph-based client that communicates with A2A agents using goal-oriented expert
personas. The system demonstrates advanced agent-to-agent communication patterns
with quality evaluation and follow-up questioning.

Key Features:
    - Goal-oriented expert personas with specific research missions
    - Quality evaluation and automatic follow-up questioning
    - Modular architecture with clean separation of concerns
    - Interactive and demo modes for testing and demonstration
    - Comprehensive logging for visibility into agent interactions

The system includes 4 specialized expert profiles:
    - Dr. Sarah Chen: ML Expert studying Kimi K2 (Assignment Example)
    - Prof. Marcus Rodriguez: Transformer Architecture Researcher
    - Alex Kim: AI Startup Founder  
    - Dr. Emma Watson: AI Security Expert

Usage:
    python second_agent/main.py

Dependencies:
    - Main A2A agent server running on localhost:10000
    - OpenAI API key for LLM functionality
    - All module dependencies in modules/ directory
"""

import asyncio
import sys
from typing import NoReturn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modular components
from modules.logging_config import setup_expert_logging
from modules.a2a_client import A2AClientWrapper, check_server_status
from modules.demo_modes import demo_single_query, demo_different_query_types, interactive_mode

# Initialize logging
logger, interaction_logger = setup_expert_logging()


async def main() -> None:
    """Run the Expert Agent System with interactive demo options.
    
    This function serves as the main entry point for the Expert Agent System.
    It provides a menu-driven interface for users to choose between different
    demonstration modes, including single queries, multi-persona demos, and
    the full interactive expert agent experience.
    
    The function handles:
        - Server connectivity verification
        - A2A client initialization
        - User interface for demo selection
        - Proper cleanup and error handling
        
    Demo Options:
        1. Single Query Demo: Quick test with minimal interaction
        2. Multi-Persona Demo: Original query classification system  
        3. Expert Agent Mode: Full goal-oriented expert behavior (Assignment compliant)
        
    Raises:
        SystemExit: If server is unavailable or fatal errors occur
        
    Returns:
        None: Runs until user exits or encounters an error
    """
    print("🎓 EXPERT AGENT SYSTEM - Advanced A2A Communication")
    print("=" * 60)
    print("🎯 Goal-oriented experts with specific research missions!")
    print("📊 Featuring: Dr. Sarah Chen studying Kimi K2 (Assignment Example)")
    
    # Check server status first
    if not await check_server_status():
        return
    
    # Initialize A2A client
    a2a_client = A2AClientWrapper()
    
    print("\nChoose demo type:")
    print("1. Single Query Demo (quick test)")
    print("2. Multi-Persona Demo (original system)")
    print("3. 🌟 Expert Agent Mode (NEW - Assignment Compliant)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    try:
        if choice == "1":
            await demo_single_query(a2a_client)
        elif choice == "2":
            await demo_different_query_types(a2a_client)
        elif choice == "3":
            await interactive_mode(a2a_client)
        else:
            print("Invalid choice. Running interactive mode...")
            await interactive_mode(a2a_client)
    finally:
        # Clean up
        if hasattr(a2a_client, 'httpx_client') and a2a_client.httpx_client:
            await a2a_client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
