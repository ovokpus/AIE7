# second_agent/main.py - Modular Expert Agent System Entry Point
"""
Expert Agent System for A2A Communication
A modular LangGraph-based client that acts as goal-oriented expert agents
"""
import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modular components
from modules.logging_config import setup_expert_logging
from modules.a2a_client import A2AClientWrapper, check_server_status
from modules.demo_modes import demo_single_query, demo_different_query_types, interactive_mode

# Initialize logging
logger, interaction_logger = setup_expert_logging()


async def main():
    """Main demo function with expert agent system"""
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
