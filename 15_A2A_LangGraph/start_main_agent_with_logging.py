#!/usr/bin/env python3
"""
Start the main A2A agent server with enhanced logging for demo purposes
Usage: python start_main_agent_with_logging.py
"""

import logging
import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import enhanced logging first
from app.enhanced_logging import setup_enhanced_logging

# Set up enhanced logging before importing main
setup_enhanced_logging()

# Now import and run the main app
from app.__main__ import main

if __name__ == '__main__':
    print("🏭 MAIN AGENT SERVER - ENHANCED LOGGING MODE")
    print("=" * 60)
    print("🎯 This is the main A2A agent that the second agent will call")
    print("📊 All processing steps will be logged in detail")
    print("🔄 Waiting for requests from the second agent...")
    print("=" * 60)
    
    # Start the main agent
    main()
