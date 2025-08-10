#!/usr/bin/env python3
"""
Basic MCP Tools Demo

This example shows how to use the MCP tools directly without LangGraph integration.
It demonstrates the core functionality of each tool.
"""

import sys
import json
from pathlib import Path

# Add the parent directory to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

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


def demo_file_operations():
    """Demonstrate file operation tools."""
    print("🗂️  File Operations Demo")
    print("=" * 40)
    
    # Create a test file
    test_file = "examples/test_demo.txt"
    content = "Hello from MCP tools demo!\nThis is a test file created by the MCP server."
    
    # Write file
    print("1. Writing test file...")
    result = write_file_content(test_file, content)
    print(f"   Result: {result}")
    
    # Read file
    print("\n2. Reading test file...")
    result = read_file_content(test_file)
    print(f"   Content: {result}")
    
    # List directory
    print("\n3. Listing examples directory...")
    result = list_directory_contents("examples")
    print(f"   Directory contents: {result}")
    
    print()


def demo_data_analysis():
    """Demonstrate data analysis tools."""
    print("📊 Data Analysis Demo")
    print("=" * 40)
    
    # Create sample CSV data
    csv_content = """name,age,score
Alice,25,85.5
Bob,30,92.0
Charlie,22,78.5
Diana,28,96.0
Eve,26,89.5"""
    
    csv_file = "examples/sample_data.csv"
    
    # Write CSV file
    print("1. Creating sample CSV file...")
    write_result = write_file_content(csv_file, csv_content)
    print(f"   Result: {write_result}")
    
    # Analyze CSV
    print("\n2. Analyzing CSV data...")
    for operation in ["summary", "columns", "sample", "stats"]:
        print(f"\n   Operation: {operation}")
        result = analyze_csv_data(csv_file, operation)
        print(f"   Result: {result}")
    
    # Calculate statistics
    print("\n3. Calculate statistics for scores...")
    scores = [85.5, 92.0, 78.5, 96.0, 89.5]
    result = calculate_statistics(scores, "all")
    print(f"   Statistics: {result}")
    
    print()


def demo_system_utilities():
    """Demonstrate system utility tools."""
    print("⚙️  System Utilities Demo")
    print("=" * 40)
    
    # Get current time
    print("1. Getting current time...")
    result = get_current_time()
    print(f"   Time: {result}")
    
    # Get environment info
    print("\n2. Getting environment info...")
    result = get_environment_info()
    env_data = json.loads(result)
    print(f"   Python version: {env_data['python_version']}")
    print(f"   Platform: {env_data['platform']}")
    print(f"   Current directory: {env_data['current_directory']}")
    
    # Validate URL
    print("\n3. Validating URLs...")
    test_urls = [
        "https://www.google.com",
        "https://httpbin.org/json",
        "invalid-url",
        "http://localhost:9999"  # This will likely fail
    ]
    
    for url in test_urls:
        print(f"\n   Testing URL: {url}")
        result = validate_url(url)
        print(f"   Result: {result}")
    
    print()


def demo_mathematical_operations():
    """Demonstrate mathematical tools."""
    print("🔢 Mathematical Operations Demo")
    print("=" * 40)
    
    # Test different statistical operations
    datasets = [
        ([1, 2, 3, 4, 5], "Simple sequence"),
        ([10.5, 15.2, 8.7, 22.1, 18.9, 12.3], "Decimal numbers"),
        ([100], "Single number"),
        ([50, 50, 50, 50], "Identical numbers"),
        (list(range(1, 101)), "1 to 100")
    ]
    
    for data, description in datasets:
        print(f"\n{description}: {data[:5]}{'...' if len(data) > 5 else ''}")
        
        # Calculate different statistics
        for stat_type in ["mean", "median", "all"]:
            if stat_type == "std" and len(data) < 2:
                continue  # Skip std for single values
            
            result = calculate_statistics(data, stat_type)
            print(f"   {stat_type}: {result}")
    
    print()


def main():
    """Run all demo functions."""
    print("🚀 MCP Tools Demo Suite")
    print("=" * 50)
    print("This demo shows the capabilities of each MCP tool.\n")
    
    try:
        demo_file_operations()
        demo_data_analysis()
        demo_system_utilities()
        demo_mathematical_operations()
        
        print("✅ All demos completed successfully!")
        print("\nCleanup: You can delete the test files in examples/ if desired:")
        print("   - examples/test_demo.txt")
        print("   - examples/sample_data.csv")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
