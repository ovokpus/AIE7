#!/usr/bin/env python3
"""
Test Runner for MCP Integration

Runs all tests related to the MCP integration to ensure everything works correctly.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))


def run_test(test_file: str, description: str) -> bool:
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {description}")
    print('='*60)
    
    try:
        # Change to the parent directory to run tests from project root
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            ["uv", "run", "python", f"test/{test_file}"],
            cwd=project_root,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 MCP Integration Test Suite")
    print("=" * 60)
    print("Running all tests to verify MCP integration functionality.")
    
    tests = [
        ("test_mcp_integration.py", "MCP Integration Tests"),
        # Note: test_served_graph.py requires a running LangGraph server
        # ("test_served_graph.py", "LangGraph Server Tests"), 
    ]
    
    passed = 0
    total = len(tests)
    
    for test_file, description in tests:
        if run_test(test_file, description):
            passed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TEST SUMMARY")
    print('='*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! MCP integration is working correctly.")
        print("\n✅ Your LangGraph agents now have 8 additional MCP tools:")
        print("   📁 File operations, 📊 Data analysis, 🌐 Web utils, ⚙️ System info")
        return 0
    else:
        print(f"⚠️ {total - passed} test(s) failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
