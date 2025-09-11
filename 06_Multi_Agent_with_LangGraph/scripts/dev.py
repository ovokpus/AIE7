#!/usr/bin/env python3
"""
Development script for LinkedIn ML Paper Post Generation API

This script provides development utilities including hot reload, logging, and debug mode.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def setup_development_environment():
    """Set up the development environment."""
    print("🔧 Setting up development environment...")
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create virtual environment if it doesn't exist
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Get the python executable from venv
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    # Install/upgrade requirements
    print("📚 Installing requirements...")
    subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
    
    # Install development dependencies
    dev_packages = [
        "pytest",
        "pytest-asyncio",
        "black",
        "flake8",
        "mypy",
        "pre-commit"
    ]
    
    print("🛠️  Installing development packages...")
    subprocess.run([str(pip_exe), "install"] + dev_packages, check=True)
    
    return python_exe


def check_environment_variables():
    """Check if required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = []
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        
        # Load .env file
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key in required_vars and value and value != "your_api_key_here":
                        required_vars.remove(key)
    
    missing_vars = required_vars
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("\nPlease create a .env file with:")
        for var in missing_vars:
            print(f"  {var}=your_api_key_here")
        return False
    else:
        print("✅ All required environment variables are set")
        return True


def run_development_server(python_exe, host="0.0.0.0", port=8000, reload=True):
    """Run the development server with hot reload."""
    print(f"🚀 Starting development server on {host}:{port}")
    print("📝 Features enabled:")
    print("  - Hot reload (files will auto-restart server)")
    print("  - Debug mode")
    print("  - Detailed error messages")
    print("  - API documentation at /docs")
    print("\n📖 API Documentation:")
    print(f"  - Swagger UI: http://{host}:{port}/docs")
    print(f"  - ReDoc: http://{host}:{port}/redoc")
    print("\n🔗 Useful endpoints:")
    print(f"  - Health check: http://{host}:{port}/health")
    print(f"  - Generate post: POST http://{host}:{port}/generate-post")
    print(f"  - Verify post: POST http://{host}:{port}/verify-post")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Set environment variables for development
    env = os.environ.copy()
    env["DEBUG"] = "true"
    env["LOG_LEVEL"] = "DEBUG"
    
    # Run the server
    cmd = [
        str(python_exe), "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "debug"
    ]
    
    if reload:
        cmd.extend(["--reload", "--reload-dir", "app"])
    
    try:
        subprocess.run(cmd, env=env, cwd=".")
    except KeyboardInterrupt:
        print("\n👋 Development server stopped")


def run_tests(python_exe):
    """Run the test suite."""
    print("🧪 Running test suite...")
    
    # Run the API tests
    test_script = Path("scripts/test_api.py")
    if test_script.exists():
        subprocess.run([str(python_exe), str(test_script)], check=True)
    else:
        print("❌ Test script not found")


def format_code(python_exe):
    """Format code using black."""
    print("🎨 Formatting code with black...")
    
    try:
        subprocess.run([str(python_exe), "-m", "black", "app/", "scripts/", "examples/"], check=True)
        print("✅ Code formatted successfully")
    except subprocess.CalledProcessError:
        print("❌ Code formatting failed")


def lint_code(python_exe):
    """Lint code using flake8."""
    print("🔍 Linting code with flake8...")
    
    try:
        subprocess.run([str(python_exe), "-m", "flake8", "app/", "--max-line-length=88", "--extend-ignore=E203,W503"], check=True)
        print("✅ Code linting passed")
    except subprocess.CalledProcessError:
        print("❌ Code linting failed")


def type_check(python_exe):
    """Type check code using mypy."""
    print("🔍 Type checking with mypy...")
    
    try:
        subprocess.run([str(python_exe), "-m", "mypy", "app/", "--ignore-missing-imports"], check=True)
        print("✅ Type checking passed")
    except subprocess.CalledProcessError:
        print("❌ Type checking failed")


def create_sample_env():
    """Create a sample .env file."""
    env_content = """# LinkedIn ML Paper Post Generation API - Environment Variables

# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Application Configuration
DEBUG=true
HOST=0.0.0.0
PORT=8000

# OpenAI Settings
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7

# Redis Configuration (optional for development)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=app.log

# LangGraph Configuration
RECURSION_LIMIT=50
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created sample .env file")
        print("⚠️  Please update the API keys with your actual keys")
    else:
        print("⚠️  .env file already exists")


def main():
    """Main development script."""
    parser = argparse.ArgumentParser(description="LinkedIn ML Paper Post Generation API - Development Script")
    parser.add_argument("command", choices=[
        "setup", "run", "test", "format", "lint", "typecheck", "env", "all"
    ], help="Command to run")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable hot reload")
    
    args = parser.parse_args()
    
    print("🚀 LinkedIn ML Paper Post Generation API - Development Script")
    print("=" * 60)
    
    if args.command == "env":
        create_sample_env()
        return
    
    if args.command in ["setup", "all"]:
        python_exe = setup_development_environment()
        create_sample_env()
    else:
        # Find python executable
        venv_path = Path("venv")
        if venv_path.exists():
            if os.name == 'nt':  # Windows
                python_exe = venv_path / "Scripts" / "python.exe"
            else:  # Unix/Linux/macOS
                python_exe = venv_path / "bin" / "python"
        else:
            python_exe = sys.executable
    
    if args.command in ["run", "all"]:
        if not check_environment_variables():
            print("\n⚠️  Environment check failed. Run 'python scripts/dev.py env' to create a sample .env file")
            sys.exit(1)
        
        if args.command == "all":
            print("\n⏳ Running quick checks before starting server...")
            try:
                format_code(python_exe)
                lint_code(python_exe)
            except:
                print("⚠️  Some checks failed, but continuing with server start...")
        
        run_development_server(
            python_exe, 
            host=args.host, 
            port=args.port, 
            reload=not args.no_reload
        )
    
    elif args.command == "test":
        run_tests(python_exe)
    
    elif args.command == "format":
        format_code(python_exe)
    
    elif args.command == "lint":
        lint_code(python_exe)
    
    elif args.command == "typecheck":
        type_check(python_exe)
    
    print("\n✅ Development script completed")


if __name__ == "__main__":
    main() 