#!/usr/bin/env python3
"""
Docker Setup Validation Script

This script validates the Docker configuration files and setup without
requiring Docker to be running. It checks for common issues and provides
recommendations for optimal deployment.
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def log_info(msg: str) -> None:
    """Log info message with blue color."""
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.NC}")

def log_success(msg: str) -> None:
    """Log success message with green color."""
    print(f"{Colors.GREEN}✅ {msg}{Colors.NC}")

def log_warning(msg: str) -> None:
    """Log warning message with yellow color."""
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.NC}")

def log_error(msg: str) -> None:
    """Log error message with red color."""
    print(f"{Colors.RED}❌ {msg}{Colors.NC}")

def log_header(msg: str) -> None:
    """Log header message with cyan color."""
    print(f"\n{Colors.CYAN}{'='*50}{Colors.NC}")
    print(f"{Colors.CYAN}{msg}{Colors.NC}")
    print(f"{Colors.CYAN}{'='*50}{Colors.NC}")

class DockerValidator:
    """Docker configuration validator."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.project_root = Path.cwd()
        
    def check_file_exists(self, filepath: str, required: bool = True) -> bool:
        """Check if a file exists."""
        file_path = self.project_root / filepath
        if file_path.exists():
            log_success(f"Found {filepath}")
            return True
        else:
            if required:
                log_error(f"Missing required file: {filepath}")
                self.errors.append(f"Missing required file: {filepath}")
            else:
                log_warning(f"Optional file not found: {filepath}")
                self.warnings.append(f"Optional file not found: {filepath}")
            return False
    
    def validate_dockerfile(self) -> bool:
        """Validate Dockerfile syntax and best practices."""
        log_info("Validating Dockerfile...")
        
        if not self.check_file_exists("Dockerfile"):
            return False
            
        with open("Dockerfile", 'r') as f:
            content = f.read()
            
        # Check for multi-stage build
        if "FROM python:3.12-slim as builder" in content:
            log_success("Uses multi-stage build for optimization")
        else:
            log_warning("Consider using multi-stage build for smaller images")
            
        # Check for non-root user
        if "USER appuser" in content:
            log_success("Runs as non-root user for security")
        else:
            log_error("Should run as non-root user")
            self.errors.append("Dockerfile should include non-root user")
            
        # Check for health check
        if "HEALTHCHECK" in content:
            log_success("Includes health check configuration")
        else:
            log_warning("Consider adding health check")
            
        # Check for exposed port
        if "EXPOSE 10000" in content:
            log_success("Exposes correct port (10000)")
        else:
            log_error("Should expose port 10000")
            self.errors.append("Dockerfile should expose port 10000")
            
        return True
    
    def validate_docker_compose(self) -> bool:
        """Validate docker-compose.yml configuration."""
        log_info("Validating docker-compose.yml...")
        
        if not self.check_file_exists("docker-compose.yml"):
            return False
            
        with open("docker-compose.yml", 'r') as f:
            content = f.read()
            
        # Check for required services
        if "a2a-agent:" in content:
            log_success("Defines a2a-agent service")
        else:
            log_error("Missing a2a-agent service definition")
            self.errors.append("docker-compose.yml missing a2a-agent service")
            
        # Check for environment variables
        if "OPENAI_API_KEY" in content:
            log_success("Configures OpenAI API key environment variable")
        else:
            log_error("Missing OpenAI API key configuration")
            self.errors.append("docker-compose.yml missing OPENAI_API_KEY")
            
        # Check for volume mounts
        if "./data:/app/data" in content:
            log_success("Mounts data directory for RAG")
        else:
            log_warning("Consider mounting data directory for RAG documents")
            
        # Check for restart policy
        if "restart: unless-stopped" in content:
            log_success("Includes restart policy")
        else:
            log_warning("Consider adding restart policy")
            
        # Check for health check
        if "healthcheck:" in content:
            log_success("Includes health check configuration")
        else:
            log_warning("Consider adding health check")
            
        return True
    
    def validate_dockerignore(self) -> bool:
        """Validate .dockerignore configuration."""
        log_info("Validating .dockerignore...")
        
        if not self.check_file_exists(".dockerignore"):
            return False
            
        with open(".dockerignore", 'r') as f:
            content = f.read()
            
        ignore_patterns = [
            "__pycache__/",
            ".git/",
            ".venv/",
            "README.md",
            "second_agent/"
        ]
        
        for pattern in ignore_patterns:
            if pattern in content:
                log_success(f"Ignores {pattern}")
            else:
                log_warning(f"Consider ignoring {pattern}")
                
        return True
    
    def validate_environment_config(self) -> bool:
        """Validate environment configuration."""
        log_info("Validating environment configuration...")
        
        # Check for template file
        if self.check_file_exists("env.template", required=False):
            with open("env.template", 'r') as f:
                content = f.read()
                
            if "OPENAI_API_KEY=" in content:
                log_success("Template includes OpenAI API key")
            else:
                log_error("Template missing OpenAI API key")
                self.errors.append("env.template missing OPENAI_API_KEY")
        
        # Check for .env file
        if self.check_file_exists(".env", required=False):
            log_warning(".env file found - ensure it contains valid API keys")
        else:
            log_info("No .env file found - user will need to create from template")
            
        return True
    
    def validate_deployment_scripts(self) -> bool:
        """Validate deployment scripts."""
        log_info("Validating deployment scripts...")
        
        # Check for deploy script
        if self.check_file_exists("deploy.sh", required=False):
            # Check if executable
            if os.access("deploy.sh", os.X_OK):
                log_success("deploy.sh is executable")
            else:
                log_warning("deploy.sh is not executable - run: chmod +x deploy.sh")
        
        return True
    
    def validate_app_structure(self) -> bool:
        """Validate application structure for containerization."""
        log_info("Validating application structure...")
        
        required_files = [
            "app/__init__.py",
            "app/__main__.py",
            "app/agent.py",
            "app/agent_executor.py",
            "pyproject.toml"
        ]
        
        for file_path in required_files:
            self.check_file_exists(file_path)
            
        # Check for data directory
        if self.check_file_exists("data", required=False):
            data_path = Path("data")
            if data_path.is_dir():
                pdf_files = list(data_path.glob("*.pdf"))
                if pdf_files:
                    log_success(f"Found {len(pdf_files)} PDF files for RAG")
                else:
                    log_info("No PDF files found in data directory")
            
        return True
    
    def check_docker_availability(self) -> bool:
        """Check if Docker is available (optional)."""
        log_info("Checking Docker availability...")
        
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                log_success(f"Docker found: {version}")
                
                # Check if daemon is running
                daemon_result = subprocess.run(
                    ["docker", "info"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if daemon_result.returncode == 0:
                    log_success("Docker daemon is running")
                    return True
                else:
                    log_warning("Docker installed but daemon not running")
                    return False
            else:
                log_warning("Docker command failed")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            log_warning("Docker not found or not accessible")
            return False
    
    def check_docker_compose_availability(self) -> bool:
        """Check if Docker Compose is available (optional)."""
        log_info("Checking Docker Compose availability...")
        
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                log_success(f"Docker Compose found: {version}")
                return True
            else:
                # Try docker compose (newer syntax)
                result = subprocess.run(
                    ["docker", "compose", "version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    version = result.stdout.strip()
                    log_success(f"Docker Compose found: {version}")
                    return True
                else:
                    log_warning("Docker Compose not found")
                    return False
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            log_warning("Docker Compose not found or not accessible")
            return False
    
    def generate_recommendations(self) -> None:
        """Generate recommendations based on validation results."""
        log_header("RECOMMENDATIONS")
        
        if not self.errors and not self.warnings:
            log_success("All Docker configurations are optimal! 🎉")
            print("\n🚀 You're ready to deploy with Docker!")
            print("\nNext steps:")
            print("1. Ensure your .env file has valid API keys")
            print("2. Run: docker-compose up --build")
            print("3. Test: curl http://localhost:10000/.well-known/agent-card.json")
            return
            
        if self.errors:
            print(f"\n{Colors.RED}🔥 Critical Issues Found:{Colors.NC}")
            for error in self.errors:
                print(f"   • {error}")
            print(f"\n{Colors.RED}Fix these issues before deploying!{Colors.NC}")
            
        if self.warnings:
            print(f"\n{Colors.YELLOW}⚡ Optimization Opportunities:{Colors.NC}")
            for warning in self.warnings:
                print(f"   • {warning}")
            print(f"\n{Colors.YELLOW}These are optional but recommended for production.{Colors.NC}")
            
        # Deployment readiness
        if not self.errors:
            print(f"\n{Colors.GREEN}🎯 Deployment Status: READY{Colors.NC}")
            print("You can proceed with Docker deployment!")
        else:
            print(f"\n{Colors.RED}🚫 Deployment Status: NOT READY{Colors.NC}")
            print("Fix the critical issues above first.")
    
    def run_validation(self) -> bool:
        """Run complete validation suite."""
        log_header("DOCKER SETUP VALIDATION")
        
        print("🐳 Validating Docker configuration for LangGraph A2A Agent")
        print("📁 Project:", self.project_root)
        
        # Core validations
        validations = [
            self.validate_dockerfile,
            self.validate_docker_compose,
            self.validate_dockerignore,
            self.validate_environment_config,
            self.validate_deployment_scripts,
            self.validate_app_structure,
        ]
        
        for validation in validations:
            try:
                validation()
            except Exception as e:
                log_error(f"Validation failed: {e}")
                self.errors.append(f"Validation error: {e}")
        
        # Optional checks
        self.check_docker_availability()
        self.check_docker_compose_availability()
        
        # Generate recommendations
        self.generate_recommendations()
        
        return len(self.errors) == 0

def main():
    """Main validation function."""
    try:
        validator = DockerValidator()
        success = validator.run_validation()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user.{Colors.NC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.NC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
