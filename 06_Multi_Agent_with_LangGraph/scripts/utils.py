#!/usr/bin/env python3
"""
Utility script for LinkedIn ML Paper Post Generation API

This script provides common utility functions and operations for the API.
"""

import os
import json
import time
import uuid
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def generate_sample_requests() -> List[Dict[str, Any]]:
    """Generate sample request payloads for testing."""
    return [
        {
            "paper_title": "Attention Is All You Need",
            "additional_context": "Focus on transformer architecture and its impact on NLP",
            "target_audience": "professional",
            "include_technical_details": True,
            "max_hashtags": 10,
            "tone": "professional"
        },
        {
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "additional_context": "Emphasize practical applications in industry",
            "target_audience": "professional", 
            "include_technical_details": False,
            "max_hashtags": 8,
            "tone": "casual"
        },
        {
            "paper_title": "ResNet: Deep Residual Learning for Image Recognition",
            "additional_context": "Academic focus on methodology and results",
            "target_audience": "academic",
            "include_technical_details": True,
            "max_hashtags": 12,
            "tone": "academic"
        },
        {
            "paper_title": "GPT-3: Language Models are Few-Shot Learners",
            "additional_context": "General audience introduction to large language models",
            "target_audience": "general",
            "include_technical_details": False,
            "max_hashtags": 6,
            "tone": "casual"
        }
    ]


def save_sample_requests(output_dir: str = "examples"):
    """Save sample requests to JSON files."""
    samples = generate_sample_requests()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save individual samples
    for i, sample in enumerate(samples, 1):
        filename = output_path / f"sample_request_{i}.json"
        with open(filename, 'w') as f:
            json.dump(sample, f, indent=2)
        print(f"✅ Saved: {filename}")
    
    # Save batch request
    batch_request = {
        "papers": samples[:2],  # First two samples
        "schedule_posts": False,
        "time_interval_minutes": 60
    }
    
    batch_filename = output_path / "batch_request.json"
    with open(batch_filename, 'w') as f:
        json.dump(batch_request, f, indent=2)
    print(f"✅ Saved: {batch_filename}")


def generate_curl_examples(base_url: str = "http://localhost:8000"):
    """Generate curl command examples for the API."""
    examples = []
    
    # Health check
    examples.append({
        "name": "Health Check",
        "command": f'curl -X GET "{base_url}/health"'
    })
    
    # Generate post
    sample_request = generate_sample_requests()[0]
    examples.append({
        "name": "Generate LinkedIn Post",
        "command": f'curl -X POST "{base_url}/generate-post" \\\n  -H "Content-Type: application/json" \\\n  -d \'{json.dumps(sample_request)}\''
    })
    
    # Check status (with placeholder task ID)
    examples.append({
        "name": "Check Task Status",
        "command": f'curl -X GET "{base_url}/status/TASK_ID_HERE"'
    })
    
    # Verify post
    verify_request = {
        "post_content": "🚀 Sample LinkedIn post about machine learning with proper formatting and hashtags. What do you think? #AI #MachineLearning",
        "paper_reference": "Sample paper reference",
        "verification_type": "both"
    }
    examples.append({
        "name": "Verify Post",
        "command": f'curl -X POST "{base_url}/verify-post" \\\n  -H "Content-Type: application/json" \\\n  -d \'{json.dumps(verify_request)}\''
    })
    
    return examples


def save_curl_examples(output_dir: str = "examples", base_url: str = "http://localhost:8000"):
    """Save curl examples to a file."""
    examples = generate_curl_examples(base_url)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = output_path / "api_examples.sh"
    
    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# LinkedIn ML Paper Post Generation API - cURL Examples\n\n")
        f.write(f"# API Base URL: {base_url}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for example in examples:
            f.write(f"# {example['name']}\n")
            f.write(f"{example['command']}\n\n")
    
    # Make the file executable
    os.chmod(filename, 0o755)
    print(f"✅ Saved curl examples: {filename}")


def check_api_health(base_url: str = "http://localhost:8000"):
    """Check if the API is running and healthy."""
    try:
        import requests
        
        response = requests.get(f"{base_url}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            
            services = data.get('services', {})
            print("   Services:")
            for service, status in services.items():
                status_icon = "✅" if status == "connected" else "⚠️"
                print(f"     {status_icon} {service}: {status}")
            
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return False


def generate_task_id():
    """Generate a UUID for task tracking."""
    task_id = str(uuid.uuid4())
    print(f"Generated task ID: {task_id}")
    return task_id


def create_deployment_checklist():
    """Create a deployment checklist."""
    checklist = """# LinkedIn ML Paper Post Generation API - Deployment Checklist

## Pre-deployment Checks

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed from requirements.txt
- [ ] .env file created with proper API keys

### API Keys Configuration
- [ ] OPENAI_API_KEY set and valid
- [ ] TAVILY_API_KEY set and valid
- [ ] API keys tested with simple requests

### Application Configuration
- [ ] DEBUG=false for production
- [ ] Proper HOST and PORT configured
- [ ] LOG_LEVEL set appropriately
- [ ] REDIS_URL configured (if using Redis)

### Security
- [ ] CORS settings configured for production
- [ ] Rate limiting configured
- [ ] Input validation working
- [ ] Error handling doesn't expose sensitive info

### Testing
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] API endpoint tests passing
- [ ] Load testing completed (if needed)

## Deployment Steps

### Docker Deployment
- [ ] Dockerfile builds successfully
- [ ] docker-compose.yml configured
- [ ] Environment variables passed to containers
- [ ] Volumes configured for persistent data
- [ ] Network connectivity tested

### Manual Deployment
- [ ] Application starts without errors
- [ ] Health check endpoint returns 200
- [ ] All API endpoints responding
- [ ] Background tasks working
- [ ] Logging properly configured

### Production Setup
- [ ] Reverse proxy configured (nginx/apache)
- [ ] SSL/TLS certificates installed
- [ ] Monitoring setup (if needed)
- [ ] Backup strategy implemented
- [ ] Log rotation configured

## Post-deployment Verification

### Functional Testing
- [ ] Health check: GET /health
- [ ] Post generation: POST /generate-post
- [ ] Status checking: GET /status/{{task_id}}
- [ ] Post verification: POST /verify-post
- [ ] Batch generation: POST /batch-generate

### Performance Testing
- [ ] Response times acceptable
- [ ] Memory usage stable
- [ ] No memory leaks detected
- [ ] Error rates acceptable

### Monitoring
- [ ] Application logs accessible
- [ ] Error alerting configured
- [ ] Performance metrics collected
- [ ] Health monitoring active

## Rollback Plan
- [ ] Previous version tagged
- [ ] Rollback procedure documented
- [ ] Database backup created (if applicable)
- [ ] Rollback tested in staging
"""
    
    checklist_file = Path("DEPLOYMENT_CHECKLIST.md")
    with open(checklist_file, 'w') as f:
        f.write(checklist)
    
    print(f"✅ Created deployment checklist: {checklist_file}")


def main():
    """Main utility function."""
    parser = argparse.ArgumentParser(description="LinkedIn ML Paper Post Generation API - Utilities")
    parser.add_argument("command", choices=[
        "samples", "curl", "health", "taskid", "checklist"
    ], help="Utility command to run")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="examples", help="Output directory")
    
    args = parser.parse_args()
    
    print("🛠️  LinkedIn ML Paper Post Generation API - Utilities")
    print("=" * 50)
    
    if args.command == "samples":
        print("📝 Generating sample request files...")
        save_sample_requests(args.output)
    
    elif args.command == "curl":
        print("📋 Generating cURL examples...")
        save_curl_examples(args.output, args.url)
    
    elif args.command == "health":
        print(f"🔍 Checking API health at {args.url}...")
        check_api_health(args.url)
    
    elif args.command == "taskid":
        print("🆔 Generating new task ID...")
        generate_task_id()
    
    elif args.command == "checklist":
        print("📋 Creating deployment checklist...")
        create_deployment_checklist()
    
    print("\n✅ Utility completed")


if __name__ == "__main__":
    main() 