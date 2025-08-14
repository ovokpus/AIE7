# 🐳 Docker Deployment Guide for LangGraph A2A Agent

This guide provides comprehensive instructions for deploying the LangGraph A2A Agent using Docker containers.

## 📑 Table of Contents

- [🎯 Quick Start](#quick-start)
- [📋 Prerequisites](#prerequisites)
- [🏗️ Build and Run](#build-and-run)
- [⚙️ Configuration](#configuration)
- [🔧 Production Deployment](#production-deployment)
- [📊 Monitoring and Logs](#monitoring-and-logs)
- [🛠️ Troubleshooting](#troubleshooting)
- [🔒 Security Considerations](#security-considerations)

## 🎯 Quick Start

### Development Setup (5 minutes)

```bash
# 1. Clone and navigate to project
cd /path/to/15_A2A_LangGraph

# 2. Create environment file from template
cp env.template .env

# 3. Edit .env with your API keys (REQUIRED)
# Add your OPENAI_API_KEY=your_key_here
# Optionally add TAVILY_API_KEY for web search

# 4. Build and run with Docker Compose
docker-compose up --build

# 5. Test the agent (in another terminal)
curl http://localhost:10000/.well-known/agent-card.json
```

### One-Command Deployment

```bash
# Automated deployment with validation
./deploy.sh
```

### Production Setup

```bash
# Use production profile with Nginx proxy
docker-compose --profile production up --build -d

# Or use the deployment script
./deploy.sh --production --detached
```

### Alternative: Direct Docker Build

```bash
# Build the image
docker build -t langgraph-a2a-agent .

# Run with environment variables
docker run -d \
  --name a2a-agent \
  -p 10000:10000 \
  -e OPENAI_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data:ro \
  langgraph-a2a-agent
```

### 🔍 Setup Validation

Validate your Docker configuration before deployment:

```bash
# Validate Docker setup (no Docker daemon required)
python validate_docker_setup.py

# Or run with full validation
./validate_docker_setup.py
```

This validation script checks:
- ✅ Dockerfile configuration and best practices
- ✅ docker-compose.yml service definitions
- ✅ Environment variable configuration
- ✅ Application structure and dependencies
- ✅ Docker and Docker Compose availability
- ✅ Security and optimization recommendations

### 🚀 Quick Health Check

After deployment, verify everything is working:

```bash
# Check agent card endpoint
curl -s http://localhost:10000/.well-known/agent-card.json | jq '.'

# Test a simple query (requires functioning agent)
curl -X POST http://localhost:10000/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, test message"}' | jq '.'
```

## 📋 Prerequisites

### Required Software
- **Docker** (20.10+)
- **Docker Compose** (2.0+)
- **curl** (for testing)

### Required API Keys
- **OpenAI API Key** (required)
- **Tavily API Key** (optional, for web search)

### System Requirements
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 5GB free space for images and data
- **Network**: Internet access for API calls

## 🏗️ Build and Run

### Option 1: Docker Compose (Recommended)

```bash
# Development mode
docker-compose up --build

# Production mode with proxy
docker-compose --profile production up --build -d

# Run in background
docker-compose up -d
```

### Option 2: Direct Docker Build

```bash
# Build the image
docker build -t langgraph-a2a-agent .

# Run the container
docker run -d \
  --name a2a-agent \
  -p 10000:10000 \
  -e OPENAI_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data:ro \
  langgraph-a2a-agent
```

### Option 3: Multi-stage Build for Production

```bash
# Build optimized production image
docker build --target production -t langgraph-a2a-agent:prod .

# Run with production settings
docker run -d \
  --name a2a-agent-prod \
  -p 10000:10000 \
  --restart unless-stopped \
  -e OPENAI_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/logs:/app/logs \
  langgraph-a2a-agent:prod
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp env.template .env
```

Configure the following variables:

#### Required Variables
```env
OPENAI_API_KEY=your_openai_api_key_here
```

#### Optional Variables
```env
# Model Configuration
TOOL_LLM_NAME=gpt-4o-mini
TOOL_LLM_URL=https://api.openai.com/v1

# Web Search (Tavily)
TAVILY_API_KEY=your_tavily_api_key_here

# RAG Configuration
RAG_DATA_DIR=data

# Development
DEBUG=false
LOG_LEVEL=INFO
```

### Volume Mounts

```yaml
volumes:
  # RAG documents (read-only)
  - ./data:/app/data:ro
  
  # Persistent logs
  - ./logs:/app/logs
  
  # Custom configuration (optional)
  - ./config:/app/config:ro
```

### Network Configuration

```yaml
networks:
  a2a-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 🔧 Production Deployment

### With Nginx Reverse Proxy

```bash
# Start with production profile
docker-compose --profile production up -d

# Or customize nginx.conf and restart
docker-compose restart nginx
```

### Health Checks

The container includes health checks:

```bash
# Check container health
docker-compose ps

# Manual health check
curl -f http://localhost:10000/.well-known/agent-card.json
```

### SSL/HTTPS Configuration

1. **Obtain SSL certificates**:
```bash
# Using Let's Encrypt (example)
certbot certonly --standalone -d your-domain.com
```

2. **Update nginx.conf**:
   - Uncomment SSL server block
   - Update certificate paths
   - Configure your domain

3. **Mount certificates**:
```yaml
volumes:
  - /etc/letsencrypt/live/your-domain.com:/etc/nginx/ssl:ro
```

### Scaling and Load Balancing

```yaml
# docker-compose.yml
services:
  a2a-agent:
    # ... other config
    deploy:
      replicas: 3
    
  nginx:
    # ... configure upstream load balancing
```

## 📊 Monitoring and Logs

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f a2a-agent

# Last 100 lines
docker-compose logs --tail=100 a2a-agent
```

### Log Persistence

Logs are automatically mounted to `./logs/` directory for persistence.

### Monitoring Endpoints

```bash
# Agent card (health/capabilities)
curl http://localhost:10000/.well-known/agent-card.json

# Nginx health check
curl http://localhost/health
```

### Performance Monitoring

```bash
# Container stats
docker stats langgraph-a2a-agent

# Resource usage
docker system df
docker system prune  # cleanup
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. **Container Won't Start**
```bash
# Check logs
docker-compose logs a2a-agent

# Common causes:
# - Missing OPENAI_API_KEY
# - Port 10000 already in use
# - Insufficient memory
```

#### 2. **API Key Errors**
```bash
# Verify environment variables
docker-compose exec a2a-agent env | grep OPENAI

# Update .env file and restart
docker-compose restart a2a-agent
```

#### 3. **Permission Issues**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./data ./logs

# Or run with specific user
docker-compose run --user $(id -u):$(id -g) a2a-agent
```

#### 4. **Network Issues**
```bash
# Check container networking
docker network ls
docker network inspect langgraph-a2a-network

# Test connectivity
docker-compose exec a2a-agent curl http://api.openai.com
```

### Debug Mode

```bash
# Run with debug logging
docker-compose run -e LOG_LEVEL=DEBUG a2a-agent

# Interactive shell
docker-compose exec a2a-agent bash
```

### Reset and Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes and networks
docker-compose down -v

# Remove images
docker rmi langgraph-a2a-agent

# Complete cleanup
docker system prune -a
```

## 🔒 Security Considerations

### Container Security

1. **Non-root user**: Container runs as `appuser` (non-root)
2. **Read-only volumes**: Data directory mounted read-only
3. **Network isolation**: Custom Docker network
4. **Security headers**: Nginx adds security headers
5. **Rate limiting**: Configured in Nginx

### API Key Security

1. **Environment variables**: Never hardcode in Dockerfile
2. **Docker secrets**: Use for production deployments
3. **File permissions**: Secure .env file (600)

```bash
# Secure environment file
chmod 600 .env
```

### Production Hardening

```bash
# Use Docker secrets for API keys
echo "your_openai_key" | docker secret create openai_api_key -

# Update docker-compose.yml to use secrets
secrets:
  openai_api_key:
    external: true
```

### Firewall Configuration

```bash
# Only allow necessary ports
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 10000/tcp  # Block direct access to agent
```

## 🚀 Deployment Scripts

### Automated Deployment Script

```bash
#!/bin/bash
# deploy.sh - Automated deployment script

set -e

echo "🚀 Deploying LangGraph A2A Agent..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose required"; exit 1; }

# Check environment
if [[ ! -f .env ]]; then
    echo "⚠️  Creating .env from template..."
    cp env.template .env
    echo "📝 Please edit .env with your API keys"
    exit 1
fi

# Deploy
echo "🏗️  Building and starting services..."
docker-compose --profile production up --build -d

echo "⏳ Waiting for services to start..."
sleep 30

# Health check
if curl -f http://localhost:10000/.well-known/agent-card.json > /dev/null 2>&1; then
    echo "✅ Deployment successful!"
    echo "🌐 Agent available at: http://localhost:10000"
    echo "📊 Nginx proxy at: http://localhost:80"
else
    echo "❌ Deployment failed - check logs:"
    docker-compose logs --tail=50
    exit 1
fi
```

Make it executable:
```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Specification](https://a2a-sdk.readthedocs.io/)

---

**Navigation**: [🏠 Main](./README.md) | [📋 Assignment Answers](./ASSIGNMENT_ANSWERS.md) | [🤖 Expert Agent](./second_agent/README.md) | [🎬 Demo Setup](./demo_setup.md)
