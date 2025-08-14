#!/bin/bash
# 🚀 LangGraph A2A Agent Deployment Script
# Automated deployment with health checks and validation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Configuration
CONTAINER_NAME="langgraph-a2a-agent"
SERVICE_NAME="a2a-agent"
HEALTH_URL="http://localhost:10000/.well-known/agent-card.json"
MAX_WAIT_TIME=60

echo "🚀 LangGraph A2A Agent Deployment"
echo "=================================="

# Check prerequisites
log_info "Checking prerequisites..."

command -v docker >/dev/null 2>&1 || { 
    log_error "Docker is required but not installed. Please install Docker first."
    exit 1
}

command -v docker-compose >/dev/null 2>&1 || { 
    log_error "Docker Compose is required but not installed. Please install Docker Compose first."
    exit 1
}

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon is not running. Please start Docker first."
    exit 1
fi

log_success "Prerequisites check passed"

# Check environment configuration
log_info "Checking environment configuration..."

if [[ ! -f .env ]]; then
    log_warning "No .env file found. Creating from template..."
    if [[ -f env.template ]]; then
        cp env.template .env
        log_warning "Please edit .env file with your API keys before running again:"
        log_warning "  - OPENAI_API_KEY (required)"
        log_warning "  - TAVILY_API_KEY (optional for web search)"
        exit 1
    else
        log_error "Neither .env nor env.template found. Please create environment configuration."
        exit 1
    fi
fi

# Check for required API key
if ! grep -q "OPENAI_API_KEY=" .env || grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env; then
    log_error "OPENAI_API_KEY is not configured in .env file. Please set your OpenAI API key."
    exit 1
fi

log_success "Environment configuration OK"

# Parse command line arguments
PROFILE="development"
DETACHED=""
BUILD="--build"

while [[ $# -gt 0 ]]; do
    case $1 in
        --production)
            PROFILE="production"
            shift
            ;;
        --dev|--development)
            PROFILE="development"
            shift
            ;;
        -d|--detached)
            DETACHED="-d"
            shift
            ;;
        --no-build)
            BUILD=""
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --production     Deploy with production profile (includes Nginx)"
            echo "  --development    Deploy with development profile (default)"
            echo "  -d, --detached   Run in detached mode"
            echo "  --no-build       Skip building images"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Stop existing containers
log_info "Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Build and start services
if [[ "$PROFILE" == "production" ]]; then
    log_info "Deploying with production profile (includes Nginx proxy)..."
    docker-compose --profile production up $BUILD $DETACHED
else
    log_info "Deploying with development profile..."
    docker-compose up $BUILD $DETACHED
fi

# If running in detached mode, wait for services and perform health check
if [[ -n "$DETACHED" ]]; then
    log_info "Waiting for services to start..."
    
    # Wait for container to be healthy
    WAIT_COUNT=0
    while [[ $WAIT_COUNT -lt $MAX_WAIT_TIME ]]; do
        if docker-compose ps | grep -q "Up (healthy)"; then
            break
        fi
        
        if docker-compose ps | grep -q "Exit"; then
            log_error "Container exited unexpectedly. Check logs:"
            docker-compose logs --tail=20 $SERVICE_NAME
            exit 1
        fi
        
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 2))
        
        if [[ $((WAIT_COUNT % 10)) -eq 0 ]]; then
            log_info "Still waiting... ($WAIT_COUNT/${MAX_WAIT_TIME}s)"
        fi
    done
    
    # Health check
    log_info "Performing health check..."
    sleep 5  # Give extra time for startup
    
    if curl -f "$HEALTH_URL" >/dev/null 2>&1; then
        log_success "Deployment successful!"
        echo ""
        echo "🌐 Services available at:"
        echo "   • A2A Agent API: http://localhost:10000"
        echo "   • Agent Card: http://localhost:10000/.well-known/agent-card.json"
        
        if [[ "$PROFILE" == "production" ]]; then
            echo "   • Nginx Proxy: http://localhost:80"
            echo "   • Nginx Health: http://localhost:80/health"
        fi
        
        echo ""
        echo "📊 Useful commands:"
        echo "   • View logs: docker-compose logs -f"
        echo "   • Stop services: docker-compose down"
        echo "   • Restart: docker-compose restart"
        
    else
        log_error "Health check failed. Service may not be ready yet."
        log_info "Check logs for more information:"
        docker-compose logs --tail=30 $SERVICE_NAME
        exit 1
    fi
else
    log_info "Running in foreground mode. Press Ctrl+C to stop."
fi
