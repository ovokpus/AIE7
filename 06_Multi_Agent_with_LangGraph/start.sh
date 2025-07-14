#!/bin/bash

# LinkedIn ML Paper Post Generation API - Startup Script
echo "🚀 Starting LinkedIn ML Paper Post Generation API"
echo "=" * 50

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please create a .env file with your API keys:"
    echo ""
    echo "OPENAI_API_KEY=your_openai_api_key_here"
    echo "TAVILY_API_KEY=your_tavily_api_key_here"
    echo "REDIS_URL=redis://localhost:6379"
    echo "DEBUG=false"
    echo ""
    exit 1
fi

# Source environment variables
set -a
source .env
set +a

echo "✅ Environment variables loaded"

# Check if required API keys are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is not set in .env file"
    exit 1
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "❌ TAVILY_API_KEY is not set in .env file"
    exit 1
fi

echo "✅ API keys validated"

# Create content directory if it doesn't exist
mkdir -p content/data
echo "✅ Content directory created"

# Install dependencies if not already installed
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🏗 Starting the application..."
echo ""
echo "📖 API Documentation will be available at:"
echo "   - Swagger UI: http://localhost:8000/docs"
echo "   - ReDoc: http://localhost:8000/redoc"
echo ""
echo "🔗 API Endpoints:"
echo "   - POST /generate-post - Generate LinkedIn post"
echo "   - GET /status/{task_id} - Check task status"
echo "   - POST /verify-post - Verify post quality"
echo "   - POST /batch-generate - Batch generation"
echo "   - GET /health - Health check"
echo ""

# Start the FastAPI application
cd app
python main.py 