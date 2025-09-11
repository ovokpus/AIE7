
<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 6: Multi-Agent with LangGraph</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 6: Pre-Work](https://www.notion.so/Session-6-Multi-Agent-Applications-with-LangGraph-22bcd547af3d80f7a2e0cc7a2c6d7d8f?source=copy_link#22bcd547af3d8021ac68fade5a7b9df2)| [Session 6: Multi-Agent Applications with LangGraph](https://www.notion.so/Session-6-Multi-Agent-Applications-with-LangGraph-22bcd547af3d80f7a2e0cc7a2c6d7d8f) | [Recording!](https://us02web.zoom.us/rec/share/CnsbWyce6zleEHYzebhqGcbg0syunLmLkWroRQ7ATRKaz3rDqGa7sj7FQfb0hb8U.aB_oEqnl75nk68ej)  (@2nEaXuk) | [Session 6 Slides](https://www.canva.com/design/DAGstHQ78gU/D_DHLWAO5KZoQ5R1RjG3YA/edit?utm_content=DAGstHQ78gU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 6 Assignment: Multi-agents](https://forms.gle/HScbKATi6nCNYZx57)| [AIE7 Feedback 7/10](https://forms.gle/itQqhBW2PY7DTFi76)

In today's assignment, we'll be creating a MULTI-Agentic LangGraph Application.

- 🤝 Breakout Room #1:
  1. Simple LCEL RAG
  2. Helper Functions for Agent Graphs
  3. Research Team - A LangGraph for Researching A Specific Topic
  
- 🤝 Breakout Room #2:
  1. Document Writing Team - A LangGraph for Writing, Editing, and Planning a LinkedIn post.
  2. Meta-Supervisor and Full Graph

### 🚧 OPTIONAL: Advanced Build

> NOTE: This is an optional task - and is not required to achieve full marks on the assignment.

Build a graph to produce a social media post about a given Machine Learning paper. 

The graph should employ an additional team that verifies the correctness of the produced paper, and verify it fits the theme and style of your selected social media platform.

## Ship 🚢

The completed notebook!

### Deliverables

- A short Loom of the notebook, and a 1min. walkthrough of the application in full

## Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped an Multi-Agent Application with LangGraph! 🎉🤖

🔍 Three Key Takeaways:
1️⃣ 
2️⃣ 
3️⃣ 

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```

## Submitting Your Homework

### Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:
1. Create a branch of your `AIE7` repo to track your changes. Example command: `git checkout -b s06-assignment`
2. Respond to the activities and questions in the `Multi_Agent_RAG_LangGraph.ipynb` notebook:
    + Edit the markdown cells of the activities and questions then enter your responses
    + Edit/Create code cell(s) where necessary as part of an activity
    + NOTE: Remember to create a header (example: `##### ✅ Answer:`) to help the grader find your responses
3. Commit, and push your completed notebook to your `origin` repository. _NOTE: Do not merge it into your main branch._
4. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the completed notebook _on your assignment branch (not main)_
    + The URL to your Loom Video
    + Your Three lessons learned/not yet learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_

### Advanced Build
In addition to the above, include on your homework submission form the URLs to your Advanced Build's:
+ GitHub Repo
+ Production Deployment

# PostAssist - Multi-Agent LinkedIn Post Generation

**PostAssist** is an AI-powered assistant that generates engaging LinkedIn posts about machine learning research papers. This complete **FastAPI backend** uses a sophisticated multi-agent system powered by **LangGraph** to research, create, and verify high-quality content.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and setup
git clone <repository-url>
cd 06_Multi_Agent_with_LangGraph

# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Start with Docker Compose
docker-compose up
```

### Option 2: Manual Setup

```bash
# Make start script executable and run
chmod +x start.sh
./start.sh
```

### Option 3: Development Mode

```bash
# Setup development environment
python scripts/dev.py setup

# Start development server with hot reload
python scripts/dev.py run
```

## 📖 API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🏗 Project Structure

```
06_Multi_Agent_with_LangGraph/
├── app/                          # FastAPI Backend Application
│   ├── main.py                   # FastAPI app with all endpoints
│   ├── config.py                 # Configuration and settings
│   ├── models/                   # Pydantic models
│   │   ├── requests.py           # Request models
│   │   ├── responses.py          # Response models
│   │   └── state.py              # LangGraph state models
│   ├── tools/                    # LangChain tools
│   │   ├── linkedin_tools.py     # LinkedIn-specific tools
│   │   └── search_tools.py       # Research and search tools
│   └── agents/                   # Multi-agent system
│       ├── helpers.py            # Agent utilities
│       ├── content_team.py       # Content creation team
│       ├── verification_team.py  # Verification team
│       └── meta_supervisor.py    # Meta-supervisor
├── examples/                     # Usage examples
│   └── example_client.py         # Complete API client example
├── scripts/                      # Utility scripts
│   ├── dev.py                    # Development utilities
│   ├── test_api.py               # API testing suite
│   └── utils.py                  # General utilities
├── Multi_Agent_RAG_LangGraph.ipynb  # Original notebook
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose setup
└── start.sh                     # Quick start script
```

## 🎯 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate-post` | Generate LinkedIn post about ML paper |
| `GET` | `/status/{task_id}` | Check task status and get results |
| `POST` | `/verify-post` | Verify post technical accuracy & style |
| `POST` | `/batch-generate` | Generate multiple posts in batch |
| `GET` | `/health` | Health check and service status |

### Example Usage

```python
import requests

# Generate a LinkedIn post
response = requests.post("http://localhost:8000/generate-post", json={
    "paper_title": "Attention Is All You Need",
    "additional_context": "Focus on practical NLP applications",
    "target_audience": "professional",
    "tone": "professional"
})

task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/status/{task_id}")
print(status.json())
```

## 🧠 Multi-Agent Architecture

The system uses a hierarchical multi-agent structure:

```
🎯 Meta-Supervisor 
    ↓
👥 Content Creation Team
    ├── 🔬 Paper Researcher (Tavily search, ArXiv research)
    └── ✍️  LinkedIn Creator (Post generation, hashtag optimization)
    ↓
🎯 Meta-Supervisor 
    ↓
✅ Verification Team  
    ├── 🔍 Technical Verifier (Accuracy checking)
    └── 📱 Style Checker (LinkedIn compliance)
    ↓
🏁 Final LinkedIn Post
```

### Key Features

- **Autonomous Operation**: Each agent works independently within its specialty
- **Quality Assurance**: Two-stage verification for technical accuracy and style
- **Scalable Design**: Easy to add new agents or modify existing workflows
- **Robust Error Handling**: Comprehensive error handling and recovery
- **Async Processing**: Background task processing with real-time status updates

## 🛠 Development

### Development Commands

```bash
# Setup development environment
python scripts/dev.py setup

# Run development server (with hot reload)
python scripts/dev.py run

# Run test suite
python scripts/dev.py test

# Format code
python scripts/dev.py format

# Lint code
python scripts/dev.py lint

# Type checking
python scripts/dev.py typecheck

# Run all checks + start server
python scripts/dev.py all
```

### Testing

```bash
# Run comprehensive API tests
python scripts/test_api.py

# Test specific endpoint
python scripts/test_api.py --url http://localhost:8000

# Save test results
python scripts/test_api.py --output test_results.json
```

### Utilities

```bash
# Generate sample requests
python scripts/utils.py samples

# Create cURL examples
python scripts/utils.py curl

# Check API health
python scripts/utils.py health

# Create deployment checklist
python scripts/utils.py checklist
```

## 🔧 Configuration

### Required Environment Variables

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000
REDIS_URL=redis://localhost:6379
```

### Application Settings

The application can be configured via `app/config.py`:
- OpenAI model selection and parameters
- Rate limiting settings
- File storage configuration  
- Logging levels
- LangGraph recursion limits

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Set production environment
export DEBUG=false

# Run with production server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Production Checklist

Run the deployment checklist generator:

```bash
python scripts/utils.py checklist
```

This creates a comprehensive `DEPLOYMENT_CHECKLIST.md` with all necessary deployment steps.

## 📊 Example Output

**Generated LinkedIn Post:**
```
🚀 **Transforming the Future of AI: Attention Is All You Need**

The groundbreaking paper that introduced the Transformer architecture has 
fundamentally changed how we approach natural language processing.

💡 **Key Takeaways:**

1. Self-attention mechanisms enable parallel processing
2. Eliminates the need for recurrent layers
3. Achieved state-of-the-art results with simpler architecture
4. Became the foundation for BERT, GPT, and T5

What are your thoughts on this research? How do you see it impacting your industry?

#MachineLearning #AI #Research #Innovation #TechTrends #Transformers #NLP
```

## 🔗 Related Files

- **Original Notebook**: `Multi_Agent_RAG_LangGraph.ipynb` - Contains the original implementation and detailed explanations
- **API Documentation**: `app/README.md` - Detailed API documentation
- **Examples**: `examples/` - Complete usage examples and sample code
- **Docker Config**: `Dockerfile` & `docker-compose.yml` - Containerization setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python scripts/test_api.py`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the [API documentation](http://localhost:8000/docs) 
2. Review the [deployment checklist](DEPLOYMENT_CHECKLIST.md)
3. Run the health check: `python scripts/utils.py health`
4. Check the logs for error details
5. Ensure API keys are properly configured

## 🎉 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for high-performance async API
- Powered by [LangGraph](https://langchain-ai.github.io/langgraph/) for multi-agent orchestration
- Uses [OpenAI GPT](https://openai.com/) models for intelligent content generation
- Integrates [Tavily](https://tavily.com/) for comprehensive web search