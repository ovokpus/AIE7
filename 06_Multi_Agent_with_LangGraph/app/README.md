# LinkedIn ML Paper Post Generation API

A FastAPI backend that uses multi-agent LangGraph workflows to generate engaging LinkedIn posts about machine learning research papers.

## 🚀 Features

- **Multi-Agent System**: Hierarchical team structure with content creation and verification teams
- **Research Automation**: Automatically researches ML papers using web search and Tavily
- **LinkedIn Optimization**: Creates posts optimized for LinkedIn engagement
- **Quality Assurance**: Technical accuracy verification and style compliance checking
- **Async Processing**: Background task processing with status tracking
- **Batch Operations**: Generate multiple posts in batch with scheduling
- **RESTful API**: Clean REST endpoints with OpenAPI documentation

## 📋 System Architecture

```
🎯 Meta-Supervisor 
    ↓
👥 Content Team
    ├── 🔬 Paper Researcher (research_ml_paper, tavily_tool)
    └── ✍️  LinkedIn Creator (create_linkedin_post)
    ↓
🎯 Meta-Supervisor 
    ↓
✅ Verification Team  
    ├── 🔍 Tech Verifier (verify_technical_accuracy)
    └── 📱 Style Checker (check_linkedin_style)
    ↓
🏁 Final LinkedIn Post
```

## 🛠 Installation

### Prerequisites

- Python 3.8+
- OpenAI API Key
- Tavily API Key
- Redis (optional, for task storage)

### Setup

1. **Clone and navigate to the project:**
```bash
cd app/
```

2. **Install dependencies:**
```bash
pip install -r ../requirements.txt
```

3. **Set up environment variables:**
```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
REDIS_URL=redis://localhost:6379
DEBUG=false
EOF
```

4. **Run the application:**
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 API Endpoints

### Core Endpoints

#### `POST /generate-post`
Generate a LinkedIn post about an ML paper.

**Request Body:**
```json
{
  "paper_title": "Attention Is All You Need",
  "additional_context": "Focus on transformer applications",
  "target_audience": "professional",
  "include_technical_details": true,
  "max_hashtags": 10,
  "tone": "professional"
}
```

**Response:**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "message": "LinkedIn post generation started successfully",
  "estimated_completion_time": "2024-01-15T10:30:00Z"
}
```

#### `GET /status/{task_id}`
Check the status of a post generation task.

**Response:**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "progress": 1.0,
  "current_step": "completed",
  "result": {
    "content": "🚀 **Transforming the Future of AI: Attention Is All You Need**\n\n...",
    "hashtags": ["#MachineLearning", "#AI", "#Research"],
    "word_count": 150,
    "character_count": 890
  },
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:03:00Z"
}
```

#### `POST /verify-post`
Verify a LinkedIn post for technical accuracy and style compliance.

**Request Body:**
```json
{
  "post_content": "Your LinkedIn post content here...",
  "paper_reference": "Attention Is All You Need paper",
  "verification_type": "both"
}
```

#### `POST /batch-generate`
Generate multiple LinkedIn posts in batch.

**Request Body:**
```json
{
  "papers": [
    {
      "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
      "tone": "professional"
    },
    {
      "paper_title": "GPT-3: Language Models are Few-Shot Learners",
      "tone": "academic"
    }
  ],
  "schedule_posts": true,
  "time_interval_minutes": 60
}
```

### Utility Endpoints

#### `GET /health`
Health check with service status.

#### `GET /`
Basic health check and version info.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `TAVILY_API_KEY` | Tavily search API key (required) | - |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `DEBUG` | Enable debug mode | `false` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

### Application Settings

The application can be configured via `app/config.py`:

```python
class Settings(BaseSettings):
    # API Configuration
    app_name: str = "LinkedIn ML Paper Post Generation API"
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    
    # Rate limiting, file storage, etc.
```

## 🎯 Usage Examples

### Generate a Single Post

```python
import requests

response = requests.post("http://localhost:8000/generate-post", json={
    "paper_title": "Attention Is All You Need",
    "additional_context": "Focus on practical NLP applications",
    "target_audience": "professional"
})

task_id = response.json()["task_id"]

# Check status
status_response = requests.get(f"http://localhost:8000/status/{task_id}")
print(status_response.json())
```

### Verify a Post

```python
verification_response = requests.post("http://localhost:8000/verify-post", json={
    "post_content": "Your LinkedIn post content...",
    "verification_type": "both"
})

print(verification_response.json())
```

## 🏗 Development

### Project Structure

```
app/
├── __init__.py
├── main.py                 # FastAPI application
├── config.py              # Configuration settings
├── models/                # Pydantic models
│   ├── requests.py        # Request models
│   ├── responses.py       # Response models
│   └── state.py          # LangGraph state models
├── tools/                 # LangChain tools
│   ├── linkedin_tools.py  # LinkedIn-specific tools
│   └── search_tools.py    # Research and search tools
└── agents/               # Multi-agent system
    ├── helpers.py        # Agent helper functions
    ├── content_team.py   # Content creation team
    ├── verification_team.py # Verification team
    └── meta_supervisor.py  # Meta-supervisor coordination
```

### Key Components

1. **Multi-Agent System**: Uses LangGraph for hierarchical agent coordination
2. **Tools**: Custom tools for LinkedIn post creation and verification
3. **Background Tasks**: Async processing with FastAPI BackgroundTasks
4. **State Management**: Redis for distributed task storage
5. **Error Handling**: Comprehensive error handling and logging

## 🔍 Testing

### Manual Testing

1. Start the server:
```bash
python app/main.py
```

2. Visit http://localhost:8000/docs for interactive API testing

3. Test with curl:
```bash
curl -X POST "http://localhost:8000/generate-post" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_title": "Attention Is All You Need",
    "tone": "professional"
  }'
```

### Integration Testing

The system integrates with:
- OpenAI GPT models for text generation
- Tavily for web search and research
- Redis for task storage (optional)

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

1. **Environment**: Set `DEBUG=false` in production
2. **Security**: Configure CORS appropriately
3. **Scaling**: Use Redis for shared task storage across instances
4. **Monitoring**: Add logging and metrics collection
5. **Rate Limiting**: Implement API rate limiting
6. **Queue System**: Consider Celery for production task queue

## 📝 Example Output

**Generated LinkedIn Post:**
```
🚀 **Transforming the Future of AI: Attention Is All You Need**

The groundbreaking paper that introduced the Transformer architecture has fundamentally changed how we approach natural language processing and machine learning.

💡 **Key Takeaways:**

1. Self-attention mechanisms enable parallel processing, dramatically improving training efficiency
2. Eliminates the need for recurrent layers, solving long-range dependency problems
3. Achieved state-of-the-art results on translation tasks with simpler architecture
4. Became the foundation for modern models like BERT, GPT, and T5

What are your thoughts on this research? How do you see it impacting your industry?

#MachineLearning #AI #Research #Innovation #TechTrends #Transformers #NLP #DeepLearning
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the configuration in `config.py`
3. Check logs for error details
4. Ensure API keys are properly configured

## 🔄 Workflow Details

The system follows this workflow:

1. **Request Processing**: FastAPI receives the request and creates a background task
2. **Meta-Supervisor**: Routes the task to the appropriate team
3. **Content Team**: 
   - Paper Researcher searches and analyzes the ML paper
   - LinkedIn Creator generates an optimized post
4. **Verification Team**:
   - Tech Verifier checks technical accuracy
   - Style Checker ensures LinkedIn compliance
5. **Response**: Returns the verified, publication-ready LinkedIn post

This multi-agent approach ensures high-quality, accurate, and engaging LinkedIn content about ML research. 