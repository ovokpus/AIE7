# MCP Integration Examples 🚀

This directory contains demonstration scripts that show how to use the Model Context Protocol (MCP) integration with LangGraph.

## Overview

The MCP integration adds powerful local tools to your LangGraph agents, including:

- 📁 **File Operations**: Read, write, and list files/directories
- 📊 **Data Analysis**: Analyze CSV files and calculate statistics  
- 🌐 **Web Utilities**: Validate URLs and check accessibility
- ⚙️  **System Utils**: Get environment info and current time

## Demo Scripts

### 1. Basic MCP Demo (`basic_mcp_demo.py`)

Shows direct usage of MCP tools without LangGraph integration.

```bash
# Run the basic demo
uv run python examples/basic_mcp_demo.py
```

**What it demonstrates:**
- File operations (create, read, list)
- CSV data analysis 
- System utilities
- Mathematical operations
- URL validation

### 2. LangGraph Integration Demo (`langgraph_mcp_demo.py`)

Shows MCP tools working within LangGraph agents.

```bash
# Run the LangGraph integration demo
uv run python examples/langgraph_mcp_demo.py
```

**What it demonstrates:**
- File analysis tasks with agents
- System analysis and reporting  
- Tool comparison (core vs MCP-enhanced)

### 3. Interactive LangGraph Demo (`interactive_langgraph_demo.py`) 🆕

**Interactive chat** with your existing LangGraph agents that now have MCP superpowers!

```bash
# Run the interactive demo (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key-here"
uv run python examples/interactive_langgraph_demo.py
```

**Features:**
- Chat with your existing agents interactively
- Switch between Simple Agent and Helpful Agent
- See tool capabilities and example prompts
- Real-time responses using all 11 tools

**Example interactions:**
- "Create a CSV with sales data and analyze it"
- "List files in current directory and create a summary" 
- "Get system info and save to a report file"
- "Calculate statistics for [10, 20, 30, 40, 50]"

## Sample Messages for Student Loan Context

Here are some realistic example messages you can send to the agent that will trigger multiple MCP tools while retrieving relevant student loan/financial aid information from the vector database:

### 📊 Message 1: Research and Report Generation
```
I need to research Federal Pell Grant eligibility requirements and common borrower complaints. Can you retrieve information about Pell Grant policies from our knowledge base, then create a comprehensive report file with your findings? Please include the current timestamp in the report.
```

**Expected tools used:**
- `retrieve_information` (RAG) - to get Pell Grant policy information from vector DB
- `write_file_content` - to create the comprehensive report
- `get_current_time` - to timestamp the report

### 🎓 Message 2: Policy Analysis and Documentation
```
I'm preparing training materials for our financial aid staff. Can you research Direct Loan Program policies from our database, validate that the official studentaid.gov website is accessible for reference, and then create a training document with the key points? Please timestamp the document.
```

**Expected tools used:**
- `retrieve_information` (RAG) - to get Direct Loan Program information from vector DB
- `validate_url` - to check studentaid.gov accessibility
- `write_file_content` - to create the training document
- `get_current_time` - to timestamp the document

### 🌐 Message 3: Environment Setup and Knowledge Retrieval
```
I need to check our system environment to ensure it's properly configured for financial aid data analysis, then research complaint patterns about loan servicing issues from our knowledge base. Please save a technical summary with both the system info and the complaint analysis.
```

**Expected tools used:**
- `get_environment_info` - to check system configuration
- `retrieve_information` (RAG) - to research loan servicing complaints from vector DB
- `write_file_content` - to save the technical summary

### 📈 Message 4: Statistical Analysis and Research Combo
```
I want to analyze some numerical data and also research academic calendar policies. First, calculate statistics for these loan amounts: [5500, 6500, 7500, 12500, 20500]. Then retrieve information about academic calendar requirements from our knowledge base and save everything to a comprehensive analysis file with today's timestamp.
```

**Expected tools used:**
- `calculate_statistics` - to analyze the loan amount data
- `retrieve_information` (RAG) - to get academic calendar policy info from vector DB
- `get_current_time` - to timestamp the analysis
- `write_file_content` - to save the comprehensive analysis

## Generated Files

The demos create sample files in this directory:

- `test_demo.txt` - Basic file operations test
- `sample_data.csv` - Sample CSV for analysis  
- `products.csv` - Product data for agent analysis
- `fastapi_research.txt` - Agent research report
- `system_report.txt` - System analysis report

## Standalone MCP Server

If you want to run the MCP server as a **separate service** (for other applications to connect to):

```bash
# Run the standalone MCP server
uv run python run_mcp_server.py
```

**Note:** The LangGraph integration uses **direct function calls** by default (faster and simpler). The standalone server is useful for:
- Testing MCP protocol compliance
- Connecting from other applications  
- Running as a microservice

## Requirements

- Python 3.13+
- All dependencies from `pyproject.toml`
- OpenAI API key (for LangGraph demos)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LangGraph     │    │   MCP Client    │    │   MCP Server    │
│     Agent       │───▶│   (Adapter)     │───▶│   (Tools)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

The MCP Client acts as an adapter, converting MCP tools into LangChain-compatible tools that can be used by LangGraph agents.

## Customization

You can extend the system by:

1. **Adding new tools** to `app/mcp_server.py`
2. **Updating tool definitions** in `app/mcp_client.py`
3. **Creating new demo scenarios** in this directory

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the project root
2. **Missing API key**: Set `OPENAI_API_KEY` for LangGraph demos
3. **Permission errors**: Check file/directory permissions

### Debug Mode

Add logging to see what's happening:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

- Try modifying the demo scripts for your use cases
- Add custom tools to the MCP server
- Integrate with your own LangGraph workflows
- Explore the full LangGraph documentation for advanced patterns

Happy coding! 🎉
