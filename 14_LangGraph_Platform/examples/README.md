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

Here are some realistic example messages you can send to the agent that will trigger multiple tools including web search, academic research, vector database retrieval, and MCP utilities:

### 📊 Message 1: Comprehensive Research and Report Generation
```
I need to research Federal Pell Grant eligibility requirements from multiple sources. Please retrieve information about Pell Grant policies from our knowledge base, search the web for the latest news about Pell Grant changes in 2024, and also find recent academic research papers about financial aid effectiveness. Then create a comprehensive report file with all your findings and include the current timestamp.
```

**Expected tools used:**
- `retrieve_information` (RAG) - to get Pell Grant policy information from vector DB
- `TavilySearchResults` (Web Search) - to find latest news about Pell Grant changes
- `ArxivQueryRun` (Academic Search) - to find research papers on financial aid effectiveness
- `write_file_content` - to create the comprehensive report
- `get_current_time` - to timestamp the report

### 🎓 Message 2: Multi-Source Policy Analysis and Documentation
```
I'm preparing training materials for our financial aid staff about student loan default rates and prevention strategies. Can you research Direct Loan Program policies from our database, search for recent academic studies on student loan default prevention, look up current news about loan default trends, and validate that the official studentaid.gov website is accessible? Then create a training document with all the key points and timestamp it.
```

**Expected tools used:**
- `retrieve_information` (RAG) - to get Direct Loan Program information from vector DB
- `ArxivQueryRun` (Academic Search) - to find studies on loan default prevention
- `TavilySearchResults` (Web Search) - to find news about default trends
- `validate_url` - to check studentaid.gov accessibility
- `write_file_content` - to create the training document
- `get_current_time` - to timestamp the document

### 🌐 Message 3: Technology and Research Integration Analysis
```
I need to understand the current state of financial aid technology systems. Please check our system environment configuration, search for recent academic papers about fintech in education, look up current news about student loan management platforms, and research complaint patterns about loan servicing technology issues from our knowledge base. Save a comprehensive technical analysis with all findings.
```

**Expected tools used:**
- `get_environment_info` - to check system configuration
- `ArxivQueryRun` (Academic Search) - to find papers about fintech in education
- `TavilySearchResults` (Web Search) - to find news about loan management platforms
- `retrieve_information` (RAG) - to research loan servicing tech complaints from vector DB
- `write_file_content` - to save the technical analysis

### 📈 Message 4: Complete Multi-Source Statistical and Policy Analysis
```
I want to do a comprehensive analysis combining numerical data with research from multiple sources. First, calculate statistics for these loan amounts: [5500, 6500, 7500, 12500, 20500]. Then search for recent academic research on optimal loan amounts for student success, look up current web news about student debt trends, retrieve information about academic calendar policies from our knowledge base, and save everything to a comprehensive analysis file with today's timestamp.
```

**Expected tools used:**
- `calculate_statistics` - to analyze the loan amount data
- `ArxivQueryRun` (Academic Search) - to find research on optimal loan amounts
- `TavilySearchResults` (Web Search) - to find news about student debt trends
- `retrieve_information` (RAG) - to get academic calendar policy info from vector DB
- `get_current_time` - to timestamp the analysis
- `write_file_content` - to save the comprehensive analysis

### 🔍 Message 5: Advanced Multi-Tool Research Workflow
```
I'm conducting a comprehensive study on income-driven repayment plans. Please search for recent academic papers about IDR plan effectiveness, find current news about IDR program changes, retrieve our internal policy information about income-driven repayment from the knowledge base, verify that the Federal Student Aid website is accessible, check our system environment for data analysis capabilities, and create a detailed research report with current timestamp.
```

**Expected tools used:**
- `ArxivQueryRun` (Academic Search) - to find papers about IDR effectiveness
- `TavilySearchResults` (Web Search) - to find news about IDR program changes
- `retrieve_information` (RAG) - to get IDR policy info from vector DB
- `validate_url` - to check Federal Student Aid website accessibility
- `get_environment_info` - to verify system capabilities
- `write_file_content` - to create the research report
- `get_current_time` - to timestamp the report

## Generated Files

The demo scripts create sample files in this directory:

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
