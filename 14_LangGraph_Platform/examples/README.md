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

Here are some realistic example messages you can send to the agent that will trigger multiple MCP tools while being relevant to the student loan/financial aid data in the `data/` directory:

### 📊 Message 1: Analysis and File Operations
```
I need to understand the current student loan complaint trends. Can you analyze the complaints data in the data directory, then create a summary report file with the key findings about the most common issues borrowers are facing? I'd also like to know what the current time is so I can timestamp this analysis.
```

**Expected MCP tools used:**
- `list_directory_contents` - to explore the data directory
- `analyze_csv_data` - to analyze the complaints.csv file  
- `write_file_content` - to create the summary report
- `get_current_time` - to timestamp the analysis

### 🎓 Message 2: Data Analysis and Research
```
I'm researching Federal Pell Grant eligibility requirements for my school's financial aid office. Can you retrieve information about Pell Grant policies from our database, then analyze the complaints data to see what specific issues students have with Pell Grants? Please save your findings to a file called 'pell_grant_analysis.txt' with today's timestamp.
```

**Expected MCP tools used:**
- `retrieve_information` (RAG tool) - to get Pell Grant policy information
- `analyze_csv_data` - to examine complaints for Pell Grant issues
- `get_current_time` - to get timestamp
- `write_file_content` - to save the analysis

### 🌐 Message 3: File Management and URL Validation
```
I need to validate that the Department of Education's student aid website (https://studentaid.gov) is accessible, then check what files we have in our data directory. After that, create a summary report of our available resources and include the current timestamp.
```

**Expected MCP tools used:**
- `validate_url` - to check studentaid.gov accessibility
- `list_directory_contents` - to check available files
- `get_current_time` - to get timestamp
- `write_file_content` - to create the resource summary

### 📈 Message 4: Statistical Analysis and Environment Check
```
I want to run some statistics on student loan complaint volumes. First, can you analyze the complaints data to get statistical summaries, then check our system environment to make sure we have the right setup for data analysis? Please save a technical report with these findings.
```

**Expected MCP tools used:**
- `analyze_csv_data` with operation='stats' - for statistical analysis
- `get_environment_info` - to check system setup
- `calculate_statistics` - if numerical data is extracted
- `write_file_content` - to save the technical report

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
