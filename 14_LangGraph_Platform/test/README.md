# MCP Integration Tests 🧪

---

## 📚 Project Navigation

| Document | Description |
|----------|-------------|
| [📖 Project Overview](../README.md) | Session overview and quick start |
| [🚀 Complete Guide](../LANGGRAPH_MCP_GUIDE.md) | Comprehensive technical documentation and architecture |
| [📋 Assignment Details](../ASSIGNMENT_ANSWERS.md) | Implementation details and technical answers |
| [🔀 Deployment Guide](../MERGE.md) | Development workflow and deployment instructions |
| [💡 Examples & Demos](../examples/README.md) | Usage examples and interactive demos |
| **[🧪 Testing Guide](README.md)** | **You are here** - Testing framework and validation |
| [⚙️ App Documentation](../app/README.md) | Core application architecture |

---

This directory contains tests for the MCP (Model Context Protocol) integration with LangGraph.

## Test Files

### `test_mcp_integration.py`
Comprehensive integration tests that verify:
- ✅ **MCP Tools Availability**: 8 tools are discoverable
- ✅ **Tool Integration**: MCP tools integrate with core tools (11 total)
- ✅ **MCP Tool Execution**: Direct tool function calls work
- ✅ **LangChain Tool Format**: Tools work with LangChain interface

### `test_served_graph.py`
Tests the served LangGraph endpoints (requires running server):
- Tests the simple_agent endpoint
- Verifies MCP tools are available via the platform

### `run_tests.py`
Test runner that executes all tests and provides a summary.

## Running Tests

### Run All Tests
```bash
# From project root
uv run python test/run_tests.py
```

### Run Individual Tests
```bash
# MCP integration tests
uv run python test/test_mcp_integration.py

# Served graph tests (requires server running)
uv run python test/test_served_graph.py
```

## Test Results

When all tests pass, you should see:

```
🎉 All tests passed! MCP integration is working correctly.

✅ Your LangGraph agents now have 8 additional MCP tools:
   📁 File operations, 📊 Data analysis, 🌐 Web utils, ⚙️ System info
```

## What The Tests Verify

1. **Tool Discovery**: All 8 MCP tools are found and have proper schemas
2. **Integration**: MCP tools work alongside existing tools (Tavily, Arxiv, RAG)
3. **Execution**: Each tool can be called and returns expected results
4. **LangChain Compatibility**: Tools work with LangChain's tool interface
5. **Agent Integration**: Your existing agents automatically have access to MCP tools

## Troubleshooting

### Import Errors
- Ensure you're running from the project root
- Check that all dependencies are installed: `uv sync`

### Tool Execution Errors
- Verify MCP tool functions in `app/mcp_tools.py`
- Check MCP client setup in `app/mcp_client.py`

### LangChain Integration Issues
- Ensure proper tool schemas are generated
- Verify `@tool` decorators in `app/mcp_client.py`

## Adding New Tests

To add tests for new MCP tools:

1. Add the tool function to `app/mcp_tools.py`
2. Register it in `app/mcp_server.py` and `app/mcp_client.py`
3. Add test cases to `test_mcp_integration.py`
4. Update tool count expectations in tests

## Dependencies

Tests require:
- All project dependencies (`pyproject.toml`)
- No external API keys (tests use local tools only)
- Python 3.13+
