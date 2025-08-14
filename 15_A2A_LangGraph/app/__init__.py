"""
LangGraph Agent Application Package.

This package contains the complete implementation of a LangGraph agent with 
A2A (Agent-to-Agent) protocol support, featuring intelligent helpfulness 
evaluation and multi-turn conversation capabilities.

The main components include:
- Agent: Core LangGraph agent with tool integration
- AgentExecutor: A2A protocol-compliant request handler
- Tools: Web search, academic search, and RAG capabilities
- Graph: LangGraph workflow with helpfulness evaluation

Example:
    To start the A2A server:
    $ python -m app --host localhost --port 10000
"""
