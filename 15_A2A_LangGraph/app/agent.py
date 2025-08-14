"""
LangGraph Agent with Tool Integration and A2A Protocol Support.

This module implements the core Agent class that provides a general-purpose
AI assistant with access to web search, academic papers, and document retrieval
capabilities. The agent uses LangGraph for workflow management and includes
intelligent helpfulness evaluation.

Key Features:
    - Web search via Tavily
    - Academic paper search via ArXiv
    - Document retrieval via RAG/Qdrant
    - Helpfulness evaluation loop
    - A2A protocol compliance
    - Streaming response support
"""

import os

from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from app.agent_graph_with_helpfulness import build_agent_graph_with_helpfulness


memory = MemorySaver()

class ResponseFormat(BaseModel):
    """Response format model for structured agent outputs.
    
    This model defines the standard format for agent responses, ensuring
    consistent communication with the A2A protocol and client applications.
    
    Attributes:
        status (Literal): The current status of the request processing.
            - 'input_required': More information needed from user
            - 'completed': Request successfully completed
            - 'error': An error occurred during processing
        message (str): The response message content
    """

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class Agent:
    """General-purpose AI agent with multi-tool capabilities and helpfulness evaluation.
    
    This agent provides a comprehensive AI assistant that can perform web searches,
    academic paper searches, and document retrieval using RAG. It implements a 
    LangGraph workflow with intelligent helpfulness evaluation and supports 
    streaming responses for real-time interaction.
    
    The agent is designed to be A2A protocol compliant and can be used as part
    of a larger agent ecosystem for sophisticated agent-to-agent communication.
    
    Capabilities:
        - Web search using Tavily API
        - Academic paper search using ArXiv API  
        - Document retrieval using RAG with Qdrant vector store
        - Helpfulness evaluation and iterative improvement
        - Streaming response generation
        - Multi-turn conversation support
        
    Attributes:
        SYSTEM_INSTRUCTION (str): Base system prompt for the agent
        FORMAT_INSTRUCTION (str): Instructions for response formatting
        SUPPORTED_CONTENT_TYPES (list): Supported input/output content types
        model (ChatOpenAI): The underlying language model
        graph (CompiledGraph): The LangGraph workflow graph
        
    Example:
        >>> agent = Agent()
        >>> async for response in agent.stream("Tell me about AI", "session_123"):
        ...     print(response['content'])
    """

    SYSTEM_INSTRUCTION = (
        'You are a helpful AI assistant with access to various tools including web search, '
        'academic paper search, and document retrieval. '
        'Use the appropriate tools to answer user questions accurately and thoroughly. '
        'If you cannot find relevant information using the available tools, '
        'clearly state that you were unable to find the requested information.'
    )

    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self) -> None:
        """Initialize the Agent with language model and LangGraph workflow.
        
        Sets up the ChatOpenAI model using environment variables for configuration
        and builds the LangGraph workflow with helpfulness evaluation capabilities.
        The agent uses memory checkpointing for multi-turn conversations.
        
        Environment Variables:
            TOOL_LLM_NAME: Model name (default: 'gpt-4o-mini')
            OPENAI_API_KEY: Required OpenAI API key
            TOOL_LLM_URL: API base URL (default: OpenAI's API)
            
        Raises:
            ValueError: If OPENAI_API_KEY is not set
        """
        self.model = ChatOpenAI(
            model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
            temperature=0,
        )
        # Use the new graph with helpfulness evaluation for A2A protocol compatibility
        self.graph = build_agent_graph_with_helpfulness(
            self.model,
            self.SYSTEM_INSTRUCTION,
            self.FORMAT_INSTRUCTION,
            checkpointer=memory
        )

    async def stream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        """Stream agent responses for real-time interaction.
        
        Processes a user query through the LangGraph workflow and yields
        intermediate status updates and the final response. This enables
        real-time feedback during tool execution and thinking phases.
        
        Args:
            query (str): The user's question or request
            context_id (str): Unique identifier for conversation context/thread
            
        Yields:
            dict[str, Any]: Response dictionaries containing:
                - is_task_complete (bool): Whether the task is finished
                - require_user_input (bool): Whether more input is needed
                - content (str): Status message or response content
                
        Example:
            >>> async for response in agent.stream("What is AI?", "session_123"):
            ...     if response['is_task_complete']:
            ...         print(f"Final: {response['content']}")
            ...     else:
            ...         print(f"Status: {response['content']}")
        """
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Searching for information...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the results...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract and format the final agent response from the graph state.
        
        Retrieves the current state of the LangGraph execution and extracts
        the structured response. Converts the ResponseFormat into the standard
        response dictionary format expected by the A2A protocol.
        
        Args:
            config (dict[str, Any]): Graph configuration containing thread ID
            
        Returns:
            dict[str, Any]: Response dictionary with keys:
                - is_task_complete (bool): True if task is finished successfully
                - require_user_input (bool): True if more input is needed
                - content (str): The response message content
                
        Note:
            This method handles all possible response states including completed,
            input_required, error, and fallback scenarios.
        """
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
