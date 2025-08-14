"""
A2A Client Wrapper for Expert Agent Communication.

This module provides a wrapper around the A2A protocol client to facilitate
communication between the Expert Agent System and the main A2A agent server.
It handles connection management, message formatting, and response processing
while maintaining proper A2A protocol compliance.

Key Features:
    - A2A protocol-compliant communication
    - Automatic connection management and retry logic
    - Expert persona context integration
    - Response parsing and error handling
    - Server connectivity validation

The wrapper abstracts the complexity of the A2A protocol and provides a
clean interface for expert agents to communicate with the main agent.

Example:
    >>> client = A2AClientWrapper()
    >>> await client.initialize()
    >>> response = await client.send_message_with_context(
    ...     "What is machine learning?",
    ...     persona_context="You are an ML expert..."
    ... )
"""

import time
from typing import Dict, Any, Optional
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient as A2AProtocolClient
from a2a.types import MessageSendParams, SendMessageRequest

from .logging_config import get_loggers

logger, _ = get_loggers()


class A2AClientWrapper:
    """A2A protocol client wrapper for expert agent communication.
    
    This class provides a high-level interface for expert agents to communicate
    with the main A2A agent server. It handles the complexity of A2A protocol
    message formatting, connection management, and response parsing.
    
    The wrapper integrates expert persona context into A2A communications,
    enabling sophisticated agent-to-agent interactions with goal-oriented
    behavior and quality evaluation.
    
    Attributes:
        base_url (str): The base URL of the A2A agent server
        client (A2AProtocolClient): The underlying A2A protocol client
        agent_card (AgentCard): The retrieved agent capability metadata
        httpx_client (httpx.AsyncClient): HTTP client for making requests
        
    Example:
        >>> client = A2AClientWrapper('http://localhost:10000')
        >>> await client.initialize()
        >>> response = await client.send_message_with_context(
        ...     message="Explain transformers",
        ...     persona_context="You are an ML researcher..."
        ... )
    """
    
    def __init__(self, base_url: str = 'http://localhost:10000') -> None:
        """Initialize the A2A client wrapper.
        
        Args:
            base_url (str): Base URL of the A2A agent server. 
                Defaults to 'http://localhost:10000'
        """
        self.base_url = base_url
        self.client = None
        self.agent_card = None
        self.httpx_client = None

    async def initialize(self) -> None:
        """Initialize the A2A client connection"""
        try:
            logger.info("🔗 Initializing A2A client connection...")
            logger.info(f"📡 Connecting to: {self.base_url}")
            
            self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.base_url
            )
            
            logger.info("📋 Fetching agent card...")
            self.agent_card = await resolver.get_agent_card()
            
            logger.info(f"🤖 Main agent capabilities: {self.agent_card.skills}")
            
            self.client = A2AProtocolClient(
                httpx_client=self.httpx_client,
                agent_card=self.agent_card
            )
            logger.info(f"✅ A2A Client initialized for '{self.agent_card.name}'")
            logger.info(f"🎯 Available skills: {[skill.name for skill in self.agent_card.skills]}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize A2A client: {e}")
            return False

    async def send_message_with_context(self, message: str, persona_context: str = None) -> Dict[str, Any]:
        """Send a message to the A2A agent with optional persona context"""
        if not self.client:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        # Add persona context if provided
        full_message = message
        if persona_context:
            full_message = f"Context: {persona_context}\n\nQuery: {message}"
            logger.info(f"🎭 Adding persona context: {persona_context[:50]}...")
        
        logger.info(f"📝 Original query: {message}")
        logger.info(f"📤 Sending enhanced message to main agent...")
        
        payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': full_message}],
                'message_id': uuid4().hex,
            }
        }
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload)
        )
        
        try:
            logger.info("⏳ Waiting for main agent response...")
            start_time = time.time()
            
            response = await self.client.send_message(request)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.info(f"📥 Received response from main agent ({response_time:.2f}s)")
            
            return response.model_dump(mode='json', exclude_none=True)
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close the httpx client"""
        if self.httpx_client:
            await self.httpx_client.aclose()


async def check_server_status(base_url: str = "http://localhost:10000") -> bool:
    """Check if the A2A server is running"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(f"{base_url}/.well-known/agent-card.json")
            if response.status_code == 200:
                print("✅ A2A agent server is running")
                return True
            else:
                print(f"❌ A2A server responded with status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Cannot connect to A2A server: {e}")
        print("💡 Please start your A2A agent with: uv run python -m app")
        return False
