"""
A2A Protocol Agent Executor Implementation.

This module implements the GeneralAgentExecutor class that bridges the
LangGraph Agent with the A2A (Agent-to-Agent) protocol. It handles request
processing, streaming responses, and task state management according to
A2A specifications.

Key Features:
    - A2A protocol compliance
    - Streaming response handling
    - Task state management
    - Error handling and validation
    - Integration with LangGraph workflows

The executor serves as the main interface between external A2A clients
and the internal LangGraph agent implementation.
"""

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from app.agent import Agent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralAgentExecutor(AgentExecutor):
    """A2A protocol-compliant executor for the LangGraph Agent.
    
    This class implements the AgentExecutor interface required by the A2A
    protocol framework. It manages the execution lifecycle of agent requests,
    including streaming responses, task state updates, and error handling.
    
    The executor bridges the gap between A2A protocol requirements and the
    LangGraph agent implementation, ensuring proper task management and
    response streaming according to A2A specifications.
    
    Attributes:
        agent (Agent): The LangGraph agent instance that performs the actual work
        
    Example:
        >>> executor = GeneralAgentExecutor()
        >>> await executor.execute(context, event_queue)
    """

    def __init__(self) -> None:
        """Initialize the executor with a LangGraph Agent instance.
        
        Creates a new Agent instance that will handle all query processing
        and response generation through the LangGraph workflow.
        """
        self.agent = Agent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute an agent request with A2A protocol compliance.
        
        This method processes a user request through the LangGraph agent and
        manages the A2A task lifecycle including status updates, streaming
        responses, and proper task completion or error handling.
        
        Args:
            context (RequestContext): The A2A request context containing user input
                and task information
            event_queue (EventQueue): A2A event queue for publishing task updates
                and responses
                
        Raises:
            ServerError: If request validation fails or agent execution encounters
                an error. Wraps InvalidParamsError for validation failures and
                InternalError for execution failures.
                
        Note:
            The method handles three response states:
            - Working: Agent is processing (streams intermediate updates)
            - Input Required: Agent needs more information (requests user input)
            - Complete: Agent finished successfully (provides final result)
        """
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            logger.info(f"Starting agent stream for query: {query}")
            async for item in self.agent.stream(query, task.context_id):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']
                logger.info(f"Stream item - complete: {is_task_complete}, requires_input: {require_user_input}")

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='result',
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate the incoming A2A request context.
        
        Performs basic validation on the request context to ensure it contains
        the necessary information for agent execution.
        
        Args:
            context (RequestContext): The A2A request context to validate
            
        Returns:
            bool: True if validation fails (error detected), False if valid
            
        Note:
            Currently returns False (no validation errors) as a placeholder.
            Future implementations may add specific validation logic.
        """
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel an ongoing agent execution.
        
        This method is called when a client requests cancellation of an ongoing
        agent task. Currently not implemented as the LangGraph agent doesn't
        support mid-execution cancellation.
        
        Args:
            context (RequestContext): The A2A request context for the task to cancel
            event_queue (EventQueue): A2A event queue for publishing cancellation events
            
        Raises:
            ServerError: Always raises UnsupportedOperationError as cancellation
                is not currently supported by the agent implementation.
        """
        raise ServerError(error=UnsupportedOperationError())
