from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


class ContentTeamState(TypedDict):
    """State for the content creation team."""
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    paper_title: Optional[str]
    research_findings: Optional[str]
    draft_post: Optional[str]


class VerificationTeamState(TypedDict):
    """State for the verification team."""
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    post_content: Optional[str]
    technical_report: Optional[str]
    style_report: Optional[str]
    verification_complete: bool


class LinkedInMetaState(TypedDict):
    """Meta-state for the overall LinkedIn post generation system."""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    task_id: Optional[str]
    paper_title: Optional[str]
    current_step: Optional[str]
    progress: float
    research_complete: bool
    content_complete: bool
    verification_complete: bool
    final_post: Optional[str]
    error_message: Optional[str]


class TaskState(TypedDict):
    """State for tracking individual tasks."""
    task_id: str
    status: str  # pending, in_progress, completed, failed
    request_data: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: str
    updated_at: str
    progress: float
    current_step: Optional[str]


class BatchTaskState(TypedDict):
    """State for tracking batch tasks."""
    batch_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    task_ids: List[str]
    status: str
    created_at: str
    updated_at: str


# Agent-specific states for internal processing
class ResearchState(TypedDict):
    """State for research agent."""
    query: str
    search_results: Optional[str]
    paper_summary: Optional[str]
    key_insights: Optional[List[str]]
    research_complete: bool


class ContentCreationState(TypedDict):
    """State for content creation agent."""
    paper_info: Dict[str, Any]
    target_audience: str
    tone: str
    include_technical_details: bool
    max_hashtags: int
    draft_content: Optional[str]
    hashtags: Optional[List[str]]
    content_complete: bool


class TechnicalVerificationState(TypedDict):
    """State for technical verification agent."""
    post_content: str
    paper_reference: Optional[str]
    accuracy_score: Optional[float]
    technical_issues: Optional[List[str]]
    verification_report: Optional[Dict[str, Any]]
    verification_complete: bool


class StyleVerificationState(TypedDict):
    """State for style verification agent."""
    post_content: str
    style_score: Optional[float]
    style_issues: Optional[List[str]]
    recommendations: Optional[List[str]]
    linkedin_compliant: bool
    verification_complete: bool 