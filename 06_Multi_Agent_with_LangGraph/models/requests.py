from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class PostGenerationRequest(BaseModel):
    """Request model for generating LinkedIn posts about ML papers."""
    
    paper_title: str = Field(
        description="Title or topic of the ML paper to create a post about",
        min_length=5,
        max_length=500,
        examples=["Attention Is All You Need (Transformer architecture)"]
    )
    
    additional_context: Optional[str] = Field(
        default=None,
        description="Additional context or specific aspects to focus on",
        max_length=1000,
        examples=["Focus on practical applications in NLP"]
    )
    
    target_audience: Optional[str] = Field(
        default="professional",
        description="Target audience for the LinkedIn post",
        examples=["professional"]
    )
    
    include_technical_details: bool = Field(
        default=True,
        description="Whether to include technical details in the post"
    )
    
    max_hashtags: int = Field(
        default=10,
        description="Maximum number of hashtags to include",
        ge=1,
        le=20
    )
    
    tone: Optional[str] = Field(
        default="professional",
        description="Tone of the post (professional, casual, academic)",
        examples=["professional"]
    )

    @field_validator('paper_title')
    @classmethod
    def validate_paper_title(cls, v):
        if not v.strip():
            raise ValueError('Paper title cannot be empty')
        return v.strip()


class PostStatusRequest(BaseModel):
    """Request model for checking post generation status."""
    
    task_id: str = Field(
        description="Task ID returned from the post generation request",
        examples=["123e4567-e89b-12d3-a456-426614174000"]
    )


class PostVerificationRequest(BaseModel):
    """Request model for verifying a generated post."""
    
    post_content: str = Field(
        description="The LinkedIn post content to verify",
        min_length=10,
        max_length=3000
    )
    
    paper_reference: Optional[str] = Field(
        default=None,
        description="Reference information about the source paper",
        max_length=1000
    )
    
    verification_type: str = Field(
        default="both",
        description="Type of verification: 'technical', 'style', or 'both'",
        examples=["both"]
    )

    @field_validator('verification_type')
    @classmethod
    def validate_verification_type(cls, v):
        allowed_types = {'technical', 'style', 'both'}
        if v not in allowed_types:
            raise ValueError(f'Verification type must be one of: {allowed_types}')
        return v


class BatchPostRequest(BaseModel):
    """Request model for generating multiple LinkedIn posts."""
    
    papers: List[PostGenerationRequest] = Field(
        description="List of papers to generate posts for",
        min_length=1,
        max_length=5
    )
    
    schedule_posts: bool = Field(
        default=False,
        description="Whether to schedule the posts for publishing"
    )
    
    time_interval_minutes: Optional[int] = Field(
        default=60,
        description="Time interval between posts in minutes",
        ge=30,
        le=1440  # 24 hours
    ) 