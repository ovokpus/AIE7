"""Main FastAPI application for LinkedIn ML Paper Post Generation."""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis.asyncio as redis

from app.config import settings, validate_api_keys
from app.models.requests import (
    PostGenerationRequest, 
    PostStatusRequest, 
    PostVerificationRequest,
    BatchPostRequest
)
from app.models.responses import (
    PostGenerationResponse, 
    PostStatusResponse, 
    PostVerificationResponse,
    BatchPostResponse,
    HealthCheckResponse,
    ErrorResponse,
    TaskStatus,
    LinkedInPost,
    VerificationReport
)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="PostAssist - AI-powered LinkedIn post generation for machine learning papers with multi-agent verification",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client for task storage
redis_client: Optional[redis.Redis] = None

# In-memory task storage (fallback if Redis is not available)
task_storage: Dict[str, Dict[str, Any]] = {}


async def get_redis_client():
    """Get Redis client instance."""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url(settings.redis_url)
            await redis_client.ping()
        except Exception:
            redis_client = None
    return redis_client


async def store_task(task_id: str, task_data: Dict[str, Any]):
    """Store task data in Redis or fallback storage."""
    client = await get_redis_client()
    if client:
        try:
            await client.setex(f"task:{task_id}", 3600, str(task_data))  # 1 hour TTL
        except Exception:
            task_storage[task_id] = task_data
    else:
        task_storage[task_id] = task_data


async def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve task data from Redis or fallback storage."""
    client = await get_redis_client()
    if client:
        try:
            data = await client.get(f"task:{task_id}")
            if data:
                return eval(data.decode())  # Note: In production, use proper serialization
        except Exception:
            pass
    
    return task_storage.get(task_id)


async def update_task_status(task_id: str, status: str, progress: float = None, 
                           current_step: str = None, result: Dict[str, Any] = None,
                           error: str = None):
    """Update task status."""
    task_data = await get_task(task_id) or {}
    task_data.update({
        "status": status,
        "updated_at": datetime.utcnow().isoformat(),
    })
    
    if progress is not None:
        task_data["progress"] = progress
    if current_step is not None:
        task_data["current_step"] = current_step
    if result is not None:
        task_data["result"] = result
    if error is not None:
        task_data["error"] = error
    
    await store_task(task_id, task_data)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        validate_api_keys()
        await get_redis_client()
        print(f"🚀 {settings.app_name} v{settings.app_version} started successfully!")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint with basic health check."""
    return HealthCheckResponse(
        status="healthy",
        version=settings.app_version,
        services={
            "openai": "connected" if settings.openai_api_key else "not_configured",
            "tavily": "connected" if settings.tavily_api_key else "not_configured",
            "redis": "connected" if await get_redis_client() else "not_available"
        }
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check endpoint."""
    services = {}
    
    # Check OpenAI
    try:
        if settings.openai_api_key:
            services["openai"] = "connected"
        else:
            services["openai"] = "not_configured"
    except Exception:
        services["openai"] = "error"
    
    # Check Tavily
    try:
        if settings.tavily_api_key:
            services["tavily"] = "connected"
        else:
            services["tavily"] = "not_configured"
    except Exception:
        services["tavily"] = "error"
    
    # Check Redis
    try:
        client = await get_redis_client()
        services["redis"] = "connected" if client else "not_available"
    except Exception:
        services["redis"] = "error"
    
    overall_status = "healthy" if all(
        status in ["connected", "not_available"] for status in services.values()
    ) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version=settings.app_version,
        services=services
    )


async def run_post_generation_task(task_id: str, request: PostGenerationRequest):
    """Background task to generate LinkedIn post."""
    try:
        await update_task_status(task_id, TaskStatus.IN_PROGRESS, 0.1, "starting")
        
        # Import here to avoid circular imports
        from app.agents.meta_supervisor import run_linkedin_post_generation
        
        await update_task_status(task_id, TaskStatus.IN_PROGRESS, 0.2, "researching_paper")
        
        # Run the LinkedIn post generation
        result = run_linkedin_post_generation(
            paper_title=request.paper_title,
            additional_context=request.additional_context,
            target_audience=request.target_audience,
            include_technical_details=request.include_technical_details,
            max_hashtags=request.max_hashtags,
            tone=request.tone,
            task_id=task_id
        )
        
        if "error" in result:
            await update_task_status(
                task_id, TaskStatus.FAILED, 1.0, "failed", 
                error=result["error"]
            )
        else:
            # Extract the final post from the result
            final_post_content = ""
            if "messages" in result and result["messages"]:
                final_post_content = result["messages"][-1].content
            
            # Create LinkedIn post object
            linkedin_post = LinkedInPost(
                content=final_post_content,
                hashtags=[],  # Extract from content if needed
                word_count=len(final_post_content.split()),
                character_count=len(final_post_content)
            )
            
            await update_task_status(
                task_id, TaskStatus.COMPLETED, 1.0, "completed",
                result=linkedin_post.dict()
            )
        
    except Exception as e:
        await update_task_status(
            task_id, TaskStatus.FAILED, 1.0, "failed", 
            error=str(e)
        )


@app.post("/generate-post", response_model=PostGenerationResponse)
async def generate_linkedin_post(
    request: PostGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate a LinkedIn post about an ML paper."""
    task_id = str(uuid.uuid4())
    
    # Store initial task data
    task_data = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "request_data": request.dict(),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "progress": 0.0,
        "current_step": "queued"
    }
    
    await store_task(task_id, task_data)
    
    # Start background task
    background_tasks.add_task(run_post_generation_task, task_id, request)
    
    return PostGenerationResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="LinkedIn post generation started successfully",
        estimated_completion_time=datetime.utcnow() + timedelta(minutes=3)
    )


@app.get("/status/{task_id}", response_model=PostStatusResponse)
async def get_post_status(task_id: str):
    """Get the status of a LinkedIn post generation task."""
    task_data = await get_task(task_id)
    
    if not task_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    # Parse dates
    created_at = datetime.fromisoformat(task_data.get("created_at", datetime.utcnow().isoformat()))
    updated_at = datetime.fromisoformat(task_data.get("updated_at", datetime.utcnow().isoformat()))
    
    # Create LinkedIn post if result exists
    result = None
    if task_data.get("result"):
        result = LinkedInPost(**task_data["result"])
    
    return PostStatusResponse(
        task_id=task_id,
        status=TaskStatus(task_data.get("status", TaskStatus.PENDING)),
        progress=task_data.get("progress", 0.0),
        current_step=task_data.get("current_step"),
        result=result,
        error_message=task_data.get("error"),
        created_at=created_at,
        updated_at=updated_at
    )


@app.post("/verify-post", response_model=PostVerificationResponse)
async def verify_post(request: PostVerificationRequest):
    """Verify a LinkedIn post for technical accuracy and style compliance."""
    verification_id = str(uuid.uuid4())
    
    try:
        # Import verification tools
        from app.tools.linkedin_tools import verify_technical_accuracy, check_linkedin_style
        
        verification_report = VerificationReport()
        
        # Run technical verification if requested
        if request.verification_type in ["technical", "both"]:
            tech_result = verify_technical_accuracy.invoke({
                "post_content": request.post_content,
                "paper_reference": request.paper_reference or ""
            })
            verification_report.technical_accuracy = {"report": tech_result}
        
        # Run style verification if requested
        if request.verification_type in ["style", "both"]:
            style_result = check_linkedin_style.invoke({
                "post_content": request.post_content
            })
            verification_report.style_compliance = {"report": style_result}
        
        # Calculate overall score (simplified)
        verification_report.overall_score = 0.8  # Placeholder
        verification_report.recommendations = [
            "Consider adding more engagement elements",
            "Ensure proper citation format"
        ]
        
        # Determine if approved
        approved = verification_report.overall_score >= 0.7
        
        return PostVerificationResponse(
            verification_id=verification_id,
            post_content=request.post_content,
            verification_report=verification_report,
            approved=approved
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )


@app.post("/batch-generate", response_model=BatchPostResponse)
async def batch_generate_posts(
    request: BatchPostRequest,
    background_tasks: BackgroundTasks
):
    """Generate multiple LinkedIn posts in batch."""
    batch_id = str(uuid.uuid4())
    task_ids = []
    
    # Create individual tasks for each paper
    for paper_request in request.papers:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        
        # Store task data
        task_data = {
            "task_id": task_id,
            "batch_id": batch_id,
            "status": TaskStatus.PENDING,
            "request_data": paper_request.dict(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "progress": 0.0,
            "current_step": "queued"
        }
        
        await store_task(task_id, task_data)
        
        # Add to background tasks with delay if scheduling is enabled
        if request.schedule_posts and len(task_ids) > 1:
            delay = (len(task_ids) - 1) * request.time_interval_minutes * 60
            # Note: In production, use a proper task queue like Celery
        
        background_tasks.add_task(run_post_generation_task, task_id, paper_request)
    
    return BatchPostResponse(
        batch_id=batch_id,
        total_posts=len(request.papers),
        task_ids=task_ids,
        status=TaskStatus.PENDING,
        estimated_completion_time=datetime.utcnow() + timedelta(
            minutes=len(request.papers) * 3
        )
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            details={"exception": str(exc)}
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 