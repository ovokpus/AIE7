#!/usr/bin/env python3
"""
Example client for LinkedIn ML Paper Post Generation API

This script demonstrates how to interact with the API to generate LinkedIn posts
about machine learning papers.
"""

import requests
import time
import json
from typing import Dict, Any, Optional


class LinkedInPostClient:
    """Client for interacting with the LinkedIn ML Paper Post Generation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def generate_post(
        self,
        paper_title: str,
        additional_context: Optional[str] = None,
        target_audience: str = "professional",
        include_technical_details: bool = True,
        max_hashtags: int = 10,
        tone: str = "professional"
    ) -> Dict[str, Any]:
        """
        Start generation of a LinkedIn post about an ML paper.
        
        Args:
            paper_title: Title of the ML paper
            additional_context: Additional context or focus areas
            target_audience: Target audience (professional, academic, general)
            include_technical_details: Whether to include technical details
            max_hashtags: Maximum number of hashtags
            tone: Tone of the post (professional, casual, academic)
            
        Returns:
            Response containing task_id and status
        """
        payload = {
            "paper_title": paper_title,
            "additional_context": additional_context,
            "target_audience": target_audience,
            "include_technical_details": include_technical_details,
            "max_hashtags": max_hashtags,
            "tone": tone
        }
        
        response = self.session.post(f"{self.base_url}/generate-post", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a post generation task.
        
        Args:
            task_id: Task ID returned from generate_post
            
        Returns:
            Task status and result if completed
        """
        response = self.session.get(f"{self.base_url}/status/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, task_id: str, timeout: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check status in seconds
            
        Returns:
            Final task status
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_status(task_id)
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                return status
            
            print(f"Status: {status['status']} - Progress: {status['progress']:.1%} - Step: {status.get('current_step', 'unknown')}")
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    def generate_and_wait(self, paper_title: str, **kwargs) -> str:
        """
        Generate a LinkedIn post and wait for completion.
        
        Args:
            paper_title: Title of the ML paper
            **kwargs: Additional arguments for generate_post
            
        Returns:
            Generated LinkedIn post content
        """
        print(f"🚀 Starting generation for paper: '{paper_title}'")
        
        # Start generation
        response = self.generate_post(paper_title, **kwargs)
        task_id = response["task_id"]
        
        print(f"📋 Task ID: {task_id}")
        print(f"⏱️  Estimated completion: {response.get('estimated_completion_time', 'Unknown')}")
        
        # Wait for completion
        final_status = self.wait_for_completion(task_id)
        
        if final_status["status"] == "completed":
            print("✅ Generation completed successfully!")
            return final_status["result"]["content"]
        else:
            error_msg = final_status.get("error_message", "Unknown error")
            raise RuntimeError(f"Generation failed: {error_msg}")
    
    def verify_post(
        self,
        post_content: str,
        paper_reference: Optional[str] = None,
        verification_type: str = "both"
    ) -> Dict[str, Any]:
        """
        Verify a LinkedIn post for technical accuracy and style compliance.
        
        Args:
            post_content: The LinkedIn post content to verify
            paper_reference: Reference information about the source paper
            verification_type: Type of verification ('technical', 'style', 'both')
            
        Returns:
            Verification results
        """
        payload = {
            "post_content": post_content,
            "paper_reference": paper_reference,
            "verification_type": verification_type
        }
        
        response = self.session.post(f"{self.base_url}/verify-post", json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_generate(self, papers: list, schedule_posts: bool = False, time_interval_minutes: int = 60) -> Dict[str, Any]:
        """
        Generate multiple LinkedIn posts in batch.
        
        Args:
            papers: List of paper configurations
            schedule_posts: Whether to schedule posts
            time_interval_minutes: Time interval between posts
            
        Returns:
            Batch generation response
        """
        payload = {
            "papers": papers,
            "schedule_posts": schedule_posts,
            "time_interval_minutes": time_interval_minutes
        }
        
        response = self.session.post(f"{self.base_url}/batch-generate", json=payload)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the LinkedIn Post Client."""
    
    # Initialize client
    client = LinkedInPostClient()
    
    # Check API health
    print("🔍 Checking API health...")
    try:
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"📊 Services: {health['services']}")
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return
    
    # Example 1: Generate a single post
    print("\n" + "="*60)
    print("📝 Example 1: Generate a single LinkedIn post")
    print("="*60)
    
    try:
        post_content = client.generate_and_wait(
            paper_title="Attention Is All You Need",
            additional_context="Focus on practical NLP applications and transformer architecture",
            target_audience="professional",
            tone="professional"
        )
        
        print("\n🎉 Generated LinkedIn Post:")
        print("-" * 40)
        print(post_content)
        print("-" * 40)
        
        # Verify the generated post
        print("\n🔍 Verifying post quality...")
        verification = client.verify_post(
            post_content=post_content,
            paper_reference="Attention Is All You Need - Transformer architecture paper",
            verification_type="both"
        )
        
        print(f"✅ Verification Status: {'APPROVED' if verification['approved'] else 'NEEDS IMPROVEMENT'}")
        print(f"📊 Overall Score: {verification['verification_report'].get('overall_score', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error generating post: {e}")
    
    # Example 2: Batch generation
    print("\n" + "="*60)
    print("📚 Example 2: Batch generate multiple posts")
    print("="*60)
    
    try:
        papers = [
            {
                "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "tone": "professional",
                "target_audience": "professional"
            },
            {
                "paper_title": "ResNet: Deep Residual Learning for Image Recognition",
                "tone": "academic",
                "target_audience": "academic",
                "additional_context": "Focus on computer vision applications"
            }
        ]
        
        batch_response = client.batch_generate(papers, schedule_posts=False)
        
        print(f"📋 Batch ID: {batch_response['batch_id']}")
        print(f"📊 Total Posts: {batch_response['total_posts']}")
        print(f"🔗 Task IDs: {batch_response['task_ids']}")
        
        # Check status of first task
        if batch_response['task_ids']:
            first_task_id = batch_response['task_ids'][0]
            print(f"\n🔍 Checking status of first task: {first_task_id}")
            
            status = client.get_status(first_task_id)
            print(f"📈 Status: {status['status']}")
            print(f"⏱️  Progress: {status['progress']:.1%}")
        
    except Exception as e:
        print(f"❌ Error with batch generation: {e}")
    
    print("\n🎉 Example completed! Check the API documentation at http://localhost:8000/docs for more details.")


if __name__ == "__main__":
    main() 