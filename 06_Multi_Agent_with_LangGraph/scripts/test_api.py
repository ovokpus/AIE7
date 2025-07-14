#!/usr/bin/env python3
"""
Test script for LinkedIn ML Paper Post Generation API

This script runs comprehensive tests on all API endpoints to ensure they're working correctly.
"""

import requests
import time
import json
import sys
from typing import Dict, Any, List
from datetime import datetime


class APITester:
    """Test suite for the LinkedIn ML Paper Post Generation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the tester."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {status} - {test_name}"
        if message:
            log_message += f" - {message}"
        
        print(log_message)
        
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": timestamp
        })
        
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
    
    def test_health_endpoints(self):
        """Test health check endpoints."""
        print("\n🔍 Testing Health Endpoints")
        print("-" * 40)
        
        # Test root endpoint
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("Root endpoint", True, f"Version: {data.get('version')}")
                else:
                    self.log_test("Root endpoint", False, f"Unhealthy status: {data.get('status')}")
            else:
                self.log_test("Root endpoint", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Root endpoint", False, str(e))
        
        # Test health endpoint
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                services = data.get("services", {})
                self.log_test("Health endpoint", True, f"Services: {list(services.keys())}")
                
                # Check individual services
                for service, status in services.items():
                    if status in ["connected", "not_available"]:
                        self.log_test(f"Service {service}", True, f"Status: {status}")
                    else:
                        self.log_test(f"Service {service}", False, f"Status: {status}")
            else:
                self.log_test("Health endpoint", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Health endpoint", False, str(e))
    
    def test_post_generation(self):
        """Test post generation endpoint."""
        print("\n📝 Testing Post Generation")
        print("-" * 40)
        
        # Test valid request
        try:
            payload = {
                "paper_title": "Test Paper: A Simple Example",
                "additional_context": "This is a test",
                "target_audience": "professional",
                "include_technical_details": True,
                "max_hashtags": 5,
                "tone": "professional"
            }
            
            response = self.session.post(f"{self.base_url}/generate-post", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                task_id = data.get("task_id")
                if task_id:
                    self.log_test("Post generation request", True, f"Task ID: {task_id[:8]}...")
                    
                    # Store task_id for status testing
                    self.test_task_id = task_id
                else:
                    self.log_test("Post generation request", False, "No task_id in response")
            else:
                self.log_test("Post generation request", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Post generation request", False, str(e))
        
        # Test invalid request (missing required field)
        try:
            payload = {
                "additional_context": "Missing paper_title"
            }
            
            response = self.session.post(f"{self.base_url}/generate-post", json=payload)
            
            if response.status_code == 422:  # Validation error
                self.log_test("Invalid post request validation", True, "Correctly rejected invalid request")
            else:
                self.log_test("Invalid post request validation", False, f"Expected 422, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Invalid post request validation", False, str(e))
    
    def test_status_endpoint(self):
        """Test status checking endpoint."""
        print("\n📊 Testing Status Endpoint")
        print("-" * 40)
        
        # Test with valid task_id (if we have one)
        if hasattr(self, 'test_task_id'):
            try:
                response = self.session.get(f"{self.base_url}/status/{self.test_task_id}")
                
                if response.status_code == 200:
                    data = response.json()
                    required_fields = ["task_id", "status", "progress", "created_at", "updated_at"]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        self.log_test("Status endpoint structure", True, f"Status: {data.get('status')}")
                    else:
                        self.log_test("Status endpoint structure", False, f"Missing fields: {missing_fields}")
                else:
                    self.log_test("Status endpoint", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test("Status endpoint", False, str(e))
        
        # Test with invalid task_id
        try:
            fake_task_id = "00000000-0000-0000-0000-000000000000"
            response = self.session.get(f"{self.base_url}/status/{fake_task_id}")
            
            if response.status_code == 404:
                self.log_test("Status endpoint 404 handling", True, "Correctly returned 404 for invalid task")
            else:
                self.log_test("Status endpoint 404 handling", False, f"Expected 404, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Status endpoint 404 handling", False, str(e))
    
    def test_verification_endpoint(self):
        """Test post verification endpoint."""
        print("\n🔍 Testing Verification Endpoint")
        print("-" * 40)
        
        # Test valid verification request
        try:
            payload = {
                "post_content": "🚀 Test LinkedIn post about machine learning with #AI #MachineLearning hashtags. What do you think?",
                "paper_reference": "Test paper reference",
                "verification_type": "both"
            }
            
            response = self.session.post(f"{self.base_url}/verify-post", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["verification_id", "post_content", "verification_report", "approved"]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.log_test("Verification endpoint", True, f"Approved: {data.get('approved')}")
                else:
                    self.log_test("Verification endpoint", False, f"Missing fields: {missing_fields}")
            else:
                self.log_test("Verification endpoint", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Verification endpoint", False, str(e))
        
        # Test different verification types
        for verification_type in ["technical", "style"]:
            try:
                payload = {
                    "post_content": "Test content for verification",
                    "verification_type": verification_type
                }
                
                response = self.session.post(f"{self.base_url}/verify-post", json=payload)
                
                if response.status_code == 200:
                    self.log_test(f"Verification type '{verification_type}'", True)
                else:
                    self.log_test(f"Verification type '{verification_type}'", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"Verification type '{verification_type}'", False, str(e))
    
    def test_batch_generation(self):
        """Test batch generation endpoint."""
        print("\n📚 Testing Batch Generation")
        print("-" * 40)
        
        # Test valid batch request
        try:
            payload = {
                "papers": [
                    {
                        "paper_title": "Test Paper 1",
                        "tone": "professional"
                    },
                    {
                        "paper_title": "Test Paper 2", 
                        "tone": "academic",
                        "target_audience": "academic"
                    }
                ],
                "schedule_posts": False,
                "time_interval_minutes": 60
            }
            
            response = self.session.post(f"{self.base_url}/batch-generate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["batch_id", "total_posts", "task_ids", "status"]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.log_test("Batch generation", True, f"Batch ID: {data.get('batch_id', '')[:8]}...")
                else:
                    self.log_test("Batch generation", False, f"Missing fields: {missing_fields}")
            else:
                self.log_test("Batch generation", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Batch generation", False, str(e))
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        print("\n⚠️  Testing Error Handling")
        print("-" * 40)
        
        # Test completely invalid JSON
        try:
            response = self.session.post(
                f"{self.base_url}/generate-post",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 422:
                self.log_test("Invalid JSON handling", True, "Correctly rejected invalid JSON")
            else:
                self.log_test("Invalid JSON handling", False, f"Expected 422, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Invalid JSON handling", False, str(e))
        
        # Test unsupported HTTP method
        try:
            response = self.session.delete(f"{self.base_url}/generate-post")
            
            if response.status_code == 405:  # Method not allowed
                self.log_test("Unsupported HTTP method", True, "Correctly rejected DELETE method")
            else:
                self.log_test("Unsupported HTTP method", False, f"Expected 405, got {response.status_code}")
                
        except Exception as e:
            self.log_test("Unsupported HTTP method", False, str(e))
    
    def test_performance(self):
        """Test basic performance metrics."""
        print("\n⚡ Testing Performance")
        print("-" * 40)
        
        # Test response times for health endpoint
        response_times = []
        
        for i in range(5):
            try:
                start_time = time.time()
                response = self.session.get(f"{self.base_url}/health")
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
                    
            except Exception:
                pass
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            
            if avg_time < 1.0:  # Less than 1 second average
                self.log_test("Health endpoint performance", True, f"Avg: {avg_time:.3f}s, Max: {max_time:.3f}s")
            else:
                self.log_test("Health endpoint performance", False, f"Slow response: {avg_time:.3f}s average")
        else:
            self.log_test("Health endpoint performance", False, "Could not measure response times")
    
    def run_all_tests(self):
        """Run all tests and generate a summary report."""
        print("🚀 LinkedIn ML Paper Post Generation API - Test Suite")
        print("=" * 60)
        print(f"Testing API at: {self.base_url}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test categories
        self.test_health_endpoints()
        self.test_post_generation()
        self.test_status_endpoint()
        self.test_verification_endpoint()
        self.test_batch_generation()
        self.test_error_handling()
        self.test_performance()
        
        # Generate summary
        self.print_summary()
        
        return self.tests_failed == 0
    
    def print_summary(self):
        """Print test summary report."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {self.tests_passed}")
        print(f"❌ Failed: {self.tests_failed}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.tests_failed > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  - {result['test']}: {result['message']}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.tests_failed == 0:
            print("\n🎉 All tests passed! API is working correctly.")
        else:
            print(f"\n⚠️  {self.tests_failed} test(s) failed. Please check the issues above.")


def main():
    """Main function to run the test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LinkedIn ML Paper Post Generation API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="Base URL of the API to test (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--output", 
        help="File to save test results JSON"
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = APITester(args.url)
    success = tester.run_all_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "base_url": args.url,
                "total_tests": tester.tests_passed + tester.tests_failed,
                "tests_passed": tester.tests_passed,
                "tests_failed": tester.tests_failed,
                "success_rate": (tester.tests_passed / (tester.tests_passed + tester.tests_failed)) * 100,
                "results": tester.test_results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 