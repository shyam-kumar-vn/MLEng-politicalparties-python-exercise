#!/usr/bin/env python3
"""
Databricks Model Inference Endpoint Test Script

This script tests the deployed Political Party Classification model endpoint
to ensure it's working correctly and returning expected predictions.
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EndpointTester:
    """Test class for Databricks model serving endpoints"""
    
    def __init__(self, endpoint_url: str, token: str):
        """Initialize the tester with endpoint URL and authentication token"""
        self.endpoint_url = endpoint_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def test_health(self) -> Dict[str, Any]:
        """Test endpoint health and availability"""
        logger.info("Testing endpoint health...")
        
        try:
            # Try to make a simple request to check if endpoint is accessible
            response = self.session.get(self.endpoint_url.replace("/invocations", ""))
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "message": "Endpoint is accessible"
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "message": f"Endpoint returned status code: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to connect to endpoint"
            }
    
    def test_single_prediction(self, text: str) -> Dict[str, Any]:
        """Test single prediction endpoint"""
        logger.info(f"Testing single prediction for text: '{text[:50]}...'")
        
        try:
            data = {
                "dataframe_records": [
                    {"text": text}
                ]
            }
            
            response = self.session.post(self.endpoint_url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('predictions', [None])[0]
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "prediction": prediction,
                    "response_time": response.elapsed.total_seconds(),
                    "full_response": result
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text,
                    "message": f"Prediction failed with status code: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Exception during prediction"
            }
    
    def test_batch_prediction(self, texts: List[str]) -> Dict[str, Any]:
        """Test batch prediction endpoint"""
        logger.info(f"Testing batch prediction for {len(texts)} texts")
        
        try:
            data = {
                "dataframe_records": [
                    {"text": text} for text in texts
                ]
            }
            
            response = self.session.post(self.endpoint_url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "predictions": predictions,
                    "response_time": response.elapsed.total_seconds(),
                    "full_response": result
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text,
                    "message": f"Batch prediction failed with status code: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Exception during batch prediction"
            }
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling"""
        logger.info("Testing edge cases...")
        
        edge_cases = {
            "empty_text": "",
            "very_long_text": "This is a very long text " * 100,
            "special_characters": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            "numbers_only": "1234567890",
            "unicode_text": "Café résumé naïve naïve",
            "html_tags": "<p>This is <b>bold</b> text</p>",
            "newlines": "Line 1\nLine 2\nLine 3"
        }
        
        results = {}
        
        for case_name, text in edge_cases.items():
            logger.info(f"Testing edge case: {case_name}")
            result = self.test_single_prediction(text)
            results[case_name] = result
            
            # Add small delay to avoid overwhelming the endpoint
            time.sleep(0.5)
        
        return results
    
    def test_performance(self, num_requests: int = 10) -> Dict[str, Any]:
        """Test endpoint performance with multiple requests"""
        logger.info(f"Testing performance with {num_requests} requests...")
        
        test_text = "Great economic policies from our party! #politics #economy"
        response_times = []
        successful_requests = 0
        
        for i in range(num_requests):
            logger.info(f"Performance test request {i+1}/{num_requests}")
            
            result = self.test_single_prediction(test_text)
            
            if result["success"]:
                successful_requests += 1
                response_times.append(result["response_time"])
            
            # Add small delay between requests
            time.sleep(0.2)
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        return {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / num_requests,
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "response_times": response_times
        }
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("Starting full test suite...")
        
        # Sample test texts
        sample_texts = [
            "Great economic policies from our party! #politics #economy",
            "We need better healthcare for all citizens",
            "Supporting local businesses and job creation",
            "Environmental protection is our top priority",
            "Strong national security policies",
            "Education reform is essential for our future",
            "Tax cuts will boost the economy",
            "Immigration policy needs reform",
            "Climate change is a serious threat",
            "Infrastructure investment creates jobs"
        ]
        
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "endpoint_url": self.endpoint_url
        }
        
        # Test 1: Health check
        logger.info("="*50)
        logger.info("TEST 1: Health Check")
        logger.info("="*50)
        test_results["health_check"] = self.test_health()
        
        # Test 2: Single predictions
        logger.info("="*50)
        logger.info("TEST 2: Single Predictions")
        logger.info("="*50)
        test_results["single_predictions"] = {}
        
        for i, text in enumerate(sample_texts[:5]):  # Test first 5 texts
            result = self.test_single_prediction(text)
            test_results["single_predictions"][f"text_{i+1}"] = result
            
            if result["success"]:
                logger.info(f"Text {i+1}: '{text[:50]}...' -> {result['prediction']}")
            else:
                logger.error(f"Text {i+1} failed: {result.get('message', 'Unknown error')}")
        
        # Test 3: Batch prediction
        logger.info("="*50)
        logger.info("TEST 3: Batch Prediction")
        logger.info("="*50)
        test_results["batch_prediction"] = self.test_batch_prediction(sample_texts)
        
        if test_results["batch_prediction"]["success"]:
            predictions = test_results["batch_prediction"]["predictions"]
            logger.info(f"Batch predictions: {predictions}")
        else:
            logger.error(f"Batch prediction failed: {test_results['batch_prediction'].get('message', 'Unknown error')}")
        
        # Test 4: Edge cases
        logger.info("="*50)
        logger.info("TEST 4: Edge Cases")
        logger.info("="*50)
        test_results["edge_cases"] = self.test_edge_cases()
        
        # Test 5: Performance
        logger.info("="*50)
        logger.info("TEST 5: Performance Test")
        logger.info("="*50)
        test_results["performance"] = self.test_performance(num_requests=5)
        
        # Calculate summary statistics
        successful_tests = 0
        total_tests = 0
        
        # Count health check
        total_tests += 1
        if test_results["health_check"]["success"]:
            successful_tests += 1
        
        # Count single predictions
        for result in test_results["single_predictions"].values():
            total_tests += 1
            if result["success"]:
                successful_tests += 1
        
        # Count batch prediction
        total_tests += 1
        if test_results["batch_prediction"]["success"]:
            successful_tests += 1
        
        # Count edge cases
        for result in test_results["edge_cases"].values():
            total_tests += 1
            if result["success"]:
                successful_tests += 1
        
        # Count performance test
        total_tests += 1
        if test_results["performance"]["success_rate"] > 0.8:  # 80% success rate threshold
            successful_tests += 1
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests,
            "overall_status": "PASS" if successful_tests / total_tests >= 0.8 else "FAIL"
        }
        
        return test_results

def load_deployment_info() -> Dict[str, Any]:
    """Load deployment information from file"""
    try:
        with open("deployment_info.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("deployment_info.json not found")
        return {}
    except Exception as e:
        logger.error(f"Error loading deployment info: {e}")
        return {}

def get_token_from_env() -> str:
    """Get authentication token from environment variables"""
    token = os.getenv("DATABRICKS_TOKEN")
    if not token:
        logger.error("DATABRICKS_TOKEN environment variable not set")
        sys.exit(1)
    return token

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Databricks model inference endpoint")
    
    parser.add_argument("--endpoint-url", type=str,
                       help="Endpoint URL for testing")
    parser.add_argument("--token", type=str,
                       help="Databricks access token")
    parser.add_argument("--deployment-info", type=str, default="deployment_info.json",
                       help="Path to deployment info file")
    parser.add_argument("--test-mode", type=str, choices=["full", "quick", "performance"],
                       default="full", help="Test mode")
    parser.add_argument("--output-file", type=str, default="test_results.json",
                       help="Output file for test results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get endpoint URL
    endpoint_url = args.endpoint_url
    if not endpoint_url:
        # Try to get from deployment info
        deployment_info = load_deployment_info()
        endpoint_url = deployment_info.get("endpoint_url")
        
        if not endpoint_url:
            logger.error("Endpoint URL not provided and not found in deployment info")
            sys.exit(1)
    
    # Get authentication token
    token = args.token or get_token_from_env()
    
    logger.info(f"Testing endpoint: {endpoint_url}")
    
    try:
        # Create tester and run tests
        tester = EndpointTester(endpoint_url, token)
        
        if args.test_mode == "quick":
            # Quick test with just health check and single prediction
            logger.info("Running quick test...")
            
            results = {
                "test_timestamp": datetime.now().isoformat(),
                "endpoint_url": endpoint_url,
                "test_mode": "quick"
            }
            
            results["health_check"] = tester.test_health()
            results["single_prediction"] = tester.test_single_prediction(
                "Great economic policies from our party!"
            )
            
            # Simple summary
            successful_tests = sum(1 for test in [results["health_check"], results["single_prediction"]] 
                                 if test["success"])
            results["summary"] = {
                "total_tests": 2,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / 2,
                "overall_status": "PASS" if successful_tests == 2 else "FAIL"
            }
            
        elif args.test_mode == "performance":
            # Performance test only
            logger.info("Running performance test...")
            
            results = {
                "test_timestamp": datetime.now().isoformat(),
                "endpoint_url": endpoint_url,
                "test_mode": "performance",
                "performance": tester.test_performance(num_requests=20)
            }
            
            # Performance summary
            perf = results["performance"]
            results["summary"] = {
                "success_rate": perf["success_rate"],
                "avg_response_time": perf["avg_response_time"],
                "overall_status": "PASS" if perf["success_rate"] >= 0.9 else "FAIL"
            }
            
        else:
            # Full test suite
            results = tester.run_full_test_suite()
        
        # Save results
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        summary = results["summary"]
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Endpoint URL: {endpoint_url}")
        print(f"Test Mode: {results.get('test_mode', 'full')}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Overall Status: {summary['overall_status']}")
        
        if "performance" in results:
            perf = results["performance"]
            print(f"Average Response Time: {perf['avg_response_time']:.3f}s")
            print(f"Performance Success Rate: {perf['success_rate']:.2%}")
        
        print("="*60)
        print(f"Detailed results saved to: {args.output_file}")
        
        # Exit with appropriate code
        if summary["overall_status"] == "PASS":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test suite failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 