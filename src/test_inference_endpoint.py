#!/usr/bin/env python3
"""
Test script for the Political Party Classification Model Inference Endpoint
This script tests the endpoint functionality locally and remotely.
"""

import requests
import json
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceEndpointTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> Dict[str, Any]:
        """Test the health endpoint"""
        logger.info("Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Health check result: {result}")
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_model_info(self) -> Dict[str, Any]:
        """Test the model info endpoint"""
        logger.info("Testing model info endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/model-info")
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Model info result: {result}")
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Model info check failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_single_prediction(self, text: str) -> Dict[str, Any]:
        """Test single prediction endpoint"""
        logger.info(f"Testing single prediction for text: '{text[:50]}...'")
        
        try:
            payload = {"text": text}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Prediction result: {result}")
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_batch_prediction(self, texts: list) -> Dict[str, Any]:
        """Test batch prediction endpoint"""
        logger.info(f"Testing batch prediction for {len(texts)} texts")
        
        try:
            payload = {"texts": texts}
            response = self.session.post(
                f"{self.base_url}/predict-batch",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Batch prediction result: {result}")
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_reload_model(self) -> Dict[str, Any]:
        """Test model reload endpoint"""
        logger.info("Testing model reload endpoint...")
        
        try:
            response = self.session.post(f"{self.base_url}/reload-model")
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Model reload result: {result}")
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("Starting full test suite...")
        
        test_results = {}
        
        # Test sample texts
        sample_texts = [
            "Great economic policies from our party! #politics #economy",
            "We need better healthcare for all citizens",
            "Supporting local businesses and job creation",
            "Environmental protection is our top priority",
            "Strong national security policies"
        ]
        
        # Test health endpoint
        test_results["health"] = self.test_health()
        
        # Test model info endpoint
        test_results["model_info"] = self.test_model_info()
        
        # Test single predictions
        test_results["single_predictions"] = {}
        for i, text in enumerate(sample_texts):
            test_results["single_predictions"][f"text_{i+1}"] = self.test_single_prediction(text)
        
        # Test batch prediction
        test_results["batch_prediction"] = self.test_batch_prediction(sample_texts)
        
        # Test model reload
        test_results["model_reload"] = self.test_reload_model()
        
        # Summary
        successful_tests = sum(1 for result in test_results.values() 
                             if isinstance(result, dict) and result.get("success", False))
        total_tests = len(test_results)
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": f"{successful_tests/total_tests*100:.1f}%"
        }
        
        logger.info(f"Test suite completed. Success rate: {test_results['summary']['success_rate']}")
        
        return test_results

def main():
    """Main test function"""
    
    # Test local endpoint
    logger.info("Testing local inference endpoint...")
    local_tester = InferenceEndpointTester("http://localhost:8000")
    
    try:
        results = local_tester.run_full_test_suite()
        
        # Save results
        with open("inference_endpoint_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Test results saved to inference_endpoint_test_results.json")
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Total tests: {results['summary']['total_tests']}")
        print(f"Successful tests: {results['summary']['successful_tests']}")
        print(f"Success rate: {results['summary']['success_rate']}")
        print("="*50)
        
        # Test remote endpoint if available
        remote_url = "https://your-databricks-endpoint-url"  # Replace with actual URL
        if remote_url != "https://your-databricks-endpoint-url":
            logger.info("Testing remote inference endpoint...")
            remote_tester = InferenceEndpointTester(remote_url)
            remote_results = remote_tester.run_full_test_suite()
            
            with open("remote_inference_endpoint_test_results.json", "w") as f:
                json.dump(remote_results, f, indent=2)
            
            logger.info("Remote test results saved to remote_inference_endpoint_test_results.json")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main() 