#!/usr/bin/env python3
"""
Databricks Model Inference Endpoint Deployment Script

This script deploys the Political Party Classification model to a Databricks serving endpoint.
It supports both creation of new endpoints and updates to existing endpoints.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput, 
        ServedModelInput,
        EndpointState
    )
    from databricks.sdk.errors import NotFound, PermissionDenied
except ImportError as e:
    logger.error(f"Required Databricks SDK not found: {e}")
    logger.error("Please install: pip install databricks-sdk")
    sys.exit(1)

class ModelEndpointDeployer:
    """Deployer class for Databricks model serving endpoints"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the deployer with configuration"""
        self.config = config
        self.client = WorkspaceClient()
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate the deployment configuration"""
        required_fields = [
            'catalog_name', 'schema_name', 'model_name', 
            'endpoint_name', 'production_alias'
        ]
        
        for field in required_fields:
            if not self.config.get(field):
                raise ValueError(f"Missing required configuration: {field}")
        
        logger.info("Configuration validated successfully")
        
    def check_model_exists(self) -> bool:
        """Check if the model exists in Unity Catalog"""
        try:
            model_name = f"{self.config['catalog_name']}.{self.config['schema_name']}.{self.config['model_name']}"
            
            # Try to get model versions
            versions = self.client.model_registry.list_model_versions(model_name)
            
            if not versions:
                logger.warning(f"No versions found for model: {model_name}")
                return False
                
            logger.info(f"Model {model_name} exists with {len(versions)} versions")
            return True
            
        except Exception as e:
            logger.error(f"Error checking model existence: {e}")
            return False
    
    def get_latest_model_version(self) -> Optional[str]:
        """Get the latest model version"""
        try:
            model_name = f"{self.config['catalog_name']}.{self.config['schema_name']}.{self.config['model_name']}"
            versions = self.client.model_registry.list_model_versions(model_name)
            
            if not versions:
                return None
                
            # Get the latest version
            latest_version = max(versions, key=lambda v: v.version)
            logger.info(f"Latest model version: {latest_version.version}")
            return str(latest_version.version)
            
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return None
    
    def check_endpoint_exists(self) -> bool:
        """Check if the serving endpoint already exists"""
        try:
            endpoints = self.client.serving_endpoints.list()
            exists = any(ep.name == self.config['endpoint_name'] for ep in endpoints)
            
            if exists:
                logger.info(f"Endpoint {self.config['endpoint_name']} already exists")
            else:
                logger.info(f"Endpoint {self.config['endpoint_name']} does not exist")
                
            return exists
            
        except Exception as e:
            logger.error(f"Error checking endpoint existence: {e}")
            return False
    
    def get_endpoint_status(self) -> Optional[str]:
        """Get the current status of the endpoint"""
        try:
            endpoint = self.client.serving_endpoints.get(self.config['endpoint_name'])
            return endpoint.state.value
            
        except Exception as e:
            logger.error(f"Error getting endpoint status: {e}")
            return None
    
    def create_endpoint(self) -> bool:
        """Create a new serving endpoint"""
        try:
            logger.info(f"Creating new endpoint: {self.config['endpoint_name']}")
            
            # Get latest model version
            model_version = self.get_latest_model_version()
            if not model_version:
                logger.error("No model version found")
                return False
            
            # Create endpoint configuration
            config = EndpointCoreConfigInput(
                served_models=[
                    ServedModelInput(
                        model_name=f"{self.config['catalog_name']}.{self.config['schema_name']}.{self.config['model_name']}",
                        model_version=model_version,
                        workload_size=self.config.get('workload_size', 'Small'),
                        scale_to_zero_enabled=self.config.get('scale_to_zero', True),
                        environment_vars={
                            "CATALOG_NAME": self.config['catalog_name'],
                            "SCHEMA_NAME": self.config['schema_name'],
                            "MODEL_NAME": self.config['model_name'],
                            "PRODUCTION_ALIAS": self.config['production_alias']
                        }
                    )
                ]
            )
            
            # Create the endpoint
            endpoint = self.client.serving_endpoints.create(
                name=self.config['endpoint_name'],
                config=config
            )
            
            logger.info(f"Endpoint created successfully: {endpoint.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating endpoint: {e}")
            return False
    
    def update_endpoint(self) -> bool:
        """Update an existing serving endpoint"""
        try:
            logger.info(f"Updating existing endpoint: {self.config['endpoint_name']}")
            
            # Get latest model version
            model_version = self.get_latest_model_version()
            if not model_version:
                logger.error("No model version found")
                return False
            
            # Update endpoint configuration
            self.client.serving_endpoints.update_config(
                name=self.config['endpoint_name'],
                served_models=[
                    ServedModelInput(
                        model_name=f"{self.config['catalog_name']}.{self.config['schema_name']}.{self.config['model_name']}",
                        model_version=model_version,
                        workload_size=self.config.get('workload_size', 'Small'),
                        scale_to_zero_enabled=self.config.get('scale_to_zero', True),
                        environment_vars={
                            "CATALOG_NAME": self.config['catalog_name'],
                            "SCHEMA_NAME": self.config['schema_name'],
                            "MODEL_NAME": self.config['model_name'],
                            "PRODUCTION_ALIAS": self.config['production_alias']
                        }
                    )
                ]
            )
            
            logger.info(f"Endpoint updated successfully: {self.config['endpoint_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating endpoint: {e}")
            return False
    
    def wait_for_endpoint_ready(self, timeout_minutes: int = 10) -> bool:
        """Wait for the endpoint to be ready"""
        import time
        
        logger.info(f"Waiting for endpoint to be ready (timeout: {timeout_minutes} minutes)")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            status = self.get_endpoint_status()
            
            if status == EndpointState.READY.value:
                logger.info("Endpoint is ready!")
                return True
            elif status == EndpointState.FAILED.value:
                logger.error("Endpoint deployment failed")
                return False
            else:
                logger.info(f"Endpoint status: {status}")
                time.sleep(30)  # Wait 30 seconds before checking again
        
        logger.error(f"Timeout waiting for endpoint to be ready after {timeout_minutes} minutes")
        return False
    
    def get_endpoint_url(self) -> Optional[str]:
        """Get the endpoint URL for API access"""
        try:
            endpoint = self.client.serving_endpoints.get(self.config['endpoint_name'])
            
            if hasattr(endpoint, 'config') and hasattr(endpoint.config, 'inference_endpoint_url'):
                return endpoint.config.inference_endpoint_url
            else:
                # Construct URL manually
                workspace_url = self.client.config.host
                endpoint_name = self.config['endpoint_name']
                return f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
                
        except Exception as e:
            logger.error(f"Error getting endpoint URL: {e}")
            return None
    
    def deploy(self) -> Dict[str, Any]:
        """Main deployment method"""
        logger.info("Starting endpoint deployment...")
        
        # Check if model exists
        if not self.check_model_exists():
            logger.error("Model does not exist. Please train and register the model first.")
            return {"success": False, "error": "Model not found"}
        
        # Check if endpoint exists
        endpoint_exists = self.check_endpoint_exists()
        
        # Deploy or update endpoint
        if endpoint_exists:
            success = self.update_endpoint()
        else:
            success = self.create_endpoint()
        
        if not success:
            return {"success": False, "error": "Endpoint deployment failed"}
        
        # Wait for endpoint to be ready
        if not self.wait_for_endpoint_ready():
            return {"success": False, "error": "Endpoint not ready within timeout"}
        
        # Get endpoint URL
        endpoint_url = self.get_endpoint_url()
        
        # Prepare deployment info
        deployment_info = {
            "success": True,
            "endpoint_name": self.config['endpoint_name'],
            "endpoint_url": endpoint_url,
            "model_uri": f"models:/{self.config['catalog_name']}.{self.config['schema_name']}.{self.config['model_name']}@{self.config['production_alias']}",
            "deployment_time": datetime.now().isoformat(),
            "configuration": self.config
        }
        
        # Save deployment info
        with open("deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info("Deployment completed successfully!")
        logger.info(f"Endpoint URL: {endpoint_url}")
        
        return deployment_info

def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}

def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        "catalog_name": "mle_batch_catalog_2025_q2",
        "schema_name": "mle_shyamkumar_vn",
        "model_name": "political_party_classifier",
        "endpoint_name": "political-party-classifier-endpoint",
        "production_alias": "production",
        "workload_size": "Small",
        "scale_to_zero": True
    }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy Databricks model inference endpoint")
    
    parser.add_argument("--config-file", type=str,
                       help="Path to configuration JSON file")
    parser.add_argument("--catalog-name", type=str,
                       help="Unity Catalog name")
    parser.add_argument("--schema-name", type=str,
                       help="Unity Catalog schema name")
    parser.add_argument("--model-name", type=str,
                       help="Model name")
    parser.add_argument("--endpoint-name", type=str,
                       help="Serving endpoint name")
    parser.add_argument("--production-alias", type=str,
                       help="Production model alias")
    parser.add_argument("--workload-size", type=str, choices=["Small", "Medium", "Large"],
                       help="Workload size for the endpoint")
    parser.add_argument("--scale-to-zero", action="store_true",
                       help="Enable scale to zero")
    parser.add_argument("--timeout", type=int, default=10,
                       help="Timeout in minutes for endpoint to be ready")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config_file:
        config = load_config_from_file(args.config_file)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    if args.catalog_name:
        config['catalog_name'] = args.catalog_name
    if args.schema_name:
        config['schema_name'] = args.schema_name
    if args.model_name:
        config['model_name'] = args.model_name
    if args.endpoint_name:
        config['endpoint_name'] = args.endpoint_name
    if args.production_alias:
        config['production_alias'] = args.production_alias
    if args.workload_size:
        config['workload_size'] = args.workload_size
    if args.scale_to_zero:
        config['scale_to_zero'] = True
    
    # Print configuration
    logger.info("Deployment Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Create deployer and deploy
        deployer = ModelEndpointDeployer(config)
        result = deployer.deploy()
        
        if result["success"]:
            print("\n" + "="*60)
            print("DEPLOYMENT SUCCESSFUL!")
            print("="*60)
            print(f"Endpoint Name: {result['endpoint_name']}")
            print(f"Endpoint URL: {result['endpoint_url']}")
            print(f"Model URI: {result['model_uri']}")
            print(f"Deployment Time: {result['deployment_time']}")
            print("="*60)
            
            # Print endpoint URL for workflow integration
            print(f"ENDPOINT_URL={result['endpoint_url']}")
            
            sys.exit(0)
        else:
            print(f"\nDeployment failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Deployment failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 