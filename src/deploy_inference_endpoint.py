#!/usr/bin/env python3
"""
Deployment script for the Political Party Classification Model Inference Endpoint
This script deploys the model inference endpoint to Databricks serving endpoints.
"""

import os
import json
import logging
import argparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy model inference endpoint")
    parser.add_argument("--catalog-name", default="mle_batch_catalog_2025_q2",
                       help="Unity Catalog name")
    parser.add_argument("--schema-name", default="mle_shyamkumar_vn",
                       help="Unity Catalog schema name")
    parser.add_argument("--model-name", default="political_party_classifier",
                       help="Model name in Unity Catalog")
    parser.add_argument("--production-alias", default="production",
                       help="Production model alias")
    parser.add_argument("--endpoint-name", default="political-party-classifier-endpoint",
                       help="Serving endpoint name")
    parser.add_argument("--workload-size", default="Small",
                       choices=["Small", "Medium", "Large"],
                       help="Workload size for the endpoint")
    parser.add_argument("--scale-to-zero", action="store_true",
                       help="Enable scale to zero")
    
    return parser.parse_args()

def create_serving_endpoint(catalog_name, schema_name, model_name, 
                          production_alias, endpoint_name, workload_size="Small", 
                          scale_to_zero=True):
    """Create or update the serving endpoint"""
    
    # Initialize Databricks client
    client = WorkspaceClient()
    
    try:
        # Check if endpoint already exists
        existing_endpoints = client.serving_endpoints.list()
        endpoint_exists = any(ep.name == endpoint_name for ep in existing_endpoints)
        
        if endpoint_exists:
            logger.info(f"Updating existing endpoint: {endpoint_name}")
            # Update existing endpoint
            client.serving_endpoints.update_config(
                name=endpoint_name,
                served_models=[
                    ServedModelInput(
                        model_name=f"{catalog_name}.{schema_name}.{model_name}",
                        model_version="latest",
                        workload_size=workload_size,
                        scale_to_zero_enabled=scale_to_zero,
                        environment_vars={
                            "CATALOG_NAME": catalog_name,
                            "SCHEMA_NAME": schema_name,
                            "MODEL_NAME": model_name,
                            "PRODUCTION_ALIAS": production_alias
                        }
                    )
                ]
            )
        else:
            logger.info(f"Creating new endpoint: {endpoint_name}")
            # Create new endpoint
            client.serving_endpoints.create(
                name=endpoint_name,
                config=EndpointCoreConfigInput(
                    served_models=[
                        ServedModelInput(
                            model_name=f"{catalog_name}.{schema_name}.{model_name}",
                            model_version="latest",
                            workload_size=workload_size,
                            scale_to_zero_enabled=scale_to_zero,
                            environment_vars={
                                "CATALOG_NAME": catalog_name,
                                "SCHEMA_NAME": schema_name,
                                "MODEL_NAME": model_name,
                                "PRODUCTION_ALIAS": production_alias
                            }
                        )
                    ]
                )
            )
        
        logger.info(f"Endpoint {endpoint_name} deployed successfully!")
        
        # Get endpoint details
        endpoint = client.serving_endpoints.get(name=endpoint_name)
        logger.info(f"Endpoint URL: {endpoint.config.inference_endpoint_url}")
        
        return endpoint
        
    except Exception as e:
        logger.error(f"Error deploying endpoint: {e}")
        raise

def deploy_custom_endpoint(catalog_name, schema_name, model_name, 
                          production_alias, endpoint_name):
    """Deploy the custom FastAPI endpoint using Databricks Container Services"""
    
    # Initialize Databricks client
    client = WorkspaceClient()
    
    try:
        # This would require setting up a custom container service
        # For now, we'll create a simple deployment configuration
        
        deployment_config = {
            "name": f"{endpoint_name}-custom",
            "model_name": f"{catalog_name}.{schema_name}.{model_name}",
            "model_version": "latest",
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
            "environment_vars": {
                "CATALOG_NAME": catalog_name,
                "SCHEMA_NAME": schema_name,
                "MODEL_NAME": model_name,
                "PRODUCTION_ALIAS": production_alias
            }
        }
        
        logger.info("Custom endpoint deployment configuration:")
        logger.info(json.dumps(deployment_config, indent=2))
        
        # Note: This would require additional setup for custom container deployment
        # The actual implementation depends on your Databricks workspace configuration
        
        return deployment_config
        
    except Exception as e:
        logger.error(f"Error deploying custom endpoint: {e}")
        raise

def main():
    """Main deployment function"""
    
    # Parse arguments
    args = parse_arguments()
    
    logger.info("Starting model inference endpoint deployment...")
    logger.info(f"Catalog: {args.catalog_name}")
    logger.info(f"Schema: {args.schema_name}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Endpoint: {args.endpoint_name}")
    logger.info(f"Workload size: {args.workload_size}")
    logger.info(f"Scale to zero: {args.scale_to_zero}")
    
    try:
        # Deploy the standard serving endpoint
        endpoint = create_serving_endpoint(
            catalog_name=args.catalog_name,
            schema_name=args.schema_name,
            model_name=args.model_name,
            production_alias=args.production_alias,
            endpoint_name=args.endpoint_name,
            workload_size=args.workload_size,
            scale_to_zero=args.scale_to_zero
        )
        
        logger.info("Deployment completed successfully!")
        logger.info(f"Endpoint name: {endpoint.name}")
        logger.info(f"Endpoint state: {endpoint.state}")
        
        if hasattr(endpoint, 'config') and hasattr(endpoint.config, 'inference_endpoint_url'):
            logger.info(f"Endpoint URL: {endpoint.config.inference_endpoint_url}")
        
        # Also prepare custom endpoint configuration
        custom_config = deploy_custom_endpoint(
            catalog_name=args.catalog_name,
            schema_name=args.schema_name,
            model_name=args.model_name,
            production_alias=args.production_alias,
            endpoint_name=args.endpoint_name
        )
        
        # Save deployment info
        deployment_info = {
            "endpoint_name": args.endpoint_name,
            "model_uri": f"models:/{args.catalog_name}.{args.schema_name}.{args.model_name}@{args.production_alias}",
            "deployment_time": str(pd.Timestamp.now()),
            "custom_config": custom_config,
            "endpoint_url": getattr(endpoint.config, 'inference_endpoint_url', None)
        }
        
        with open("deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info("Deployment information saved to deployment_info.json")
        
        # Print endpoint URL for workflow integration
        if hasattr(endpoint, 'config') and hasattr(endpoint.config, 'inference_endpoint_url'):
            print(f"ENDPOINT_URL={endpoint.config.inference_endpoint_url}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

if __name__ == "__main__":
    import pandas as pd
    main() 