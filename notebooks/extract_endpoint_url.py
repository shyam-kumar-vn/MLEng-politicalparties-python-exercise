# Databricks notebook source
# MAGIC %md
# MAGIC # Extract Endpoint URL from Deployment
# MAGIC 
# MAGIC This notebook extracts the endpoint URL from the deployment output and makes it available for downstream workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Import required libraries
import json
import os
from src.utils import get_widget_value

# Get parameters from workflow
DEPLOYMENT_OUTPUT_FILE = get_widget_value("deployment_output_file", "deployment_info.json")
ENDPOINT_NAME = get_widget_value("endpoint_name", "political-party-classifier-endpoint")

print(f"Looking for deployment output in: {DEPLOYMENT_OUTPUT_FILE}")
print(f"Endpoint name: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Endpoint URL

# COMMAND ----------

# DBTITLE 1,Read deployment output and extract endpoint URL
try:
    # Read the deployment info file
    with open(DEPLOYMENT_OUTPUT_FILE, 'r') as f:
        deployment_info = json.load(f)
    
    print("Deployment info loaded successfully:")
    print(json.dumps(deployment_info, indent=2))
    
    # Extract endpoint URL
    endpoint_url = deployment_info.get('endpoint_url')
    
    if endpoint_url:
        print(f"✅ Endpoint URL extracted: {endpoint_url}")
        
        # Save endpoint URL to a file for workflow consumption
        endpoint_url_file = "endpoint_url.txt"
        with open(endpoint_url_file, 'w') as f:
            f.write(endpoint_url)
        
        print(f"Endpoint URL saved to: {endpoint_url_file}")
        
        # Also print in a format that can be captured by the workflow
        print(f"ENDPOINT_URL={endpoint_url}")
        
    else:
        print("❌ No endpoint URL found in deployment info")
        print("Available keys:", list(deployment_info.keys()))
        
        # Try to construct the URL manually
        workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().get("api_url")
        if workspace_url:
            manual_url = f"{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"
            print(f"Constructed endpoint URL: {manual_url}")
            print(f"ENDPOINT_URL={manual_url}")
        else:
            print("❌ Could not construct endpoint URL - workspace URL not available")
            
except FileNotFoundError:
    print(f"❌ Deployment output file not found: {DEPLOYMENT_OUTPUT_FILE}")
    
    # Try to construct the URL manually as fallback
    try:
        workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().get("api_url")
        if workspace_url:
            manual_url = f"{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"
            print(f"Using fallback endpoint URL: {manual_url}")
            print(f"ENDPOINT_URL={manual_url}")
        else:
            print("❌ Could not construct fallback endpoint URL")
    except Exception as e:
        print(f"❌ Error constructing fallback URL: {e}")
        
except Exception as e:
    print(f"❌ Error reading deployment output: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Endpoint Status

# COMMAND ----------

# DBTITLE 1,Verify endpoint is ready for inference
if 'endpoint_url' in locals() and endpoint_url:
    try:
        import requests
        
        # Get authentication token
        token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Test endpoint with a simple request
        test_data = {
            "dataframe_records": [
                {"text": "This is a test tweet for endpoint verification"}
            ]
        }
        
        print("Testing endpoint connectivity...")
        response = requests.post(endpoint_url, headers=headers, json=test_data, timeout=30)
        
        if response.status_code == 200:
            print("✅ Endpoint is ready and responding correctly")
            result = response.json()
            print(f"Test prediction: {result.get('predictions', ['No predictions'])}")
        else:
            print(f"⚠️ Endpoint responded with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"⚠️ Could not verify endpoint status: {e}")
        print("Endpoint URL extracted but verification failed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

# DBTITLE 1,Display extraction summary
print("="*60)
print("ENDPOINT URL EXTRACTION SUMMARY")
print("="*60)
print(f"Deployment file: {DEPLOYMENT_OUTPUT_FILE}")
print(f"Endpoint name: {ENDPOINT_NAME}")

if 'endpoint_url' in locals() and endpoint_url:
    print(f"✅ Endpoint URL: {endpoint_url}")
    print("✅ Ready for batch inference workflow")
else:
    print("❌ Endpoint URL extraction failed")
    print("⚠️ Batch inference may use auto-generated URL")

print("="*60) 