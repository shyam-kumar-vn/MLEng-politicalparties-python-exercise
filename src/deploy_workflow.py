#!/usr/bin/env python3
"""
Databricks Workflow Deployment Script

This script deploys the training workflow to Databricks.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def check_databricks_cli():
    """Check if Databricks CLI is installed and configured."""
    try:
        result = subprocess.run(["databricks", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Databricks CLI is installed")
            return True
        else:
            print("❌ Databricks CLI is not installed")
            return False
    except FileNotFoundError:
        print("❌ Databricks CLI is not installed")
        return False

def deploy_workflow(workflow_config_path, workspace_url, cluster_id):
    """Deploy the workflow to Databricks."""
    
    # Read workflow configuration
    with open(workflow_config_path, 'r') as f:
        workflow_config = json.load(f)
    
    # Replace cluster placeholder with actual cluster ID
    workflow_json = json.dumps(workflow_config).replace("{{cluster_id}}", cluster_id)
    
    # Create temporary workflow file
    temp_workflow_path = "temp_workflow.json"
    with open(temp_workflow_path, 'w') as f:
        f.write(workflow_json)
    
    try:
        # Deploy workflow using Databricks CLI
        command = f"databricks jobs create --json-file {temp_workflow_path}"
        result = run_command(command, "Deploying workflow to Databricks")
        
        # Extract job ID from response
        if "job_id" in result:
            job_id = result.split("job_id")[1].split(",")[0].strip().replace('"', '').replace(':', '')
            print(f"✅ Workflow deployed successfully with Job ID: {job_id}")
            return job_id
        else:
            print("⚠️  Workflow deployed but couldn't extract Job ID")
            return None
            
    finally:
        # Clean up temporary file
        if os.path.exists(temp_workflow_path):
            os.remove(temp_workflow_path)

def upload_notebooks(workspace_path, notebooks_dir):
    """Upload notebooks to Databricks workspace."""
    
    notebooks_path = Path(notebooks_dir)
    
    for notebook_file in notebooks_path.glob("*.py"):
        # Convert notebook path to Databricks path
        db_path = f"{workspace_path}/{notebook_file.stem}"
        
        # Upload notebook
        command = f"databricks workspace import --language PYTHON --overwrite {notebook_file} {db_path}"
        run_command(command, f"Uploading notebook: {notebook_file.name}")

def upload_src_directory(workspace_path, src_dir):
    """Upload the src directory to Databricks workspace."""
    
    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    # Upload the entire src directory
    db_src_path = f"{workspace_path}/../src"
    command = f"databricks workspace import --language PYTHON --overwrite {src_dir} {db_src_path}"
    run_command(command, f"Uploading src directory: {src_dir}")
    
    # Also upload individual Python files to ensure they're accessible
    for py_file in src_path.rglob("*.py"):
        if py_file.is_file():
            # Calculate relative path from src
            rel_path = py_file.relative_to(src_path)
            db_file_path = f"{db_src_path}/{rel_path.parent}/{rel_path.stem}"
            
            # Create directory structure if needed
            if rel_path.parent != Path('.'):
                db_dir_path = f"{db_src_path}/{rel_path.parent}"
                mkdir_command = f"databricks workspace mkdirs {db_dir_path}"
                try:
                    subprocess.run(mkdir_command, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    # Directory might already exist, continue
                    pass
            
            # Upload file
            command = f"databricks workspace import --language PYTHON --overwrite {py_file} {db_file_path}"
            run_command(command, f"Uploading source file: {rel_path}")

def main():
    """Main deployment function."""
    
    print("🚀 Databricks Workflow Deployment")
    print("=" * 50)
    
    # Configuration
    WORKFLOW_CONFIG_PATH = "databricks/workflows/training_workflow.json"
    NOTEBOOKS_DIR = "databricks/notebooks"
    SRC_DIR = "src"
    WORKSPACE_PATH = "/Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/databricks/notebooks"
    
    # Get configuration from environment or user input
    workspace_url = os.getenv("DATABRICKS_HOST", input("Enter Databricks workspace URL: "))
    cluster_id = os.getenv("DATABRICKS_CLUSTER_ID", input("Enter Databricks cluster ID: "))
    
    # Check prerequisites
    if not check_databricks_cli():
        print("Please install Databricks CLI first:")
        print("pip install databricks-cli")
        print("databricks configure --token")
        sys.exit(1)
    
    # Check if files exist
    if not os.path.exists(WORKFLOW_CONFIG_PATH):
        print(f"❌ Workflow config not found: {WORKFLOW_CONFIG_PATH}")
        sys.exit(1)
    
    if not os.path.exists(NOTEBOOKS_DIR):
        print(f"❌ Notebooks directory not found: {NOTEBOOKS_DIR}")
        sys.exit(1)
    
    if not os.path.exists(SRC_DIR):
        print(f"❌ Source directory not found: {SRC_DIR}")
        sys.exit(1)
    
    print("\n📁 Uploading source code...")
    upload_src_directory(WORKSPACE_PATH, SRC_DIR)
    
    print("\n📁 Uploading notebooks...")
    upload_notebooks(WORKSPACE_PATH, NOTEBOOKS_DIR)
    
    print("\n🔄 Deploying workflow...")
    job_id = deploy_workflow(WORKFLOW_CONFIG_PATH, workspace_url, cluster_id)
    
    if job_id:
        print(f"\n✅ Deployment completed successfully!")
        print(f"Job ID: {job_id}")
        print(f"View workflow at: {workspace_url}/#job/{job_id}")
        print(f"\n📝 Next steps:")
        print(f"1. Upload your data: databricks fs cp data/Tweets.csv dbfs:/FileStore/tables/Tweets.csv")
        print(f"2. Run the workflow manually or schedule it")
        print(f"3. Monitor the training process in the Databricks Jobs UI")
    else:
        print("\n⚠️  Deployment completed but couldn't get Job ID")

if __name__ == "__main__":
    main() 