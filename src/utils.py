"""
Common utilities for Databricks notebooks.
"""

def get_widget_value(widget_name, default_value):
    """
    Safely get widget value with fallback to default.
    
    This function handles cases where Databricks widgets are not defined
    when running notebooks directly vs. as part of a workflow.
    
    Args:
        widget_name (str): Name of the widget to retrieve
        default_value: Default value to return if widget is not defined
        
    Returns:
        The widget value if defined, otherwise the default value
    """
    try:
        value = dbutils.widgets.get(widget_name)
        return value if value else default_value
    except:
        return default_value


def get_common_parameters():
    """
    Get common parameters used across all notebooks.
    
    Returns:
        dict: Dictionary containing common parameters
    """
    return {
        "catalog_name": get_widget_value("catalog_name", "mle_batch_catalog_2025_q2"),
        "schema_name": get_widget_value("schema_name", "mle_shyamkumar_vn"),
        "dbfs_base_path": get_widget_value("dbfs_base_path", "/dbfs/FileStore/shyamkumar.vn")
    }


def setup_notebook_environment():
    """
    Setup common environment for notebooks.
    
    This function sets up the Python path and imports common libraries.
    """
    import sys
    import os
    
    # Add src directory to Python path
    src_path = '/Workspace/Users/shyamkumar.vn@thoughtworks.com/MLEng-politicalparties-python-exercise-fork/src'
    if src_path not in sys.path:
        sys.path.append(src_path)
    
    print("Notebook environment setup complete")


def get_table_name(catalog_name, schema_name, table_name):
    """
    Generate a fully qualified table name.
    
    Args:
        catalog_name (str): Catalog name
        schema_name (str): Schema name
        table_name (str): Table name
        
    Returns:
        str: Fully qualified table name
    """
    return f"{catalog_name}.{schema_name}.{table_name}"


def get_model_uri(catalog_name, schema_name, model_name, version="latest"):
    """
    Generate a model URI for Unity Catalog.
    
    Args:
        catalog_name (str): Catalog name
        schema_name (str): Schema name
        model_name (str): Model name
        version (str): Model version (default: "latest")
        
    Returns:
        str: Model URI
    """
    return f"models:/{catalog_name}.{schema_name}.{model_name}/{version}"


def setup_mlflow_experiment(experiment_name):
    """
    Safely set up MLflow experiment, creating it if it doesn't exist.
    
    Args:
        experiment_name (str): Name of the experiment to set up
        
    Returns:
        str: The experiment name that was set
    """
    try:
        mlflow.set_experiment(experiment_name)
        print(f"Using existing experiment: {experiment_name}")
        return experiment_name
    except Exception as e:
        print(f"Experiment {experiment_name} not found. Creating new experiment...")
        try:
            # Create the experiment
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(f"Created and set experiment: {experiment_name}")
            return experiment_name
        except Exception as create_error:
            print(f"Failed to create experiment {experiment_name}: {create_error}")
            print("Using default experiment...")
            # Fall back to default experiment
            mlflow.set_experiment(None)
            return None


def print_parameters(params_dict):
    """
    Print parameters in a formatted way.
    
    Args:
        params_dict (dict): Dictionary of parameters to print
    """
    print("=" * 50)
    print("NOTEBOOK PARAMETERS")
    print("=" * 50)
    for key, value in params_dict.items():
        print(f"{key}: {value}")
    print("=" * 50) 