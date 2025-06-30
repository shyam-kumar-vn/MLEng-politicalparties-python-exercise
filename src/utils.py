"""
Common utilities for Databricks notebooks.
"""
import mlflow
mlflow.set_registry_uri("databricks-uc")
def create_widget_if_not_exists(widget_name, widget_type="text", default_value=""):
    """
    Create a Databricks widget if it doesn't already exist.
    
    This is sometimes necessary in workflows to ensure parameters are accessible.
    
    Args:
        widget_name (str): Name of the widget to create
        widget_type (str): Type of widget ("text", "dropdown", "multiselect", "combobox")
        default_value: Default value for the widget
    """
    try:
        # Try to get the widget value - if it fails, the widget doesn't exist
        dbutils.widgets.get(widget_name)
    except:
        # Widget doesn't exist, create it
        try:
            dbutils.widgets.text(widget_name, default_value)
            print(f"Created widget '{widget_name}' with default value '{default_value}'")
        except Exception as e:
            print(f"Warning: Could not create widget '{widget_name}': {e}")


def get_widget_value(widget_name, default_value):
    """
    Safely get widget value with fallback to default.
    
    This function handles cases where Databricks widgets are not defined
    when running notebooks directly vs. as part of a workflow.
    
    In Databricks workflows, parameters are passed via base_parameters and
    are accessible using dbutils.widgets.get() just like regular widgets.
    
    Args:
        widget_name (str): Name of the widget/parameter to retrieve
        default_value: Default value to return if widget/parameter is not defined
        
    Returns:
        The widget/parameter value if defined, otherwise the default value
    """
    try:
        # First, try to create the widget if it doesn't exist (for workflow parameters)
        create_widget_if_not_exists(widget_name, "text", str(default_value))
        
        # Try to get the value from widgets (works for both regular widgets and workflow parameters)
        value = dbutils.widgets.get(widget_name)
        
        # Return the value if it's not empty, otherwise return default
        if value and value.strip() != "":
            return value
        else:
            return default_value
            
    except Exception as e:
        # If widget access fails (e.g., dbutils not available), return default
        print(f"Warning: Could not access widget '{widget_name}': {e}")
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


def get_model_uri(catalog_name, schema_name, model_name, version=None):
    """
    Generate a model URI for Unity Catalog.
    
    Args:
        catalog_name (str): Catalog name
        schema_name (str): Schema name
        model_name (str): Model name
        version (str, optional): Model version. If None, uses the model name without version.
        
    Returns:
        str: Model URI
    """
    if version:
        return f"models:/{catalog_name}.{schema_name}.{model_name}/{version}"
    else:
        return f"models:/{catalog_name}.{schema_name}.{model_name}"


def create_text_classification_signature():
    """
    Create a model signature for text classification models.
    
    Returns:
        ModelSignature: MLflow model signature for text classification
    """
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec
    import numpy as np
    
    # Define input schema (expecting text input)
    input_schema = Schema([
        TensorSpec(np.dtype(np.str_), (-1,), "text")
    ])
    
    # Define output schema (expecting string predictions)
    output_schema = Schema([
        TensorSpec(np.dtype(np.str_), (-1,), "prediction")
    ])
    
    # Create and return model signature
    return ModelSignature(inputs=input_schema, outputs=output_schema)


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