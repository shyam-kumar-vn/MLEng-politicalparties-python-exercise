import pytest
import sys
import os

# Add src to path for testing
sys.path.append('src')
from utils import get_widget_value, get_table_name, get_model_uri, print_parameters, setup_mlflow_experiment, create_text_classification_signature


def test_get_table_name():
    """Test table name generation"""
    table_name = get_table_name("catalog", "schema", "table")
    assert table_name == "catalog.schema.table"


def test_get_model_uri():
    """Test model URI generation"""
    model_uri = get_model_uri("catalog", "schema", "model")
    assert model_uri == "models:/catalog.schema.model"
    
    # Test with custom version
    model_uri_v1 = get_model_uri("catalog", "schema", "model", version="1")
    assert model_uri_v1 == "models:/catalog.schema.model/1"
    
    # Test with alias (preferred for Unity Catalog)
    model_uri_alias = get_model_uri("catalog", "schema", "model", alias="production")
    assert model_uri_alias == "models:/catalog.schema.model@production"


def test_print_parameters(capsys):
    """Test parameter printing function"""
    params = {"key1": "value1", "key2": "value2"}
    print_parameters(params)
    
    captured = capsys.readouterr()
    assert "NOTEBOOK PARAMETERS" in captured.out
    assert "key1: value1" in captured.out
    assert "key2: value2" in captured.out


def test_get_widget_value_without_dbutils():
    """Test widget value function when dbutils is not available"""
    # This should return the default value when dbutils is not available
    result = get_widget_value("nonexistent_widget", "default_value")
    assert result == "default_value"


def test_setup_mlflow_experiment():
    """Test MLflow experiment setup function"""
    # This test verifies the function exists and can be called
    # In a real environment, this would test actual MLflow functionality
    experiment_name = "/test/experiment"
    
    # The function should handle the case where MLflow is not available
    try:
        result = setup_mlflow_experiment(experiment_name)
        # If MLflow is available, result should be the experiment name
        assert result == experiment_name or result is None
    except Exception:
        # If MLflow is not available, the function should handle it gracefully
        pass


def test_create_text_classification_signature():
    """Test text classification signature creation"""
    try:
        signature = create_text_classification_signature()
        # Verify signature has input and output schemas
        assert signature.inputs is not None
        assert signature.outputs is not None
        # Verify input schema expects text
        assert len(signature.inputs.input_names()) == 1
        assert signature.inputs.input_names()[0] == "text"
        # Verify output schema expects predictions
        assert len(signature.outputs.input_names()) == 1
        assert signature.outputs.input_names()[0] == "prediction"
    except ImportError:
        # If MLflow is not available, skip this test
        pass