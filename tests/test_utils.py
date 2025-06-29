import pytest
import sys
import os

# Add src to path for testing
sys.path.append('src')
from utils import get_widget_value, get_table_name, get_model_uri, print_parameters


def test_get_table_name():
    """Test table name generation"""
    table_name = get_table_name("catalog", "schema", "table")
    assert table_name == "catalog.schema.table"


def test_get_model_uri():
    """Test model URI generation"""
    model_uri = get_model_uri("catalog", "schema", "model")
    assert model_uri == "models:/catalog.schema.model/latest"
    
    # Test with custom version
    model_uri_v1 = get_model_uri("catalog", "schema", "model", "1")
    assert model_uri_v1 == "models:/catalog.schema.model/1"


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