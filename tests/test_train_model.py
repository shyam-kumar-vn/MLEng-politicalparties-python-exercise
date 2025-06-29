import sys
import numpy as np
import pytest
import pandas as pd
from unittest.mock import Mock, patch
import os
sys.path.append('src')
from text_loader.loader import DataLoader
from train_model import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Test DataLoader Integration
def test_dataloader_shapes():
    """Test that DataLoader produces correct shapes"""
    # Create toy data
    toy_data = pd.DataFrame({
        'Tweet': ['Hello world', 'Test tweet', 'Another message'],
        'Party': ['PartyA', 'PartyB', 'PartyA']
    })
    
    # Mock the load_data method to return our toy data
    with patch.object(DataLoader, 'load_data', return_value=None):
        loader = DataLoader()
        loader.data = toy_data
        
        # Test preprocessing methods
        X = loader.preprocess_tweets()
        y = loader.preprocess_parties()
        
        # Verify shapes
        assert X.shape[0] == 3  # 3 samples
        assert y.shape[0] == 3  # 3 labels
        assert len(np.unique(y)) <= 2  # Should have 2 unique labels

# Step 2: Test Model Training on Synthetic Data
def test_model_training_on_toy_data():
    """Test model training with toy data"""
    # Create toy data
    X_train = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[0, 0, 1], [1, 0, 0]])
    y_test = np.array([1, 0])
    
    # Train model
    clf, metrics = train_model(X_train, y_train, X_test, y_test)
    
    # Verify model was trained
    assert clf is not None
    assert hasattr(clf, 'predict')
    
    # Verify metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'confusion_matrix' in metrics
    
    # Verify predictions work
    assert set(np.unique(clf.predict(X_test))).issubset({0, 1}), "Predictions should be in the set of labels"

def test_column_renaming_for_predictions():
    """Test that column renaming logic works correctly for model predictions"""
    # Create test data with 'Tweet' column (like in our notebooks)
    test_data = pd.DataFrame({
        'Tweet': ['Hello world', 'Test tweet', 'Another message'],
        'Party': ['PartyA', 'PartyB', 'PartyA']
    })
    
    # Simulate the column renaming logic from notebooks
    prediction_data = test_data[['Tweet']].copy()
    prediction_data.columns = ['text']
    
    # Verify the renaming worked correctly
    assert 'text' in prediction_data.columns
    assert 'Tweet' not in prediction_data.columns
    assert len(prediction_data) == 3
    assert prediction_data['text'].iloc[0] == 'Hello world'
    assert prediction_data['text'].iloc[1] == 'Test tweet'
    assert prediction_data['text'].iloc[2] == 'Another message'

def test_dataframe_column_access():
    """Test different ways of accessing DataFrame columns"""
    # Create test data
    df = pd.DataFrame({
        'text': ['Hello', 'World'],
        'Tweet': ['Tweet1', 'Tweet2'],
        'other': ['other1', 'other2']
    })
    
    # Test accessing by column name
    if 'text' in df.columns:
        text_column = df['text']
    else:
        text_column = df.iloc[:, 0]
    
    assert len(text_column) == 2
    assert text_column.iloc[0] == 'Hello'
    
    # Test fallback to first column
    df_no_text = df.drop(columns=['text'])
    if 'text' in df_no_text.columns:
        fallback_column = df_no_text['text']
    else:
        fallback_column = df_no_text.iloc[:, 0]
    
    assert len(fallback_column) == 2
    assert fallback_column.iloc[0] == 'Tweet1'

if __name__ == "__main__":
    pytest.main([__file__]) 