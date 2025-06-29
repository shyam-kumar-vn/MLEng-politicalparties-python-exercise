import sys
import numpy as np
import pytest
sys.path.append('src')
from text_loader.loader import DataLoader
from train_model import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Test DataLoader Integration
def test_dataloader_shapes():
    loader = DataLoader()
    X = loader.preprocess_tweets()
    y = loader.preprocess_parties()
    assert len(X) == len(y), "Features and labels must have the same number of samples."
    assert X.shape[0] > 0, "Feature array should not be empty."
    assert y.shape[0] > 0, "Label array should not be empty."

# Step 2: Test Model Training on Synthetic Data
def test_model_training_on_toy_data():
    # Create a small synthetic dataset
    X = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
    y = np.array([0, 1, 1, 0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Use the same training function as the main script
    clf, metrics = train_model(X_train, y_train, X_test, y_test)
    
    # Validate the returned model and metrics
    assert hasattr(clf, 'predict'), "Trained model should have a predict method"
    assert 'accuracy' in metrics, "Metrics should include accuracy"
    assert metrics['accuracy'] >= 0.0, "Model should produce a valid accuracy score"
    assert set(np.unique(clf.predict(X_test))).issubset({0, 1}), "Predictions should be in the set of labels" 