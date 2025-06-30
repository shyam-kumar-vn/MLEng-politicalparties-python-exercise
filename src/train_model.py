import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from src.text_loader.loader import DataLoader

def train_model(X_train, y_train, X_test, y_test, model_type="LogisticRegression", max_iter=1000, random_state=42):
    """
    Train a classification model and return the trained model and evaluation metrics.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_type: Type of model to train (default: LogisticRegression, options: XGBoost, LogisticRegression)
        max_iter: Maximum iterations for training (default: 1000, only used for LogisticRegression)
        random_state: Random state for reproducibility (default: 42)
    
    Returns:
        tuple: (trained_model, metrics_dict)
    """
    # Model selection and training
    if model_type == "XGBoost":
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='logloss',
            verbosity=0
        )
    elif model_type == "LogisticRegression":
        clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    clf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return clf, metrics

def main():
    """Main training pipeline"""
    # 1. Data Loading and Preprocessing
    loader = DataLoader()
    X = loader.preprocess_tweets()
    y = loader.preprocess_parties()

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Train Model (now using LogisticRegression by default)
    clf, metrics = train_model(X_train, y_train, X_test, y_test, model_type="LogisticRegression")

    # 4. MLflow Tracking
    mlflow.set_experiment("tweet_party_classification")
    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        
        # Log LogisticRegression model
        mlflow.sklearn.log_model(clf, "model")
        
        # Optionally log confusion matrix as an artifact
        import numpy as np
        np.savetxt("confusion_matrix.csv", metrics['confusion_matrix'], delimiter=",", fmt="%d")
        mlflow.log_artifact("confusion_matrix.csv")
        os.remove("confusion_matrix.csv")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:\n", metrics['confusion_matrix'])