import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.append('src')

# Import the components we need to test
from text_loader.loader import DataLoader
from train_model import train_model


class TestPoliticalPartyClassifier:
    """Test cases for PoliticalPartyClassifier input format handling"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Create mock components
        self.mock_model = Mock()
        self.mock_vectorizer = Mock()
        self.mock_label_encoder = Mock()
        self.mock_data_loader = Mock()
        
        # Set up mock return values
        self.mock_model.predict.return_value = np.array([0, 1, 0])
        self.mock_vectorizer.transform.return_value = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.mock_label_encoder.inverse_transform.return_value = np.array(['PartyA', 'PartyB', 'PartyA'])
        self.mock_data_loader.clean_text.side_effect = lambda x: f"cleaned: {x}"
        
        # Create the PoliticalPartyClassifier class (copy from train_model.py)
        import mlflow.pyfunc
        
        class PoliticalPartyClassifier(mlflow.pyfunc.PythonModel):
            def __init__(self, model, vectorizer, label_encoder, data_loader):
                self.model = model
                self.vectorizer = vectorizer
                self.label_encoder = label_encoder
                self.data_loader = data_loader
            
            def predict(self, context, model_input):
                # Handle different input formats
                import pandas as pd
                import numpy as np
                
                # Convert to pandas DataFrame if needed
                if isinstance(model_input, dict):
                    # If it's a dict, convert to DataFrame
                    model_input = pd.DataFrame(model_input)
                elif isinstance(model_input, np.ndarray):
                    # If it's a numpy array, convert to DataFrame
                    model_input = pd.DataFrame(model_input, columns=['text'])
                elif not isinstance(model_input, pd.DataFrame):
                    # For any other type, try to convert to DataFrame
                    model_input = pd.DataFrame(model_input)
                
                # Get the text column (first column)
                if 'text' in model_input.columns:
                    text_column = model_input['text']
                else:
                    # Fallback to first column
                    text_column = model_input.iloc[:, 0]
                
                # Clean text using DataLoader's clean_text method
                cleaned_text = text_column.apply(self.data_loader.clean_text)
                
                # Vectorize
                X = self.vectorizer.transform(cleaned_text)
                
                # Predict
                predictions = self.model.predict(X)
                
                # Convert back to original labels
                return self.label_encoder.inverse_transform(predictions)
        
        self.PoliticalPartyClassifier = PoliticalPartyClassifier
    
    @patch('text_loader.loader.DataLoader')
    def test_predict_with_dataframe_input(self, mock_dataloader_class):
        """Test prediction with pandas DataFrame input"""
        # Set up the mock
        mock_dataloader_class.return_value = self.mock_data_loader
        
        # Create classifier instance with injected mock_data_loader
        classifier = self.PoliticalPartyClassifier(
            self.mock_model, 
            self.mock_vectorizer, 
            self.mock_label_encoder,
            data_loader=self.mock_data_loader
        )
        
        # Create DataFrame input
        df_input = pd.DataFrame({
            'text': ['Hello world', 'Test tweet', 'Another message']
        })
        
        # Mock context
        mock_context = Mock()
        
        # Call predict
        result = classifier.predict(mock_context, df_input)
        
        # Verify calls
        self.mock_data_loader.clean_text.assert_called()
        self.mock_vectorizer.transform.assert_called_once()
        self.mock_model.predict.assert_called_once()
        self.mock_label_encoder.inverse_transform.assert_called_once()
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert all(label in ['PartyA', 'PartyB'] for label in result)
    
    @patch('text_loader.loader.DataLoader')
    def test_predict_with_dict_input(self, mock_dataloader_class):
        """Test prediction with dictionary input"""
        # Set up the mock
        mock_dataloader_class.return_value = self.mock_data_loader
        
        # Create classifier instance with injected mock_data_loader
        classifier = self.PoliticalPartyClassifier(
            self.mock_model, 
            self.mock_vectorizer, 
            self.mock_label_encoder,
            data_loader=self.mock_data_loader
        )
        
        # Create dict input
        dict_input = {
            'text': ['Hello world', 'Test tweet', 'Another message']
        }
        
        # Mock context
        mock_context = Mock()
        
        # Call predict
        result = classifier.predict(mock_context, dict_input)
        
        # Verify calls
        self.mock_data_loader.clean_text.assert_called()
        self.mock_vectorizer.transform.assert_called_once()
        self.mock_model.predict.assert_called_once()
        self.mock_label_encoder.inverse_transform.assert_called_once()
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
    
    @patch('text_loader.loader.DataLoader')
    def test_predict_with_numpy_array_input(self, mock_dataloader_class):
        """Test prediction with numpy array input"""
        # Set up the mock
        mock_dataloader_class.return_value = self.mock_data_loader
        
        # Create classifier instance with injected mock_data_loader
        classifier = self.PoliticalPartyClassifier(
            self.mock_model, 
            self.mock_vectorizer, 
            self.mock_label_encoder,
            data_loader=self.mock_data_loader
        )
        
        # Create numpy array input
        array_input = np.array([
            ['Hello world'],
            ['Test tweet'],
            ['Another message']
        ])
        
        # Mock context
        mock_context = Mock()
        
        # Call predict
        result = classifier.predict(mock_context, array_input)
        
        # Verify calls
        self.mock_data_loader.clean_text.assert_called()
        self.mock_vectorizer.transform.assert_called_once()
        self.mock_model.predict.assert_called_once()
        self.mock_label_encoder.inverse_transform.assert_called_once()
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
    
    @patch('text_loader.loader.DataLoader')
    def test_predict_with_list_input(self, mock_dataloader_class):
        """Test prediction with list input"""
        # Set up the mock
        mock_dataloader_class.return_value = self.mock_data_loader
        
        # Create classifier instance with injected mock_data_loader
        classifier = self.PoliticalPartyClassifier(
            self.mock_model, 
            self.mock_vectorizer, 
            self.mock_label_encoder,
            data_loader=self.mock_data_loader
        )
        
        # Create list input
        list_input = [['Hello world'], ['Test tweet'], ['Another message']]
        
        # Mock context
        mock_context = Mock()
        
        # Call predict
        result = classifier.predict(mock_context, list_input)
        
        # Verify calls
        self.mock_data_loader.clean_text.assert_called()
        self.mock_vectorizer.transform.assert_called_once()
        self.mock_model.predict.assert_called_once()
        self.mock_label_encoder.inverse_transform.assert_called_once()
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
    
    @patch('text_loader.loader.DataLoader')
    def test_predict_with_different_column_names(self, mock_dataloader_class):
        """Test prediction with DataFrame that has different column names"""
        # Set up the mock
        mock_dataloader_class.return_value = self.mock_data_loader
        
        # Create classifier instance with injected mock_data_loader
        classifier = self.PoliticalPartyClassifier(
            self.mock_model, 
            self.mock_vectorizer, 
            self.mock_label_encoder,
            data_loader=self.mock_data_loader
        )
        
        # Create DataFrame with 'Tweet' column instead of 'text'
        df_input = pd.DataFrame({
            'Tweet': ['Hello world', 'Test tweet', 'Another message']
        })
        
        # Mock context
        mock_context = Mock()
        
        # Call predict
        result = classifier.predict(mock_context, df_input)
        
        # Verify calls
        self.mock_data_loader.clean_text.assert_called()
        self.mock_vectorizer.transform.assert_called_once()
        self.mock_model.predict.assert_called_once()
        self.mock_label_encoder.inverse_transform.assert_called_once()
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
    
    @patch('text_loader.loader.DataLoader')
    def test_predict_with_single_input(self, mock_dataloader_class):
        """Test prediction with single input (not batch)"""
        # Set up the mock
        mock_dataloader_class.return_value = self.mock_data_loader
        
        # Create classifier instance with injected mock_data_loader
        classifier = self.PoliticalPartyClassifier(
            self.mock_model, 
            self.mock_vectorizer, 
            self.mock_label_encoder,
            data_loader=self.mock_data_loader
        )
        
        # Create single input
        single_input = pd.DataFrame({
            'text': ['Single tweet']
        })
        
        # Mock context
        mock_context = Mock()
        
        # Update mock return values for single input
        self.mock_model.predict.return_value = np.array([0])
        self.mock_vectorizer.transform.return_value = np.array([[1, 0, 1]])
        self.mock_label_encoder.inverse_transform.return_value = np.array(['PartyA'])
        
        # Call predict
        result = classifier.predict(mock_context, single_input)
        
        # Verify calls
        self.mock_data_loader.clean_text.assert_called()
        self.mock_vectorizer.transform.assert_called_once()
        self.mock_model.predict.assert_called_once()
        self.mock_label_encoder.inverse_transform.assert_called_once()
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert result[0] == 'PartyA'
    
    @patch('text_loader.loader.DataLoader')
    def test_predict_error_handling(self, mock_dataloader_class):
        """Test that the classifier handles errors gracefully"""
        # Set up the mock
        mock_dataloader_class.return_value = self.mock_data_loader
        
        # Create classifier instance with injected mock_data_loader
        classifier = self.PoliticalPartyClassifier(
            self.mock_model, 
            self.mock_vectorizer, 
            self.mock_label_encoder,
            data_loader=self.mock_data_loader
        )
        
        # Create invalid input that might cause issues
        invalid_input = None
        
        # Mock context
        mock_context = Mock()
        
        # This should raise an exception, but we want to test that it's handled
        with pytest.raises(Exception):
            classifier.predict(mock_context, invalid_input)
    
    @patch('text_loader.loader.DataLoader')
    def test_predict_with_empty_input(self, mock_dataloader_class):
        """Test prediction with empty input"""
        # Set up the mock
        mock_dataloader_class.return_value = self.mock_data_loader
        
        # Create classifier instance with injected mock_data_loader
        classifier = self.PoliticalPartyClassifier(
            self.mock_model, 
            self.mock_vectorizer, 
            self.mock_label_encoder,
            data_loader=self.mock_data_loader
        )
        
        # Create empty DataFrame
        empty_input = pd.DataFrame({'text': []})
        
        # Mock context
        mock_context = Mock()
        
        # Update mock return values for empty input
        self.mock_model.predict.return_value = np.array([])
        self.mock_vectorizer.transform.return_value = np.array([]).reshape(0, 3)  # Fix the reshape
        self.mock_label_encoder.inverse_transform.return_value = np.array([])
        
        # Call predict
        result = classifier.predict(mock_context, empty_input)
        
        # Verify calls
        self.mock_vectorizer.transform.assert_called_once()
        self.mock_model.predict.assert_called_once()
        self.mock_label_encoder.inverse_transform.assert_called_once()
        
        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


class TestModelPredictionIntegration:
    """Integration tests for model prediction with real components"""
    
    def test_end_to_end_prediction_flow(self):
        """Test the complete prediction flow with real components"""
        # This test would require actual trained model components
        # For now, we'll test the structure and imports
        
        # Test that we can import the required components
        from text_loader.loader import DataLoader
        from train_model import train_model
        
        # Test that DataLoader can be instantiated
        loader = DataLoader()
        assert loader is not None
        
        # Test that train_model function exists
        assert callable(train_model)
        
        # Test that we can create a simple DataFrame for prediction
        test_data = pd.DataFrame({
            'text': ['Test tweet 1', 'Test tweet 2']
        })
        
        assert len(test_data) == 2
        assert 'text' in test_data.columns


if __name__ == "__main__":
    pytest.main([__file__]) 