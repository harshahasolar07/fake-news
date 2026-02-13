import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import TextPreprocessor, FeatureExtractor


def test_text_preprocessing():
    """Test text preprocessing functionality."""
    preprocessor = TextPreprocessor()
    
    # Test basic cleaning
    text = "This is a TEST! http://example.com @user #hashtag"
    cleaned = preprocessor.clean_text(text)
    
    assert "http" not in cleaned.lower()
    assert "@user" not in cleaned
    assert "#" not in cleaned
    assert len(cleaned) > 0


def test_empty_text():
    """Test handling of empty text."""
    preprocessor = TextPreprocessor()
    
    result = preprocessor.clean_text("")
    assert result == ""
    
    result = preprocessor.clean_text(None)
    assert result == ""


def test_feature_extraction():
    """Test TF-IDF feature extraction."""
    extractor = FeatureExtractor(max_features=100)
    
    texts = [
        "This is a news article",
        "This is another article",
        "Completely different content"
    ]
    
    features = extractor.fit_transform(texts)
    
    assert features.shape[0] == 3
    assert features.shape[1] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])