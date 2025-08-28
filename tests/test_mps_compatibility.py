import pytest
import torch
import os
from unittest.mock import patch, MagicMock

from vibevoice.model import load_vibevoice_model
from vibevoice.utils.device_config import get_optimal_config


# Get the test model name from environment variable, with a default
TEST_MODEL_NAME = os.getenv("VIBEVOICE_TEST_MODEL", "microsoft/VibeVoice-1.5B")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_model_loading():
    """Test that model loads correctly on MPS device."""
    # Get optimal config which should return MPS device on Apple Silicon
    config = get_optimal_config()
    
    # Verify that the device is MPS
    assert config["device"].type == "mps", f"Expected device type 'mps', got '{config['device'].type}'"
    
    # Load the model with the resolved config
    with patch('vibevoice.modular.modeling_vibevoice_inference.VibeVoiceForConditionalGenerationInference.from_pretrained') as mock_model, \
         patch('vibevoice.processor.vibevoice_processor.VibeVoiceProcessor.from_pretrained') as mock_processor:
        
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance
        
        # Load the model
        model, processor = load_vibevoice_model(
            TEST_MODEL_NAME,
            device=config["device"],
            torch_dtype=config["dtype"],
            attn_implementation=config["attn_implementation"]
        )
        
        # Verify the model was loaded and moved to MPS
        mock_model.assert_called_once_with(
            TEST_MODEL_NAME,
            torch_dtype=config["dtype"],
            attn_implementation=config["attn_implementation"]
        )
        mock_processor.assert_called_once_with(TEST_MODEL_NAME)
        mock_model_instance.to.assert_called_once_with(torch.device("mps"))
        
        # Verify the returned objects are not None
        assert model is not None
        assert processor is not None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_inference():
    """Test basic inference on MPS device."""
    # Load model and processor via get_optimal_config
    config = get_optimal_config()
    
    with patch('vibevoice.modular.modeling_vibevoice_inference.VibeVoiceForConditionalGenerationInference.from_pretrained') as mock_model, \
         patch('vibevoice.processor.vibevoice_processor.VibeVoiceProcessor.from_pretrained') as mock_processor:
        
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance
        
        # Mock the tokenizer behavior
        mock_processor_instance.tokenize_text.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]], device="mps"),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], device="mps")
        }
        
        # Mock the model generate method
        input_length = 5
        output_length = input_length + 16  # max_new_tokens=16
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]], device="mps")
        
        # Load the model
        model, processor = load_vibevoice_model(
            TEST_MODEL_NAME,
            device=config["device"],
            torch_dtype=config["dtype"],
            attn_implementation=config["attn_implementation"]
        )
        
        # Tokenize a short prompt
        prompt = "Hello, this is a test prompt for MPS inference."
        inputs = processor.tokenize_text(prompt)
        
        # Move tensors to MPS if they're not already
        input_ids = inputs["input_ids"].to("mps")
        attention_mask = inputs["attention_mask"].to("mps")
        
        # Call model.generate with max_new_tokens=16
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=16
            )
        
        # Assert output shape length increased relative to input
        assert output.shape[1] > input_ids.shape[1], f"Output length ({output.shape[1]}) should be greater than input length ({input_ids.shape[1]})"
        assert output.shape[1] == output_length, f"Expected output length {output_length}, got {output.shape[1]}"