import unittest
from unittest.mock import patch, MagicMock
import torch

from vibevoice.model import load_vibevoice_model


class TestModelLoading(unittest.TestCase):
    @patch("vibevoice.model.get_optimal_config")
    @patch("vibevoice.model.VibeVoiceProcessor.from_pretrained")
    @patch("vibevoice.model.VibeVoiceForConditionalGenerationInference.from_pretrained")
    def test_load_model_auto_config(
        self,
        mock_model_from_pretrained,
        mock_processor_from_pretrained,
        mock_get_optimal_config,
    ):
        # Arrange
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_processor_from_pretrained.return_value = mock_processor
        mock_get_optimal_config.return_value = (
            "cuda",
            torch.float16,
            "flash_attention_2",
        )

        model_name = "test-model"

        # Act
        model, processor = load_vibevoice_model(model_name)

        # Assert
        mock_get_optimal_config.assert_called_once()
        mock_model_from_pretrained.assert_called_once_with(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        mock_processor_from_pretrained.assert_called_once_with(model_name)
        mock_model.to.assert_called_once_with(torch.device("cuda"))
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)

    @patch("vibevoice.model.get_optimal_config")
    @patch("vibevoice.model.VibeVoiceProcessor.from_pretrained")
    @patch("vibevoice.model.VibeVoiceForConditionalGenerationInference.from_pretrained")
    def test_load_model_explicit_args(
        self,
        mock_model_from_pretrained,
        mock_processor_from_pretrained,
        mock_get_optimal_config,
    ):
        # Arrange
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_processor_from_pretrained.return_value = mock_processor
        mock_get_optimal_config.return_value = (
            "cuda",
            torch.float16,
            "flash_attention_2",
        )

        model_name = "test-model"
        device = "cpu"
        dtype = torch.float32
        attn_impl = "sdpa"

        # Act
        model, processor = load_vibevoice_model(
            model_name, device=device, torch_dtype=dtype, attn_implementation=attn_impl
        )

        # Assert
        mock_get_optimal_config.assert_called_once()
        mock_model_from_pretrained.assert_called_once_with(
            model_name, torch_dtype=dtype, attn_implementation=attn_impl
        )
        mock_processor_from_pretrained.assert_called_once_with(model_name)
        mock_model.to.assert_called_once_with(torch.device(device))
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)


if __name__ == "__main__":
    unittest.main()
