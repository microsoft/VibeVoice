"""
Processor class for VibeVoice models.
"""

import os
import json
import warnings
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch

from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.utils import logging

from .audio_utils import AudioNormalizer

logger = logging.get_logger(__name__)

# Change from ProcessorMixin to FeatureExtractionMixin which is designed for single components
class VibeVoiceTokenizerProcessor(FeatureExtractionMixin):
    """
    Processor for VibeVoice acoustic tokenizer models.
    
    This processor handles audio preprocessing for VibeVoice models, including:
    - Audio format conversion (stereo to mono)
    - Optional audio normalization
    - Streaming support for infinite-length audio
    
    Args:
        sampling_rate (int, optional): Expected sampling rate. Defaults to 24000.
        normalize_audio (bool, optional): Whether to normalize audio. Defaults to True.
        target_dB_FS (float, optional): Target dB FS for normalization. Defaults to -25.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-6.
    """
    model_input_names = ["input_features"]
    
    def __init__(
        self,
        sampling_rate: int = 24000,
        normalize_audio: bool = True,
        target_dB_FS: float = -25,
        eps: float = 1e-6,
        max_audio_duration: float = 600.0,  # Maximum audio duration in seconds (10 minutes)
        max_file_size_mb: float = 100.0,  # Maximum file size in MB
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.max_audio_duration = max_audio_duration
        self.max_file_size_mb = max_file_size_mb
        
        # Allowed audio file formats (whitelist)
        self.allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.pt', '.npy'}
        
        # Initialize audio normalizer if needed
        if self.normalize_audio:
            self.normalizer = AudioNormalizer(target_dB_FS=target_dB_FS, eps=eps)
        else:
            self.normalizer = None
        
        # Save config
        self.feature_extractor_dict = {
            "sampling_rate": sampling_rate,
            "normalize_audio": normalize_audio,
            "target_dB_FS": target_dB_FS,
            "eps": eps,
            "max_audio_duration": max_audio_duration,
            "max_file_size_mb": max_file_size_mb,
        }
    
    def _validate_audio_file(self, audio_path: str) -> None:
        """
        Validate audio file for security: check extension, size, and existence.
        
        Args:
            audio_path (str): Path to audio file
            
        Raises:
            ValueError: If file is invalid or exceeds size limits
            FileNotFoundError: If file does not exist
        """
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Validate file extension (whitelist)
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported or potentially unsafe file format: {file_ext}. "
                f"Allowed formats: {', '.join(sorted(self.allowed_extensions))}"
            )
        
        # Check file size
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"Audio file size ({file_size_mb:.2f} MB) exceeds maximum allowed "
                f"size ({self.max_file_size_mb} MB). This prevents resource exhaustion attacks."
            )
    
    def _validate_audio_array(self, audio: np.ndarray) -> None:
        """
        Validate audio array for security: check shape, duration, and content.
        
        Args:
            audio (np.ndarray): Audio array to validate
            
        Raises:
            ValueError: If audio array is invalid or exceeds limits
        """
        # Check if array is empty
        if audio.size == 0:
            raise ValueError("Audio array is empty")
        
        # Check for NaN or Inf values (potential malicious input)
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or Inf values, which may indicate corrupted or malicious data")
        
        # Get audio duration
        if len(audio.shape) == 1:
            num_samples = audio.shape[0]
        elif len(audio.shape) == 2:
            # Get the time dimension (assuming it's the larger one)
            num_samples = max(audio.shape[0], audio.shape[1])
        else:
            raise ValueError(f"Audio should be 1D or 2D, got shape: {audio.shape}")
        
        duration_seconds = num_samples / self.sampling_rate
        
        # Check duration limit
        if duration_seconds > self.max_audio_duration:
            raise ValueError(
                f"Audio duration ({duration_seconds:.2f}s) exceeds maximum allowed "
                f"duration ({self.max_audio_duration}s). This prevents resource exhaustion attacks."
            )
        
        # Check for reasonable amplitude range (prevent extreme values)
        max_abs_value = np.abs(audio).max()
        if max_abs_value > 1e6:
            raise ValueError(
                f"Audio contains extreme amplitude values (max: {max_abs_value}), "
                "which may indicate malicious or corrupted data"
            )
    
    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono if needed.
        
        Args:
            audio (np.ndarray): Input audio array
            
        Returns:
            np.ndarray: Mono audio array
        """
        if len(audio.shape) == 1:
            return audio
        elif len(audio.shape) == 2:
            if audio.shape[0] == 2:  # (2, time)
                return np.mean(audio, axis=0)
            elif audio.shape[1] == 2:  # (time, 2)
                return np.mean(audio, axis=1)
            else:
                # If one dimension is 1, squeeze it
                if audio.shape[0] == 1:
                    return audio.squeeze(0)
                elif audio.shape[1] == 1:
                    return audio.squeeze(1)
                else:
                    raise ValueError(f"Unexpected audio shape: {audio.shape}")
        else:
            raise ValueError(f"Audio should be 1D or 2D, got shape: {audio.shape}")
    
    def _process_single_audio(self, audio: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Process a single audio array.
        
        Args:
            audio: Single audio input
            
        Returns:
            np.ndarray: Processed audio
        """
        # Convert to numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)
        
        # Validate audio array for security
        self._validate_audio_array(audio)
        
        # Ensure mono
        audio = self._ensure_mono(audio)
        
        # Normalize if requested
        if self.normalize_audio and self.normalizer is not None:
            audio = self.normalizer(audio)
        
        return audio
    
    def __call__(
        self,
        audio: Union[str, np.ndarray, List[float], List[np.ndarray], List[List[float]], List[str]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        """
        Process audio for VibeVoice models.
        
        Args:
            audio: Audio input(s) to process. Can be:
                - str: Path to audio file
                - np.ndarray: Audio array
                - List[float]: Audio as list of floats
                - List[np.ndarray]: Batch of audio arrays
                - List[str]: Batch of audio file paths
            sampling_rate (int, optional): Sampling rate of the input audio
            return_tensors (str, optional): Return format ('pt' for PyTorch, 'np' for NumPy)
            
        Returns:
            dict: Processed audio inputs with keys:
                - input_features: Audio tensor(s) ready for the model
        """
        if audio is None:
            raise ValueError("Audio input is required")
        
        # Validate sampling rate
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            logger.warning(
                f"Input sampling rate ({sampling_rate}) differs from expected "
                f"sampling rate ({self.sampling_rate}). Please resample your audio."
            )
        
        # Security: Validate input type to prevent unexpected objects
        if not isinstance(audio, (str, np.ndarray, list)):
            raise ValueError(
                f"Invalid audio input type: {type(audio).__name__}. "
                f"Expected str, np.ndarray, or list."
            )
        
        # Handle different input types
        if isinstance(audio, str):
            # Single audio file path
            audio = self._load_audio_from_path(audio)
            is_batched = False
        elif isinstance(audio, list):
            if len(audio) == 0:
                raise ValueError("Empty audio list provided")
            
            # Check if it's a list of file paths
            if all(isinstance(item, str) for item in audio):
                # Batch of audio file paths
                audio = [self._load_audio_from_path(path) for path in audio]
                is_batched = True
            else:
                # Check if it's batched audio arrays
                is_batched = isinstance(audio[0], (np.ndarray, list))
        else:
            # Single audio array or list
            is_batched = False
        
        # Process audio
        if is_batched:
            processed_audio = [self._process_single_audio(a) for a in audio]
        else:
            processed_audio = [self._process_single_audio(audio)]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            if len(processed_audio) == 1:
                # Create a proper batch dimension (B, T)
                input_features = torch.from_numpy(processed_audio[0]).unsqueeze(0).unsqueeze(1)
            else:
                # For batched input with different lengths, create a batch properly
                input_features = torch.stack([torch.from_numpy(a) for a in processed_audio]).unsqueeze(1)
        elif return_tensors == "np":
            if len(processed_audio) == 1:
                input_features = processed_audio[0][np.newaxis, np.newaxis, :]
            else:
                input_features = np.stack(processed_audio)[:, np.newaxis, :]
        else:
            input_features = processed_audio[0] if len(processed_audio) == 1 else processed_audio
        
        outputs = {
            "audio": input_features,  # Use "audio" instead of "input_features"
        }
        
        return outputs

    def _load_audio_from_path(self, audio_path: str) -> np.ndarray:
        """
        Load audio from file path with security validation.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.ndarray: Loaded audio array
        """
        # Validate file before loading (security check)
        self._validate_audio_file(audio_path)
        
        # Get file extension to determine loading method
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        if file_ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            # Audio file - use librosa
            try:
                import librosa
                audio_array, sr = librosa.load(
                    audio_path, 
                    sr=self.sampling_rate, 
                    mono=True
                )
                return audio_array
            except Exception as e:
                raise ValueError(
                    f"Failed to load audio file {audio_path}. "
                    f"The file may be corrupted or maliciously crafted. Error: {str(e)}"
                )
        elif file_ext == '.pt':
            # PyTorch tensor file - use weights_only=True for security
            try:
                audio_tensor = torch.load(audio_path, map_location='cpu', weights_only=True).squeeze()
                if isinstance(audio_tensor, torch.Tensor):
                    audio_array = audio_tensor.numpy()
                else:
                    audio_array = np.array(audio_tensor)
                return audio_array.astype(np.float32)
            except Exception as e:
                raise ValueError(
                    f"Failed to load PyTorch tensor file {audio_path}. "
                    f"The file may be corrupted or contain unsafe code. Error: {str(e)}"
                )
        elif file_ext == '.npy':
            # NumPy file - use allow_pickle=False for security
            try:
                audio_array = np.load(audio_path, allow_pickle=False)
                return audio_array.astype(np.float32)
            except Exception as e:
                raise ValueError(
                    f"Failed to load NumPy file {audio_path}. "
                    f"The file may be corrupted or contain unsafe pickled objects. Error: {str(e)}"
                )
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {', '.join(sorted(self.allowed_extensions))}"
            )
    
    def preprocess_audio(
        self, 
        audio_path_or_array: Union[str, np.ndarray],
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Convenience method to preprocess audio from file path or array.
        This method is kept for backward compatibility but __call__ is recommended.
        
        Args:
            audio_path_or_array: Path to audio file or numpy array
            normalize: Whether to normalize (overrides default setting)
            
        Returns:
            np.ndarray: Preprocessed audio array
        """
        if isinstance(audio_path_or_array, str):
            audio_array = self._load_audio_from_path(audio_path_or_array)
        else:
            audio_array = np.array(audio_path_or_array, dtype=np.float32)
        
        # Override normalization setting if specified
        original_normalize = self.normalize_audio
        if normalize is not None:
            self.normalize_audio = normalize
        
        try:
            processed = self._process_single_audio(audio_array)
        finally:
            # Restore original setting
            self.normalize_audio = original_normalize
        
        return processed
    
    # Override to_dict method for configuration saving
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the object to a dict containing all attributes needed for serialization.
        """
        return self.feature_extractor_dict

    def save_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
        output_path: str = "output.wav",
        sampling_rate: Optional[int] = None,
        normalize: bool = False,
        batch_prefix: str = "audio_",
    ):
        """
        Save audio data to WAV file(s).
        
        Args:
            audio: Audio data to save. Can be:
                - torch.Tensor: PyTorch tensor with shape (B, C, T) or (B, T) or (T)
                - np.ndarray: NumPy array with shape (B, C, T) or (B, T) or (T)
                - List of tensors or arrays
            output_path: Path where to save the audio. If saving multiple files,
                this is treated as a directory and individual files will be saved inside.
            sampling_rate: Sampling rate for the saved audio. Defaults to the processor's rate.
            normalize: Whether to normalize audio before saving.
            batch_prefix: Prefix for batch files when saving multiple audios.
                
        Returns:
            List[str]: Paths to the saved audio files.
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required to save audio files. "
                "Install it with: pip install soundfile"
            )
        
        # Ensure audio is in the right format
        if isinstance(audio, torch.Tensor):
            # Convert PyTorch tensor to numpy
            audio_np = audio.float().detach().cpu().numpy()
        elif isinstance(audio, np.ndarray):
            audio_np = audio
        elif isinstance(audio, list):
            # Handle list of tensors or arrays
            if all(isinstance(a, torch.Tensor) for a in audio):
                audio_np = [a.float().detach().cpu().numpy() for a in audio]
            else:
                audio_np = audio
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        saved_paths = []
        
        # Handle based on shape or type
        if isinstance(audio_np, list):
            # Multiple separate audios to save
            output_dir = output_path
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each audio
            for i, audio_item in enumerate(audio_np):
                audio_item = self._prepare_audio_for_save(audio_item, normalize)
                file_path = os.path.join(output_dir, f"{batch_prefix}{i}.wav")
                sf.write(file_path, audio_item, sampling_rate)
                saved_paths.append(file_path)
                
        else:
            # Handle different dimensions
            if len(audio_np.shape) >= 3:  # (B, C, T) or similar
                # Get batch size
                batch_size = audio_np.shape[0]
                
                if batch_size > 1:
                    # Multiple audios in a batch
                    output_dir = output_path
                    
                    # Ensure output directory exists
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save each audio in the batch
                    for i in range(batch_size):
                        # Extract single audio and remove channel dim if present
                        single_audio = audio_np[i]
                        if len(single_audio.shape) > 1:
                            if single_audio.shape[0] == 1:  # (1, T)
                                single_audio = single_audio.squeeze(0)
                        
                        single_audio = self._prepare_audio_for_save(single_audio, normalize)
                        file_path = os.path.join(output_dir, f"{batch_prefix}{i}.wav")
                        sf.write(file_path, single_audio, sampling_rate)
                        saved_paths.append(file_path)
                else:
                    # Single audio with batch and channel dims
                    audio_item = audio_np.squeeze()  # Remove batch and channel dimensions
                    audio_item = self._prepare_audio_for_save(audio_item, normalize)
                    sf.write(output_path, audio_item, sampling_rate)
                    saved_paths.append(output_path)
            else:
                # Single audio without batch dimension
                audio_item = self._prepare_audio_for_save(audio_np, normalize)
                sf.write(output_path, audio_item, sampling_rate)
                saved_paths.append(output_path)
        
        return saved_paths

    def _prepare_audio_for_save(self, audio: np.ndarray, normalize: bool) -> np.ndarray:
        """
        Prepare audio for saving by ensuring it's the right shape and optionally normalizing.
        
        Args:
            audio: Audio data as numpy array
            normalize: Whether to normalize audio
            
        Returns:
            np.ndarray: Processed audio ready for saving
        """
        # Ensure right dimensionality
        if len(audio.shape) > 1 and audio.shape[0] == 1:  # (1, T)
            audio = audio.squeeze(0)
        
        # Normalize if requested
        if normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        
        return audio


__all__ = ["VibeVoiceTokenizerProcessor", "AudioNormalizer"]