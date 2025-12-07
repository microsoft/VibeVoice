"""
Processor class for VibeVoice models.
Fixed and hardened version.
"""

import os
import warnings
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch

from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AudioNormalizer:
    """
    Audio normalization class for VibeVoice tokenizer.

    This class provides audio normalization to ensure consistent input levels
    for the VibeVoice tokenizer while maintaining audio quality.
    """

    def __init__(self, target_dB_FS: float = -25.0, eps: float = 1e-6):
        """
        Initialize the audio normalizer.

        Args:
            target_dB_FS (float): Target dB FS level for the audio. Default: -25
            eps (float): Small value to avoid division by zero. Default: 1e-6
        """
        self.target_dB_FS = float(target_dB_FS)
        self.eps = float(eps)

    def tailor_dB_FS(self, audio: np.ndarray) -> tuple:
        """
        Scale audio toward the target dB FS while avoiding clipping if scaling would clip.

        This computes the RMS-based scalar to get to target_dB_FS, then ensures applying
        that scalar will not push values beyond [-1, 1]. If it would, we reduce the scalar
        so the max absolute value becomes 1 - eps.

        Args:
            audio (np.ndarray): Input audio signal (1D)

        Returns:
            tuple: (normalized_audio, rms, applied_scalar)
        """
        if audio.size == 0:
            return audio, 0.0, 1.0

        rms = float(np.sqrt(np.mean(np.square(audio), where=~np.isnan(audio))))
        # target linear amplitude for RMS
        target_linear = 10 ** (self.target_dB_FS / 20.0)
        # candidate scalar to reach target RMS
        candidate_scalar = target_linear / (rms + self.eps)

        # If applying candidate_scalar would clip, reduce scalar to avoid clipping.
        max_after = np.max(np.abs(audio) * candidate_scalar)
        if max_after > 1.0:
            # scale down to avoid clipping; keep a small epsilon margin
            adjusted_scalar = (1.0 - self.eps) / max_after * candidate_scalar
            applied_scalar = min(candidate_scalar, adjusted_scalar)
        else:
            applied_scalar = candidate_scalar

        normalized_audio = audio * applied_scalar
        return normalized_audio, rms, applied_scalar

    def avoid_clipping(self, audio: np.ndarray, scalar: Optional[float] = None) -> tuple:
        """
        Avoid clipping by scaling down if necessary.

        Args:
            audio (np.ndarray): Input audio signal
            scalar (float, optional): Explicit scaling factor

        Returns:
            tuple: (normalized_audio, scalar)
        """
        if audio.size == 0:
            return audio, 1.0

        if scalar is None:
            max_val = float(np.max(np.abs(audio)))
            if max_val > 1.0:
                scalar = max_val + self.eps
            else:
                scalar = 1.0

        return audio / float(scalar), float(scalar)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize the audio by adjusting to target dB FS and avoiding clipping.

        Args:
            audio (np.ndarray): Input audio signal

        Returns:
            np.ndarray: Normalized audio signal
        """
        # Compute scale to reach RMS target, but ensure no clipping after scale.
        audio, _, _ = self.tailor_dB_FS(audio)
        # For safety ensure no clipping remains (tailor_dB_FS already checks, but double-check)
        audio, _ = self.avoid_clipping(audio)
        return audio


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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sampling_rate = int(sampling_rate)
        self.normalize_audio = bool(normalize_audio)

        # Initialize audio normalizer if needed
        if self.normalize_audio:
            self.normalizer = AudioNormalizer(target_dB_FS=float(target_dB_FS), eps=float(eps))
        else:
            self.normalizer = None

        # Save config
        self.feature_extractor_dict = {
            "sampling_rate": self.sampling_rate,
            "normalize_audio": self.normalize_audio,
            "target_dB_FS": float(target_dB_FS),
            "eps": float(eps),
        }

    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo or multichannel audio to mono by averaging across channels.

        Args:
            audio (np.ndarray): Input audio array

        Returns:
            np.ndarray: Mono audio array (1D)
        """
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # assume channels are the last dimension or the first; prefer last
            # common shapes: (T, C) or (C, T). Handle both.
            if audio.shape[0] <= 8 and audio.shape[1] > 1000:
                # likely (C, T)
                return np.mean(audio, axis=0).astype(np.float32)
            else:
                # likely (T, C) or ambiguous: average across last axis
                return np.mean(audio, axis=-1).astype(np.float32)
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
        # Convert to numpy array and ensure float32
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)

        # Ensure mono
        audio = self._ensure_mono(audio)

        # Normalize if requested and normalizer is available
        if self.normalize_audio and self.normalizer is not None:
            # normalizer handles clipping avoidance
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
                - input_features: Audio tensor(s) ready for the model (following HF naming)
        """
        if audio is None:
            raise ValueError("Audio input is required")

        # Validate sampling rate and warn if mismatched (resampling not performed automatically)
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            logger.warning(
                f"Input sampling rate ({sampling_rate}) differs from expected sampling rate ({self.sampling_rate}). "
                "This processor does not resample automatically; please resample your audio beforehand or implement resampling."
            )

        # Determine if input is batch or single
        is_batched = False
        if isinstance(audio, str):
            # Single audio file path
            audio = self._load_audio_from_path(audio)
            is_batched = False
        elif isinstance(audio, list):
            if len(audio) == 0:
                raise ValueError("Empty audio list provided")
            # If list of strings: list of file paths
            if all(isinstance(item, str) for item in audio):
                audio = [self._load_audio_from_path(path) for path in audio]
                is_batched = True
            else:
                # Distinguish between a list of floats (single audio) and a list of arrays (batch)
                first = audio[0]
                # Consider ints/floats as single audio when list contains numbers
                if isinstance(first, (float, int, np.floating, np.integer)):
                    # This is a single audio represented as a Python list of floats
                    audio = np.array(audio, dtype=np.float32)
                    is_batched = False
                else:
                    # treat as batch of arrays/lists
                    is_batched = True
        else:
            # numpy array or torch tensor input
            is_batched = False

        # Process audio(s)
        if is_batched:
            processed_audio = [self._process_single_audio(a) for a in audio]
        else:
            processed_audio = [self._process_single_audio(audio)]

        # Convert to requested tensor format
        if return_tensors == "pt":
            # Convert each processed piece to tensor
            torch_list = [torch.from_numpy(a) for a in processed_audio]
            # If single, return shape (1, 1, T) for compatibility with many audio models (batch, channels, time)
            if len(torch_list) == 1:
                input_features = torch_list[0].unsqueeze(0).unsqueeze(1)
            else:
                # If all tensors have same length, stack; otherwise return list
                lengths = [t.shape[-1] for t in torch_list]
                if len(set(lengths)) == 1:
                    input_features = torch.stack(torch_list).unsqueeze(1)
                else:
                    # lengths differ: return list of tensors (caller must handle padding)
                    input_features = [t.unsqueeze(0).unsqueeze(0) for t in torch_list]
        elif return_tensors == "np":
            if len(processed_audio) == 1:
                input_features = processed_audio[0][np.newaxis, np.newaxis, :]
            else:
                lengths = [a.shape[-1] for a in processed_audio]
                if len(set(lengths)) == 1:
                    input_features = np.stack(processed_audio)[:, np.newaxis, :]
                else:
                    # differing lengths: return list of arrays
                    input_features = [a[np.newaxis, np.newaxis, :] for a in processed_audio]
        else:
            # return python objects (numpy arrays or list of numpy arrays)
            input_features = processed_audio[0] if len(processed_audio) == 1 else processed_audio

        outputs = {
            "input_features": input_features,
        }

        return outputs

    def _load_audio_from_path(self, audio_path: str) -> np.ndarray:
        """
        Load audio from file path.

        Args:
            audio_path (str): Path to audio file

        Returns:
            np.ndarray: Loaded audio array (1D, mono)
        """
        if not isinstance(audio_path, str):
            raise ValueError("audio_path must be a string path")

        file_ext = os.path.splitext(audio_path)[1].lower()

        if file_ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"]:
            try:
                import librosa
            except ImportError as e:
                raise ImportError(
                    "librosa is required to load compressed audio formats. Install it with: pip install librosa"
                ) from e

            audio_array, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            return audio_array.astype(np.float32)
        elif file_ext == ".pt":
            audio_tensor = torch.load(audio_path, map_location="cpu")
            if isinstance(audio_tensor, torch.Tensor):
                audio_array = audio_tensor.squeeze().numpy()
            else:
                audio_array = np.array(audio_tensor)
            return audio_array.astype(np.float32)
        elif file_ext in [".npy", ".npz"]:
            # np.load handles both .npy and .npz
            arr = np.load(audio_path, allow_pickle=False)
            # If npz, get first array inside or named arrays
            if isinstance(arr, np.lib.npyio.NpzFile):
                # prefer first item
                keys = list(arr.keys())
                if len(keys) == 0:
                    raise ValueError(f"Empty .npz archive: {audio_path}")
                audio_array = arr[keys[0]]
            else:
                audio_array = arr
            return audio_array.astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported formats: "
                ".wav, .mp3, .flac, .m4a, .ogg, .opus, .pt, .npy, .npz"
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
            self.normalize_audio = bool(normalize)

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
        Merge with parent dict if available to remain compatible with HF save/load.
        """
        parent_dict = {}
        try:
            parent = super()
            if hasattr(parent, "to_dict"):
                parent_dict = parent.to_dict()
        except Exception:
            parent_dict = {}

        merged = dict(parent_dict)
        merged.update(self.feature_extractor_dict)
        return merged

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
        except Exception as e:
            raise ImportError(
                "soundfile is required to save audio files. Install it with: pip install soundfile"
            ) from e

        # Convert torch to numpy where necessary
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        elif isinstance(audio, np.ndarray):
            audio_np = audio
        elif isinstance(audio, list):
            # convert list elements
            converted = []
            for a in audio:
                if isinstance(a, torch.Tensor):
                    converted.append(a.detach().cpu().numpy())
                else:
                    converted.append(np.array(a))
            audio_np = converted
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

        saved_paths: List[str] = []

        def _write_single(a: np.ndarray, path: str):
            # Ensure 1D or 2D with shape (T,) or (T, channels)
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 3:
                # (B, C, T) -- handle batch size 1 as single audio, else error
                if a.shape[0] == 1:
                    a = np.squeeze(a, axis=0)
                else:
                    raise ValueError(
                        "3D array detected with batch size > 1. For batches, pass as a list or use output_path as a directory. "
                        f"Got shape {a.shape}."
                    )
            if a.ndim == 2:
                # (C, T) -> transpose to (T, C)
                if a.shape[0] <= 8 and a.shape[1] > a.shape[0]:
                    # likely (C, T)
                    a = a.T
                # else assume (T, C)
            # Normalize if requested
            if normalize:
                max_val = np.max(np.abs(a))
                if max_val > 0:
                    a = a / (max_val + 1e-12)
            # soundfile expects shape (T,) or (T, channels)
            sf.write(path, a, sampling_rate)

        if isinstance(audio_np, list):
            # treat output_path as directory
            os.makedirs(output_path, exist_ok=True)
            for i, item in enumerate(audio_np):
                file_path = os.path.join(output_path, f"{batch_prefix}{i}.wav")
                _write_single(item, file_path)
                saved_paths.append(file_path)
        else:
            # numpy array path handling
            a = np.asarray(audio_np)
            if a.ndim >= 3:
                # treat as batch in first dim
                batch_size = a.shape[0]
                os.makedirs(output_path if os.path.splitext(output_path)[1] == "" else os.path.dirname(output_path), exist_ok=True)
                # ensure output_path is a directory
                out_dir = output_path
                if os.path.splitext(output_path)[1] != "":
                    # user provided a file path but we have batch; make directory next to it
                    out_dir = os.path.splitext(output_path)[0] + "_batch"
                    os.makedirs(out_dir, exist_ok=True)
                for i in range(batch_size):
                    single = a[i]
                    file_path = os.path.join(out_dir, f"{batch_prefix}{i}.wav")
                    _write_single(single, file_path)
                    saved_paths.append(file_path)
            else:
                # single audio
                _write_single(a, output_path)
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
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        if normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / (max_val + 1e-12)
        return audio


__all__ = ["VibeVoiceTokenizerProcessor", "AudioNormalizer"]
