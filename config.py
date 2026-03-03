"""
Configuration settings for the Speech Transcription POC
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """Audio capture configuration"""
    sample_rate: int = 16000  # Whisper expects 16kHz
    channels: int = 1  # Mono audio
    chunk_duration: float = 5.0  # Process audio in 5-second chunks
    silence_threshold: float = 0.01  # Threshold for silence detection
    min_speech_duration: float = 0.5  # Minimum speech duration to process


@dataclass
class WhisperConfig:
    """Whisper transcription configuration"""
    model_size: str = "medium"  # Options: tiny, base, small, medium, large-v3
    device: str = "cpu"  # Use "cuda" if you have NVIDIA GPU
    compute_type: str = "int8"  # Use "float16" for GPU
    language: str = "en"  # Target language
    beam_size: int = 5
    vad_filter: bool = True  # Voice Activity Detection filter


@dataclass
class DiarizationConfig:
    """Speaker diarization configuration"""
    # You need to accept the pyannote terms on Hugging Face and get a token
    # Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
    hf_token: Optional[str] = None  # Set via environment variable HF_TOKEN
    min_speakers: int = 1
    max_speakers: int = 10
    

@dataclass
class AppConfig:
    """Main application configuration"""
    audio: AudioConfig
    whisper: WhisperConfig
    diarization: DiarizationConfig
    output_file: str = "transcript.txt"
    show_timestamps: bool = True
    

def load_config() -> AppConfig:
    """Load configuration with environment variable overrides"""
    hf_token = os.getenv("HF_TOKEN")
    
    # Check for GPU availability
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
    except ImportError:
        device = "cpu"
        compute_type = "int8"
    
    return AppConfig(
        audio=AudioConfig(),
        whisper=WhisperConfig(device=device, compute_type=compute_type),
        diarization=DiarizationConfig(hf_token=hf_token),
    )
