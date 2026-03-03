"""
WhisperX-based transcription module for fast transcription with word-level timestamps
Falls back to faster-whisper if whisperx is not available
"""
import numpy as np
import os
import torch
from typing import Optional
from dataclasses import dataclass

from config import WhisperConfig

@dataclass
class TranscriptionSegment:
    """Represents a transcribed segment"""
    text: str
    start: float
    end: float
    confidence: float
    words: list = None


@dataclass
class TranscriptionResult:
    """Result of transcription"""
    segments: list[TranscriptionSegment]
    full_text: str
    language: str
    duration: float


class Transcriber:
    """
    Transcribes audio using WhisperX (optimized Whisper with word-level timestamps)
    Falls back to faster-whisper if whisperx is not available
    """
    
    def __init__(self, config: WhisperConfig):
        self.config = config
        self.model = None
        self.use_whisperx = True
        self.align_model = None
        self.align_metadata = None
        
    def load_model(self) -> None:
        """Load the Whisper model"""
        try:
            import whisperx
            self.use_whisperx = True
            
            device = self.config.device
            compute_type = self.config.compute_type
            
            print(f"Loading WhisperX model '{self.config.model_size}' on {device}...")
            self.model = whisperx.load_model(
                self.config.model_size,
                device=device,
                compute_type=compute_type,
            )
            print("WhisperX model loaded successfully!")
            
        except ImportError:
            print("WhisperX not available, falling back to faster-whisper...")
            self.use_whisperx = False
            from faster_whisper import WhisperModel
            
            print(f"Loading Whisper model '{self.config.model_size}' on {self.config.device}...")
            self.model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            print("Whisper model loaded successfully!")
    
    def transcribe(self, audio_data: np.ndarray, 
                   sample_rate: int = 16000) -> TranscriptionResult:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate of the audio (should be 16000)
            
        Returns:
            TranscriptionResult with segments and full text
        """
        if self.model is None:
            self.load_model()
        
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio if needed
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        if self.use_whisperx:
            return self._transcribe_whisperx(audio_data, sample_rate)
        else:
            return self._transcribe_faster_whisper(audio_data, sample_rate)
    
    def _transcribe_whisperx(self, audio_data: np.ndarray, 
                              sample_rate: int) -> TranscriptionResult:
        """Transcribe using WhisperX"""
        import whisperx
        
        # WhisperX expects audio dict or file path
        # For in-memory audio, we need to pass it correctly
        audio_dict = {"waveform": audio_data, "sample_rate": sample_rate}
        
        # Transcribe with batched inference (faster)
        result = self.model.transcribe(
            audio_data,
            batch_size=16,
            language=self.config.language if self.config.language != "auto" else None,
        )
        
        # Get detected language
        detected_language = result.get("language", self.config.language)
        
        # Load alignment model for word-level timestamps (lazy load)
        if self.align_model is None and detected_language:
            try:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.config.device,
                )
            except Exception as e:
                print(f"Warning: Could not load alignment model: {e}")
        
        # Align whisper output for word-level timestamps
        if self.align_model is not None:
            try:
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio_data,
                    self.config.device,
                    return_char_alignments=False,
                )
            except Exception as e:
                print(f"Warning: Alignment failed: {e}")
        
        # Convert to our format
        segments = []
        full_text_parts = []
        
        for seg in result.get("segments", []):
            words = seg.get("words", [])
            segment = TranscriptionSegment(
                text=seg.get("text", "").strip(),
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                confidence=seg.get("score", 0.0) if "score" in seg else 0.0,
                words=words,
            )
            segments.append(segment)
            full_text_parts.append(seg.get("text", "").strip())
        
        duration = len(audio_data) / sample_rate
        
        return TranscriptionResult(
            segments=segments,
            full_text=" ".join(full_text_parts),
            language=detected_language,
            duration=duration,
        )
    
    def _transcribe_faster_whisper(self, audio_data: np.ndarray, 
                                    sample_rate: int) -> TranscriptionResult:
        """Transcribe using faster-whisper (fallback)"""
        segments_generator, info = self.model.transcribe(
            audio_data,
            language=self.config.language if self.config.language != "auto" else None,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )
        
        segments = []
        full_text_parts = []
        
        for segment in segments_generator:
            seg = TranscriptionSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                confidence=segment.avg_logprob,
            )
            segments.append(seg)
            full_text_parts.append(segment.text.strip())
        
        duration = len(audio_data) / sample_rate
        
        return TranscriptionResult(
            segments=segments,
            full_text=" ".join(full_text_parts),
            language=info.language if info.language else self.config.language,
            duration=duration,
        )
    
    def transcribe_file(self, filepath: str) -> TranscriptionResult:
        """Transcribe audio from a file"""
        from audio_capture import load_audio_from_file
        
        audio_data, sample_rate = load_audio_from_file(filepath)
        return self.transcribe(audio_data, sample_rate)
