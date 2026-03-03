"""
Speaker diarization module using pyannote.audio
Identifies WHO is speaking WHEN in the audio
"""
import numpy as np
from typing import Optional
from dataclasses import dataclass
import torch
import os

from config import DiarizationConfig

@dataclass
class SpeakerSegment:
    """Represents a segment where a specific speaker is talking"""
    speaker: str
    start: float
    end: float


@dataclass
class DiarizationResult:
    """Result of speaker diarization"""
    segments: list[SpeakerSegment]
    num_speakers: int
    duration: float


class Diarizer:
    """
    Performs speaker diarization using pyannote.audio
    Identifies different speakers in the audio
    """
    
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.pipeline = None
        
    def load_model(self) -> None:
        """Load the pyannote speaker diarization pipeline"""
        from pyannote.audio import Pipeline
        
        # Get HF token from config or environment
        hf_token = self.config.hf_token or os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError(
                "Hugging Face token required for pyannote.audio.\n"
                "1. Create account at https://huggingface.co\n"
                "2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Create token at https://huggingface.co/settings/tokens\n"
                "4. Set HF_TOKEN environment variable or pass to config"
            )
        
        print("Loading pyannote speaker diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
            print("Diarization pipeline loaded on GPU")
        else:
            print("Diarization pipeline loaded on CPU")
    
    def diarize(self, audio_data: np.ndarray, 
                sample_rate: int = 16000) -> DiarizationResult:
        """
        Perform speaker diarization on audio data
        
        Args:
            audio_data: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate of the audio
            
        Returns:
            DiarizationResult with speaker segments
        """
        if self.pipeline is None:
            self.load_model()
        
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Create a dictionary format that pyannote expects
        audio_dict = {
            "waveform": torch.tensor(audio_data).unsqueeze(0),
            "sample_rate": sample_rate,
        }
        
        # Run diarization
        diarization = self.pipeline(
            audio_dict,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )
        
        # Extract segments - handle different pyannote.audio versions
        segments = []
        speakers = set()
        
        # Try multiple ways to access the annotation
        annotation = None
        
        # Method 1: pyannote.audio 3.3.x DiarizeOutput has speaker_diarization attribute
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        # Method 2: Direct annotation attribute (older versions)
        elif hasattr(diarization, 'annotation'):
            annotation = diarization.annotation
        # Method 3: Direct access if it's already an Annotation
        elif hasattr(diarization, 'itertracks'):
            annotation = diarization
        # Method 4: It might be a tuple/namedtuple with annotation as first element
        elif isinstance(diarization, tuple) and len(diarization) > 0:
            annotation = diarization[0]
        
        # Iterate over tracks if we have an annotation
        if annotation is not None and hasattr(annotation, 'itertracks'):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                seg = SpeakerSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end,
                )
                segments.append(seg)
                speakers.add(speaker)
        elif annotation is not None:
            # Fallback: try to iterate directly if it's iterable
            try:
                for item in annotation:
                    if hasattr(item, 'start') and hasattr(item, 'end'):
                        speaker = getattr(item, 'speaker', getattr(item, 'label', 'Speaker'))
                        seg = SpeakerSegment(
                            speaker=str(speaker),
                            start=item.start,
                            end=item.end,
                        )
                        segments.append(seg)
                        speakers.add(str(speaker))
            except TypeError:
                pass
        
        duration = len(audio_data) / sample_rate
        
        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers),
            duration=duration,
        )
    
    def diarize_file(self, filepath: str) -> DiarizationResult:
        """Perform diarization on an audio file"""
        from audio_capture import load_audio_from_file
        
        audio_data, sample_rate = load_audio_from_file(filepath)
        return self.diarize(audio_data, sample_rate)


def merge_transcription_with_diarization(
    transcription_segments: list,
    diarization_segments: list[SpeakerSegment],
) -> list[dict]:
    """
    Merge transcription segments with speaker diarization
    to create a transcript with speaker labels
    
    Returns list of dicts with: speaker, text, start, end
    """
    merged = []
    
    for trans_seg in transcription_segments:
        # Find the speaker for this transcription segment
        # Use the speaker who has the most overlap with this segment
        best_speaker = "Unknown"
        best_overlap = 0
        
        for diar_seg in diarization_segments:
            # Calculate overlap
            overlap_start = max(trans_seg.start, diar_seg.start)
            overlap_end = min(trans_seg.end, diar_seg.end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg.speaker
        
        merged.append({
            "speaker": best_speaker,
            "text": trans_seg.text,
            "start": trans_seg.start,
            "end": trans_seg.end,
        })
    
    return merged
