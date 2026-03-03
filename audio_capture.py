"""
Audio capture module for real-time microphone input
"""
import numpy as np
import sounddevice as sd
from typing import Callable, Optional
from dataclasses import dataclass
import threading
import queue
import time

from config import AudioConfig


@dataclass
class AudioChunk:
    """Represents a chunk of audio data"""
    data: np.ndarray
    timestamp: float
    duration: float


class AudioCapture:
    """
    Captures audio from the microphone in real-time.
    Provides audio chunks for processing.
    """
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio_queue: queue.Queue[AudioChunk] = queue.Queue()
        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None
        self.buffer: list[np.ndarray] = []
        self.buffer_start_time: float = 0
        self.lock = threading.Lock()
        
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Copy the audio data
        audio_data = indata.copy().flatten()
        
        with self.lock:
            if len(self.buffer) == 0:
                self.buffer_start_time = time.time()
            self.buffer.append(audio_data)
            
            # Calculate total buffer duration
            total_samples = sum(len(chunk) for chunk in self.buffer)
            buffer_duration = total_samples / self.config.sample_rate
            
            # If we have enough audio, create a chunk
            if buffer_duration >= self.config.chunk_duration:
                combined_audio = np.concatenate(self.buffer)
                chunk = AudioChunk(
                    data=combined_audio,
                    timestamp=self.buffer_start_time,
                    duration=buffer_duration
                )
                self.audio_queue.put(chunk)
                self.buffer = []
    
    def list_devices(self) -> list[dict]:
        """List available audio input devices"""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        return input_devices
    
    def start(self, device_index: Optional[int] = None) -> None:
        """Start capturing audio from the microphone"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.buffer = []
        
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        self.stream = sd.InputStream(
            device=device_index,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            callback=self._audio_callback,
            blocksize=int(self.config.sample_rate * 0.1),  # 100ms blocks
        )
        self.stream.start()
        print(f"Started audio capture at {self.config.sample_rate}Hz")
    
    def stop(self) -> Optional[AudioChunk]:
        """Stop capturing audio and return any remaining audio"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Return any remaining audio in the buffer
        with self.lock:
            if self.buffer:
                combined_audio = np.concatenate(self.buffer)
                chunk = AudioChunk(
                    data=combined_audio,
                    timestamp=self.buffer_start_time,
                    duration=len(combined_audio) / self.config.sample_rate
                )
                self.buffer = []
                return chunk
        
        return None
    
    def get_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """Get the next audio chunk from the queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def has_chunks(self) -> bool:
        """Check if there are chunks available"""
        return not self.audio_queue.empty()


def save_audio_to_file(audio_data: np.ndarray, sample_rate: int, 
                       filename: str) -> None:
    """Save audio data to a WAV file"""
    import soundfile as sf
    sf.write(filename, audio_data, sample_rate)
    print(f"Saved audio to {filename}")


def load_audio_from_file(filename: str) -> tuple[np.ndarray, int]:
    """Load audio from a file and resample to 16kHz if needed"""
    import soundfile as sf
    audio_data, sample_rate = sf.read(filename)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        from scipy import signal
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)
        sample_rate = 16000
    
    return audio_data.astype(np.float32), sample_rate
