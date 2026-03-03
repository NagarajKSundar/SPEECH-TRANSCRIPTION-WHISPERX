"""
Real-time Speech Transcription with Speaker Diarization
Main application that combines audio capture, transcription, and diarization
"""
# IMPORTANT: Import torch_compat FIRST to patch torch.load for PyTorch 2.6+ compatibility
import torch_compat  # noqa: F401 - patches torch.load at import time

import os
import sys
import time
import threading
import argparse
from datetime import datetime
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from config import load_config, AppConfig
from audio_capture import AudioCapture, AudioChunk, save_audio_to_file
from transcriber import Transcriber, TranscriptionResult
from diarizer import Diarizer, DiarizationResult, merge_transcription_with_diarization


console = Console()


class TranscriptionApp:
    """
    Main application for real-time speech transcription with speaker diarization
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.audio_capture = AudioCapture(config.audio)
        self.transcriber = Transcriber(config.whisper)
        self.diarizer: Optional[Diarizer] = None
        self.transcript: list[dict] = []
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.enable_diarization = True
        
    def initialize(self, enable_diarization: bool = True) -> None:
        """Initialize models"""
        self.enable_diarization = enable_diarization
        
        console.print("[bold blue]Initializing models...[/bold blue]")
        
        # Load Whisper model
        with console.status("[bold green]Loading Whisper transcription model..."):
            self.transcriber.load_model()
        console.print("[green]Whisper model loaded![/green]")
        
        # Load diarization model if enabled
        if enable_diarization:
            try:
                with console.status("[bold green]Loading speaker diarization model..."):
                    self.diarizer = Diarizer(self.config.diarization)
                    self.diarizer.load_model()
                console.print("[green]Speaker diarization model loaded![/green]")
            except ValueError as e:
                console.print(f"[yellow]Warning: {e}[/yellow]")
                console.print("[yellow]Continuing without speaker diarization...[/yellow]")
                self.enable_diarization = False
                self.diarizer = None
        
        console.print("[bold green]All models initialized![/bold green]\n")
    
    def _process_audio_chunk(self, chunk: AudioChunk) -> None:
        """Process a single audio chunk"""
        # Transcribe
        trans_result = self.transcriber.transcribe(chunk.data)
        
        if not trans_result.segments:
            return
        
        # Diarize if enabled
        if self.enable_diarization and self.diarizer:
            diar_result = self.diarizer.diarize(chunk.data)
            merged = merge_transcription_with_diarization(
                trans_result.segments,
                diar_result.segments,
            )
        else:
            # No diarization - just use transcription
            merged = [
                {
                    "speaker": "Speaker",
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                }
                for seg in trans_result.segments
            ]
        
        # Add timestamp offset
        for item in merged:
            item["start"] += chunk.timestamp
            item["end"] += chunk.timestamp
            item["timestamp"] = datetime.fromtimestamp(chunk.timestamp).strftime("%H:%M:%S")
        
        # Add to transcript
        self.transcript.extend(merged)
        
        # Print new segments
        for item in merged:
            self._print_segment(item)
    
    def _print_segment(self, segment: dict) -> None:
        """Print a transcript segment"""
        speaker = segment["speaker"]
        text = segment["text"]
        timestamp = segment.get("timestamp", "")
        
        # Color code speakers
        speaker_colors = {
            "SPEAKER_00": "cyan",
            "SPEAKER_01": "magenta",
            "SPEAKER_02": "yellow",
            "SPEAKER_03": "green",
            "SPEAKER_04": "blue",
        }
        color = speaker_colors.get(speaker, "white")
        
        if self.config.show_timestamps:
            console.print(f"[dim]{timestamp}[/dim] [{color}][{speaker}][/{color}]: {text}")
        else:
            console.print(f"[{color}][{speaker}][/{color}]: {text}")
    
    def _processing_loop(self) -> None:
        """Main processing loop running in a separate thread"""
        while self.is_running:
            chunk = self.audio_capture.get_chunk(timeout=0.5)
            if chunk:
                try:
                    self._process_audio_chunk(chunk)
                except Exception as e:
                    console.print(f"[red]Error processing audio: {e}[/red]")
    
    def start_realtime(self, device_index: Optional[int] = None) -> None:
        """Start real-time transcription"""
        self.is_running = True
        self.transcript = []
        
        # Start audio capture
        self.audio_capture.start(device_index)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        console.print("\n[bold green]Recording started![/bold green]")
        console.print("[dim]Press Ctrl+C to stop...[/dim]\n")
        console.print("-" * 60)
    
    def stop(self) -> None:
        """Stop real-time transcription"""
        self.is_running = False
        
        # Stop audio capture and get remaining audio
        remaining = self.audio_capture.stop()
        
        # Wait for processing thread
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        # Process remaining audio
        if remaining and remaining.duration > 0.5:
            console.print("\n[dim]Processing remaining audio...[/dim]")
            self._process_audio_chunk(remaining)
        
        console.print("\n" + "-" * 60)
        console.print("[bold green]Recording stopped![/bold green]")
    
    def transcribe_file(self, filepath: str) -> list[dict]:
        """Transcribe an audio file with speaker diarization"""
        from audio_capture import load_audio_from_file
        
        console.print(f"\n[bold blue]Processing file: {filepath}[/bold blue]")
        
        # Load audio
        with console.status("[bold green]Loading audio file..."):
            audio_data, sample_rate = load_audio_from_file(filepath)
        console.print(f"[green]Loaded {len(audio_data)/sample_rate:.1f}s of audio[/green]")
        
        # Transcribe
        with console.status("[bold green]Transcribing..."):
            trans_result = self.transcriber.transcribe(audio_data, sample_rate)
        console.print(f"[green]Transcription complete: {len(trans_result.segments)} segments[/green]")
        
        # Diarize if enabled
        if self.enable_diarization and self.diarizer:
            with console.status("[bold green]Identifying speakers..."):
                diar_result = self.diarizer.diarize(audio_data, sample_rate)
            console.print(f"[green]Found {diar_result.num_speakers} speakers[/green]")
            
            # Merge results
            merged = merge_transcription_with_diarization(
                trans_result.segments,
                diar_result.segments,
            )
            
            # Renumber speakers by order of first appearance
            merged = self._renumber_speakers_by_appearance(merged)
        else:
            merged = [
                {
                    "speaker": "Speaker",
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                }
                for seg in trans_result.segments
            ]
        
        self.transcript = merged
        return merged
    
    def _renumber_speakers_by_appearance(self, segments: list[dict]) -> list[dict]:
        """Renumber speakers so SPEAKER_00 is the first to appear, SPEAKER_01 is second, etc."""
        if not segments:
            return segments
        
        # Find order of first appearance
        speaker_order = []
        for seg in segments:
            speaker = seg["speaker"]
            if speaker not in speaker_order and speaker != "Unknown":
                speaker_order.append(speaker)
        
        # Create mapping from old speaker ID to new speaker ID
        speaker_mapping = {}
        for idx, old_speaker in enumerate(speaker_order):
            speaker_mapping[old_speaker] = f"SPEAKER_{idx:02d}"
        
        # Apply mapping
        for seg in segments:
            old_speaker = seg["speaker"]
            if old_speaker in speaker_mapping:
                seg["speaker"] = speaker_mapping[old_speaker]
        
        return segments
    
    def save_transcript(self, filepath: Optional[str] = None) -> str:
        """Save transcript to a file"""
        filepath = filepath or self.config.output_file
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("TRANSCRIPT WITH SPEAKER DIARIZATION\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            current_speaker = None
            for item in self.transcript:
                speaker = item["speaker"]
                text = item["text"]
                start = item.get("start", 0)
                end = item.get("end", 0)
                
                if speaker != current_speaker:
                    f.write(f"\n[{speaker}] ({start:.1f}s - {end:.1f}s):\n")
                    current_speaker = speaker
                
                f.write(f"  {text}\n")
        
        console.print(f"\n[green]Transcript saved to: {filepath}[/green]")
        return filepath
    
    def print_summary(self) -> None:
        """Print a summary of the transcript"""
        if not self.transcript:
            console.print("[yellow]No transcript available[/yellow]")
            return
        
        # Count speakers
        speakers = {}
        for item in self.transcript:
            speaker = item["speaker"]
            if speaker not in speakers:
                speakers[speaker] = {"count": 0, "words": 0}
            speakers[speaker]["count"] += 1
            speakers[speaker]["words"] += len(item["text"].split())
        
        # Create summary table
        table = Table(title="Transcript Summary")
        table.add_column("Speaker", style="cyan")
        table.add_column("Segments", justify="right")
        table.add_column("Words", justify="right")
        
        for speaker, stats in speakers.items():
            table.add_row(speaker, str(stats["count"]), str(stats["words"]))
        
        console.print("\n")
        console.print(table)


def list_audio_devices() -> None:
    """List available audio input devices"""
    from audio_capture import AudioCapture
    from config import AudioConfig
    
    capture = AudioCapture(AudioConfig())
    devices = capture.list_devices()
    
    table = Table(title="Available Audio Input Devices")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Channels", justify="right")
    table.add_column("Sample Rate", justify="right")
    
    for device in devices:
        table.add_row(
            str(device["index"]),
            device["name"],
            str(device["channels"]),
            f"{device['sample_rate']:.0f} Hz",
        )
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Speech Transcription with Speaker Diarization"
    )
    parser.add_argument(
        "--file", "-f",
        help="Audio file to transcribe (instead of real-time)",
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        help="Audio input device index (use --list-devices to see available)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization (transcription only)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for transcript",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v2"],
        default="medium",
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for WhisperX transcription (default: 16, higher = faster but more memory)",
    )
    
    args = parser.parse_args()
    
    # List devices and exit
    if args.list_devices:
        list_audio_devices()
        return
    
    # Load config
    config = load_config()
    
    # Override model size if specified
    if args.model:
        config.whisper.model_size = args.model
    
    # Override output file if specified
    if args.output:
        config.output_file = args.output
    
    # Create app
    app = TranscriptionApp(config)
    
    # Initialize models
    enable_diarization = not args.no_diarization
    app.initialize(enable_diarization=enable_diarization)
    
    try:
        if args.file:
            # File mode
            app.transcribe_file(args.file)
            app.print_summary()
            
            # Print transcript
            console.print("\n[bold]Full Transcript:[/bold]")
            console.print("-" * 60)
            for item in app.transcript:
                app._print_segment(item)
            
            # Save transcript
            app.save_transcript()
        else:
            # Real-time mode
            console.print(Panel.fit(
                "[bold]Real-time Speech Transcription[/bold]\n"
                "Speak into your microphone. Press Ctrl+C to stop.",
                title="Speech-to-Text POC",
            ))
            
            app.start_realtime(device_index=args.device)
            
            # Wait for Ctrl+C
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            
            app.stop()
            app.print_summary()
            app.save_transcript()
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
