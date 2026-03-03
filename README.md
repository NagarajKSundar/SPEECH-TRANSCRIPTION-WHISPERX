# Speech Transcription POC (WhisperX + Speaker Diarization)

Proof-of-concept Python app for:
- real-time microphone transcription
- audio-file transcription
- optional speaker diarization (who spoke when)

It uses WhisperX for transcription and pyannote for diarization.

## Features

- Real-time transcription from microphone input
- File transcription (`.wav`, `.mp3`, and other formats supported by `soundfile`)
- Speaker diarization with Hugging Face + pyannote
- Word/segment timestamps
- CPU-first defaults (works without GPU)
- Multiple Whisper model sizes (`tiny` to `large-v3`)
- Rich console output and transcript summary table

## Project Files

- `main.py` - CLI entry point and app orchestration
- `audio_capture.py` - microphone capture and audio chunking
- `transcriber.py` - WhisperX transcription (with faster-whisper fallback)
- `diarizer.py` - pyannote diarization and merge logic
- `config.py` - runtime config defaults
- `torch_compat.py` - PyTorch 2.6+ compatibility patch
- `simple_test.py` - installation sanity checks
- `setup_windows.ps1` / `setup_windows.bat` - Windows setup helpers

## Requirements

- Python 3.10 or 3.11 recommended
- 8GB+ RAM (16GB recommended for bigger models)
- Microphone (for real-time mode)
- Hugging Face token (required only for diarization)

## Setup

### 1. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install whisperx
pip install -r requirements.txt
```

On Windows, you can also run:

```powershell
.\setup_windows.ps1
```

### 3. Configure Hugging Face token (for diarization)

1. Create/login to [Hugging Face](https://huggingface.co/)
2. Accept model terms:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Create a read token: https://huggingface.co/settings/tokens
4. Set token either:

```bash
export HF_TOKEN="your_token_here"
```

or in `.env` (copy from `.env.example`):

```env
HF_TOKEN=your_token_here
```

## Verify Installation

```bash
python simple_test.py
```

## Usage

### List input devices

```bash
python main.py --list-devices
```

### Real-time transcription

```bash
python main.py
```

Use a specific microphone:

```bash
python main.py --device 2
```

Disable diarization:

```bash
python main.py --no-diarization
```

### File transcription

```bash
python main.py --file meeting.wav
python main.py --file meeting.wav --output meeting_transcript.txt
```

### Model selection

```bash
python main.py --model small
python main.py --model large-v3
```

## CLI Options

```text
--file, -f           Audio file to transcribe
--device, -d         Input device index for microphone capture
--list-devices       List available audio input devices
--no-diarization     Disable speaker diarization
--output, -o         Output transcript file path
--model              tiny | base | small | medium | large-v3 | large-v2
--batch-size         Batch size argument (currently parsed but not wired in transcriber config)
```

## Output

By default, transcripts are saved to `transcript.txt` in this format:

```text
[SPEAKER_00] (0.0s - 3.5s):
  Hello, welcome to the meeting.
```

## Notes

- `torch_compat.py` is imported first in `main.py` to handle PyTorch 2.6+ loading behavior used by pyannote dependencies.
- If diarization is disabled or token is missing, transcription still works.

## Troubleshooting

- `HF_TOKEN` missing:
  - diarization fails, transcription continues without speaker labels
- `sounddevice` issues:
  - verify microphone permissions and installed audio backend
- `No module named ...`:
  - ensure virtual environment is activated and dependencies installed

