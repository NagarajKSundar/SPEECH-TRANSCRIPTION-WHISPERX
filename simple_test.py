"""
Simple test script to verify the installation works
Run this first to check if all dependencies are installed correctly
"""
import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...\n")
    
    dependencies = [
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("faster_whisper", "faster-whisper"),
        ("pyannote.audio", "pyannote.audio"),
        ("rich", "rich"),
    ]
    
    all_ok = True
    for import_name, package_name in dependencies:
        try:
            __import__(import_name)
            print(f"  [OK] {package_name}")
        except ImportError as e:
            print(f"  [MISSING] {package_name} - pip install {package_name}")
            all_ok = False
    
    return all_ok


def check_audio_devices():
    """Check available audio devices"""
    print("\nChecking audio devices...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        print(f"  Found {len(input_devices)} input device(s)")
        for i, d in enumerate(input_devices):
            print(f"    {i}: {d['name']}")
        return True
    except Exception as e:
        print(f"  [ERROR] Could not query audio devices: {e}")
        return False


def check_whisper_model():
    """Test loading a tiny Whisper model"""
    print("\nTesting Whisper model loading (tiny model)...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("  [OK] Whisper model loads correctly")
        del model
        return True
    except Exception as e:
        print(f"  [ERROR] Could not load Whisper model: {e}")
        return False


def check_hf_token():
    """Check if Hugging Face token is set"""
    import os
    print("\nChecking Hugging Face token...")
    token = os.getenv("HF_TOKEN")
    if token:
        print(f"  [OK] HF_TOKEN is set (length: {len(token)})")
        return True
    else:
        print("  [WARNING] HF_TOKEN not set - speaker diarization will not work")
        print("  Set it with: $env:HF_TOKEN = 'your_token_here' (PowerShell)")
        return False


def test_transcription():
    """Test transcription with a simple audio sample"""
    print("\nTesting transcription with generated audio...")
    try:
        import numpy as np
        from faster_whisper import WhisperModel
        
        # Generate 2 seconds of silence (just to test the pipeline)
        sample_rate = 16000
        duration = 2
        audio = np.zeros(sample_rate * duration, dtype=np.float32)
        
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio, language="en")
        
        # Consume the generator
        list(segments)
        
        print("  [OK] Transcription pipeline works")
        del model
        return True
    except Exception as e:
        print(f"  [ERROR] Transcription test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Speech Transcription POC - Installation Test")
    print("=" * 60)
    
    results = []
    
    # Check dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # Check audio devices
    results.append(("Audio Devices", check_audio_devices()))
    
    # Check HF token
    results.append(("HF Token", check_hf_token()))
    
    # Test Whisper
    results.append(("Whisper Model", check_whisper_model()))
    
    # Test transcription
    results.append(("Transcription", test_transcription()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed and name != "HF Token":  # HF Token is optional
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! You can now run:")
        print("  python main.py --list-devices")
        print("  python main.py")
    else:
        print("\n[WARNING] Some tests failed. Please fix the issues above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
