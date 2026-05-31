import subprocess
import numpy as np
import tempfile
import os
from pathlib import Path

# Path to your whisper.cpp installation
WHISPER_CPP_DIR = Path("/home/cheng/Downloads/AI/whisper.cpp")  # Adjust this!
WHISPER_BINARY = WHISPER_CPP_DIR / "build/bin/whisper-cli"  # or just "main" on older versions
WHISPER_MODEL = WHISPER_CPP_DIR / "models/ggml-base.en.bin"  # Adjust model path

# Verify paths exist
assert WHISPER_BINARY.exists(), f"Whisper binary not found at {WHISPER_BINARY}"
assert WHISPER_MODEL.exists(), f"Whisper model not found at {WHISPER_MODEL}"

def transcribe_audio_piped(audio: np.ndarray) -> str:
    """
    Transcribe by piping raw PCM audio to whisper.cpp via stdin.
    This is the fastest method - no temp files needed.
    """
    # Convert float32 [-1,1] to int16 PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    raw_pcm = audio_int16.tobytes()
    
    # Call whisper.cpp with raw audio on stdin
    cmd = [
        str(WHISPER_BINARY),
        "-m", str(WHISPER_MODEL),
        "-f", "raw",          # Read raw PCM from stdin
        "-ar", "16000",       # Sample rate
        "-ac", "1",           # Mono
        "-t", "4",            # Threads (adjust based on your CPU)
        "--print-progress", "false",
        "--no-timestamps",    # We only need the text
        "-"                   # Read from stdin
    ]
    
    try:
        result = subprocess.run(
            cmd,
            input=raw_pcm,
            capture_output=True,
            timeout=10,  # Prevent hanging
            check=True
        )
        # Parse output - whisper.cpp outputs the transcribed text to stdout
        text = result.stdout.decode('utf-8').strip()
        # Remove any leading/trailing whitespace or timing info
        return text
        
    except subprocess.TimeoutExpired:
        print("Whisper transcription timed out")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"Whisper error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        return ""

# Alternative: Using temp WAV files (simpler but slower)
def transcribe_audio_file(audio: np.ndarray) -> str:
    """
    Transcribe using a temporary WAV file.
    Use this if piping raw PCM doesn't work with your whisper.cpp version.
    """
    import wave
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        
    try:
        # Write WAV file
        with wave.open(tmp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Call whisper.cpp
        cmd = [
            str(WHISPER_BINARY),
            "-m", str(WHISPER_MODEL),
            "-f", tmp_path,
            "-t", "4",
            "--print-progress", "false",
            "--no-timestamps"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
            check=True
        )
        text = result.stdout.decode('utf-8').strip()
        return text
        
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

# Choose one:
transcribe = transcribe_audio_piped  # or transcribe_audio_file