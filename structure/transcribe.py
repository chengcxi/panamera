import subprocess
import numpy as np
import tempfile
from scipy.signal import resample
from scipy.io import wavfile
import os
from pathlib import Path

# Path to your whisper.cpp installation
WHISPER_CPP_DIR = Path("/home/cheng/Downloads/AI/whisper.cpp")
WHISPER_BINARY = WHISPER_CPP_DIR / "build/bin/whisper-cli"
WHISPER_MODEL = WHISPER_CPP_DIR / "models/ggml-base.en.bin"

# Verify paths exist
assert WHISPER_BINARY.exists(), f"Whisper binary not found at {WHISPER_BINARY}"
assert WHISPER_MODEL.exists(), f"Whisper model not found at {WHISPER_MODEL}"

def transcribe_audio_piped(audio: np.ndarray, debug=False) -> str:
    """
    Transcribe by piping raw PCM audio to whisper.cpp via stdin.
    """
    # Ensure audio is float32 and normalized
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Ensure audio is in range [-1, 1]
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / np.max(np.abs(audio))
    
    # Convert float32 [-1,1] to int16 PCM (little-endian)
    audio_int16 = (audio * 32767).astype(np.int16)
    raw_pcm = audio_int16.tobytes()
    
    if debug:
        print(f"Audio stats: min={audio.min():.3f}, max={audio.max():.3f}, mean={audio.mean():.3f}")
        print(f"PCM bytes: {len(raw_pcm)} bytes, {len(audio_int16)} samples")
    
    # Call whisper.cpp with raw audio on stdin
    cmd = [
        str(WHISPER_BINARY),
        "-m", str(WHISPER_MODEL),
        "-f", "-",              # Read from stdin
        "-ac", "1",            # Mono
        "-ar", "16000",        # Sample rate
        "-c", "0",             # No captions
        "-t", "4",             # Threads
        "--no-timestamps",     # No timestamps
        "-ovtt",               # Output as VTT (cleaner output)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            input=raw_pcm,
            capture_output=True,
            timeout=10,
            check=False  # Don't raise on error, we'll check manually
        )
        
        if debug:
            if result.stderr:
                print(f"STDERR: {result.stderr.decode('utf-8', errors='ignore')[:500]}")
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            print(f"Whisper error (code {result.returncode}): {error_msg}")
            return ""
        
        # Parse output
        text = result.stdout.decode('utf-8', errors='ignore').strip()
        
        # Clean up VTT formatting if present
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            # Skip timestamp lines and VTT header
            if '-->' in line or line.startswith('WEBVTT'):
                continue
            if line.strip():
                clean_lines.append(line.strip())
        
        return ' '.join(clean_lines)
        
    except subprocess.TimeoutExpired:
        print("Whisper transcription timed out")
        return ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ""

def transcribe_audio_file(audio: np.ndarray, debug=False) -> str:
    """
    Transcribe using a temporary WAV file (more reliable).
    """
    import wave
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Write WAV file
        with wave.open(tmp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        if debug:
            print(f"WAV file created: {tmp_path}")
            print(f"Duration: {len(audio_int16)/16000:.2f} seconds")
        
        # Call whisper.cpp
        cmd = [
            str(WHISPER_BINARY),
            "-m", str(WHISPER_MODEL),
            "-f", tmp_path,
            "-t", "4",
            "--no-timestamps",
            "-ovtt",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
            check=False
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            print(f"Whisper error (code {result.returncode}): {error_msg}")
            return ""
        
        text = result.stdout.decode('utf-8', errors='ignore').strip()
        
        # Clean VTT formatting
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if '-->' in line or line.startswith('WEBVTT'):
                continue
            if line.strip():
                clean_lines.append(line.strip())
        
        return ' '.join(clean_lines)
        
    except Exception as e:
        print(f"Error in file transcription: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Use the file-based method first (more reliable)
transcribe = transcribe_audio_file

# Test with your file
test_wav_path = "output.wav"

if not os.path.exists(test_wav_path):
    print(f"❌ Error: The file '{test_wav_path}' does not exist!")
else:
    print(f"Reading file: {test_wav_path}...")
    
    # Read the wav file
    sample_rate, audio_data = wavfile.read(test_wav_path)
    
    print(f"Original: {sample_rate}Hz, shape={audio_data.shape}, dtype={audio_data.dtype}")
    
    # 1. Handle Stereo to Mono conversion
    if len(audio_data.shape) > 1:
        print("Converting stereo to mono...")
        audio_data = audio_data.mean(axis=1)
    
    # 2. Normalize to float32 [-1.0, 1.0]
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32767.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483647.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
    elif audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # 3. Check if audio is silent
    if np.abs(audio_data).max() < 0.01:
        print("⚠️ Warning: Audio appears to be very quiet or silent!")
    
    # 4. Resample to 16kHz if needed
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz...")
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_data = resample(audio_data, num_samples)
    
    print(f"Processed: {len(audio_data)} samples, {len(audio_data)/16000:.2f} seconds")
    
    # Try both methods
    print("\n--- Trying file-based transcription ---")
    text = transcribe_audio_file(audio_data, debug=True)
    print(f"Result: '{text}'")
    
    if not text:
        print("\n--- Trying piped transcription ---")
        text = transcribe_audio_piped(audio_data, debug=True)
        print(f"Result: '{text}'")