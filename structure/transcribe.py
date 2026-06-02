import subprocess
import numpy as np
import tempfile
from scipy.signal import resample
from scipy.io import wavfile
import sounddevice as sd
import os
from pathlib import Path
import time

# Path to your whisper.cpp installation
WHISPER_CPP_DIR = Path("/home/cheng/Downloads/AI/whisper.cpp")
WHISPER_BINARY = WHISPER_CPP_DIR / "build/bin/whisper-cli"
WHISPER_MODEL = WHISPER_CPP_DIR / "models/ggml-base.en.bin"

# Audio settings
RECORD_SAMPLE_RATE = 48000  # Your microphone's native rate
WHISPER_SAMPLE_RATE = 16000  # What Whisper expects

# Verify paths exist
assert WHISPER_BINARY.exists(), f"Whisper binary not found at {WHISPER_BINARY}"
assert WHISPER_MODEL.exists(), f"Whisper model not found at {WHISPER_MODEL}"

def record_audio(duration=5, device=None):
    """
    Record audio from microphone at 48kHz
    
    Args:
        duration: Recording duration in seconds
        device: Input device index (None = default)
    
    Returns:
        numpy array of audio data (float32, range [-1, 1]) at 48kHz
    """
    print(f"\n🎤 Recording at {RECORD_SAMPLE_RATE} Hz for {duration} seconds...")
    print("Speak now!")
    
    # Record audio at 48kHz
    audio_data = sd.rec(
        int(duration * RECORD_SAMPLE_RATE),
        samplerate=RECORD_SAMPLE_RATE,
        channels=1,  # Mono
        dtype='float32',
        device=device
    )
    
    # Wait for recording to complete
    sd.wait()
    
    # Flatten in case of multi-channel
    audio_data = audio_data.flatten()
    
    # Normalize to [-1, 1] range
    max_val = np.abs(audio_data).max()
    if max_val > 0:
        audio_data = audio_data / max_val
    
    print(f"✅ Recording complete! Captured {len(audio_data)} samples at {RECORD_SAMPLE_RATE} Hz")
    print(f"Audio level: min={audio_data.min():.3f}, max={audio_data.max():.3f}, mean={audio_data.mean():.3f}")
    
    return audio_data

def resample_for_whisper(audio_48k):
    """
    Resample audio from 48kHz to 16kHz for Whisper
    """
    print(f"🔄 Resampling from {RECORD_SAMPLE_RATE} Hz to {WHISPER_SAMPLE_RATE} Hz...")
    
    # Calculate number of samples at 16kHz
    num_samples_16k = int(len(audio_48k) * WHISPER_SAMPLE_RATE / RECORD_SAMPLE_RATE)
    
    # Resample using scipy.signal.resample
    audio_16k = resample(audio_48k, num_samples_16k)
    
    print(f"Resampled to {len(audio_16k)} samples at {WHISPER_SAMPLE_RATE} Hz")
    
    return audio_16k

def transcribe_audio_piped(audio: np.ndarray, debug=False) -> str:
    """
    Transcribe by piping raw PCM audio to whisper.cpp via stdin.
    Audio should already be at 16kHz.
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
        print(f"PCM bytes: {len(raw_pcm)} bytes, {len(audio_int16)} samples")
        print(f"Duration: {len(audio_int16) / WHISPER_SAMPLE_RATE:.2f} seconds")
    
    # Call whisper.cpp with raw audio on stdin
    cmd = [
        str(WHISPER_BINARY),
        "-m", str(WHISPER_MODEL),
        "-f", "-",              # Read from stdin
        "-ac", "1",            # Mono
        "-ar", str(WHISPER_SAMPLE_RATE),  # 16000 Hz
        "-t", "4",             # Threads
        "--no-timestamps",     # No timestamps
        "--print-progress", "false",
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            input=raw_pcm,
            capture_output=True,
            timeout=10,
            check=False
        )
        elapsed = time.time() - start_time
        
        if debug and result.stderr:
            stderr_text = result.stderr.decode('utf-8', errors='ignore')
            if stderr_text.strip():
                print(f"Whisper stderr: {stderr_text[:200]}")
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            print(f"Whisper error (code {result.returncode}): {error_msg}")
            return ""
        
        # Parse output
        text = result.stdout.decode('utf-8', errors='ignore').strip()
        
        if debug:
            print(f"Transcription took {elapsed:.2f} seconds")
        
        return text
        
    except subprocess.TimeoutExpired:
        print("Whisper transcription timed out")
        return ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ""

def list_audio_devices():
    """List available audio input devices"""
    print("\n📋 Available audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  Device {i}: {device['name']}")
            print(f"      Default samplerate: {device['default_samplerate']} Hz")
            print(f"      Max input channels: {device['max_input_channels']}")

def test_microphone_piped():
    """Test piped transcription with microphone input"""
    
    # List available devices
    list_audio_devices()
    
    # Let user select device (or use default)
    try:
        device_input = input("\nEnter device number (or press Enter for default): ").strip()
        device = int(device_input) if device_input else None
    except ValueError:
        print("Invalid input, using default device")
        device = None
    
    # Recording settings
    duration = float(input("Recording duration in seconds (default 5): ").strip() or "5")
    
    # Record audio at 48kHz
    print("\n" + "="*50)
    audio_48k = record_audio(duration=duration, device=device)
    
    # Check if recording captured any sound
    if np.abs(audio_48k).max() < 0.01:
        print("⚠️ Warning: Very low audio level detected! Check your microphone.")
        retry = input("Retry? (y/n): ").lower()
        if retry == 'y':
            return test_microphone_piped()
    
    # Resample to 16kHz for Whisper
    audio_16k = resample_for_whisper(audio_48k)
    
    # Transcribe
    print("\n🎙️ Transcribing with whisper.cpp (piped)...")
    print("-" * 50)
    text = transcribe_audio_piped(audio_16k, debug=True)
    print("-" * 50)
    
    if text:
        print(f"📝 Transcription: {text}")
    else:
        print("❌ No transcription produced. Possible issues:")
        print("   - Microphone volume too low")
        print("   - No speech detected")
        print("   - Background noise or silence")
        print("   - Try speaking louder or closer to the microphone")
    
    return text

def continuous_mode():
    """Continuous recording and transcription mode"""
    print("\n" + "="*60)
    print("🎙️ CONTINUOUS MODE - Speak, then pause to transcribe")
    print("="*60)
    
    # List devices
    list_audio_devices()
    
    try:
        device_input = input("\nEnter device number (or press Enter for default): ").strip()
        device = int(device_input) if device_input else None
    except ValueError:
        device = None
    
    print("\nInstructions:")
    print("  - Speak naturally, recording happens in chunks")
    print("  - Press Enter to record each chunk")
    print("  - Press Ctrl+C to stop")
    print()
    
    chunk_duration = 3  # seconds per chunk
    
    try:
        while True:
            input("Press Enter to start recording (or Ctrl+C to quit)...")
            
            # Record at 48kHz
            audio_48k = record_audio(duration=chunk_duration, device=device)
            
            # Check if audio has meaningful content
            if np.abs(audio_48k).max() < 0.02:
                print("⚠️ No significant audio detected, skipping...")
                continue
            
            # Resample to 16kHz
            audio_16k = resample_for_whisper(audio_48k)
            
            # Transcribe
            print("Transcribing...", end=" ", flush=True)
            text = transcribe_audio_piped(audio_16k, debug=False)
            
            if text:
                print(f"\n📝 {text}\n")
            else:
                print("No speech detected\n")
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

def quick_test():
    """Quick 3-second test"""
    print("\nQuick test - 3 second recording at 48kHz")
    try:
        # Get default device info
        default_device = sd.query_devices(None, 'input')
        print(f"\nUsing default input device: {default_device['name']}")
        
        # Record at 48kHz
        audio_48k = record_audio(duration=3, device=None)
        
        if np.abs(audio_48k).max() < 0.02:
            print("\n⚠️ Low audio level! Please check:")
            print("   - Microphone permissions")
            print("   - Microphone volume settings")
            print("   - Physical microphone connection")
            return
        
        # Resample and transcribe
        audio_16k = resample_for_whisper(audio_48k)
        text = transcribe_audio_piped(audio_16k, debug=True)
        
        if text:
            print(f"\n📝 Result: {text}")
        else:
            print("\n❌ No transcription - speak louder or check microphone")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if microphone is connected: arecord -l")
        print("2. Test microphone: arecord -d 3 -r 48000 test.wav")
        print("3. Playback test: aplay test.wav")

def transcribe_file(file_path: str) -> str:
    """
    Transcribe an audio file using whisper.cpp.
    Supports any format ffmpeg can decode.
    """

    cmd = [
        str(WHISPER_BINARY),
        "-m", str(WHISPER_MODEL),
        "-f", file_path,
        "-t", "4",
        "-l", "en",
        "--no-timestamps",
        "--print-progress", "false"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            check=True
        )

        text = result.stdout.decode("utf-8").strip()
        text = text.replace("[BLANK_AUDIO]", "").strip()

        return text

    except subprocess.CalledProcessError as e:
        print(f"Whisper error: {e}")
        print(e.stderr.decode("utf-8", errors="ignore"))
        return ""

    except Exception as e:
        print(f"Error: {e}")
        return ""

if __name__ == "__main__":
    print("🎤 MICROPHONE PIPED TRANSCRIPTION TEST (48kHz → 16kHz)")
    print("="*60)
    print(f"Recording at: {RECORD_SAMPLE_RATE} Hz")
    print(f"Whisper expects: {WHISPER_SAMPLE_RATE} Hz")
    print()
    
    # Check if sounddevice is installed
    try:
        import sounddevice
    except ImportError:
        print("❌ sounddevice not installed! Please run: pip install sounddevice")
        exit(1)
    
    # Ask user for mode
    print("Select mode:")
    print("  1. Single recording (test once)")
    print("  2. Continuous mode (record, transcribe, repeat)")
    print("  3. Quick test (3-second recording)")
    print("  4. Transcribe audio file")
    
    mode = input("\nChoice (1/2/3/4): ").strip()
    
    if mode == "2":
        continuous_mode()
    elif mode == "3":
        quick_test()
    elif mode == "4":
        path = input("\nAudio file path: ").strip()

        if not os.path.exists(path):
            print("❌ File not found")
            exit(1)

        print(f"\n🎙️ Transcribing: {path}")
        print("-" * 50)

        start = time.time()
        text = transcribe_file(path)
        elapsed = time.time() - start

        print("-" * 50)

        if text:
            print(f"\n📝 Transcription:\n{text}")
            print(f"\n⏱️ Time: {elapsed:.2f}s")
        else:
            print("\n❌ No transcription produced")
    else:
        test_microphone_piped()