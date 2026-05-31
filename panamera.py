import subprocess
import threading
import time
import numpy as np
from PIL import Image
import cv2
import sounddevice as sd
from scipy.signal import resample
import torch
from pathlib import Path
import logging

# Configuration
WHISPER_CPP_DIR = Path("/home/cheng/Downloads/AI/whisper.cpp")
WHISPER_BINARY = WHISPER_CPP_DIR / "build/bin/whisper-cli"
WHISPER_MODEL = WHISPER_CPP_DIR / "models/ggml-base.en.bin"

HARDWARE_SAMPLE_RATE = 44100  # Your working rate
TARGET_SAMPLE_RATE = 16000    # Required by VAD and Whisper

# Camera triggers
CAMERA_TRIGGERS = ["look", "see", "camera", "view", "what do you see"]

# GPU optimization
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# 1. MiniCPM-V Setup
print("Loading MiniCPM-V (4-bit)...")
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

model_id = "openbmb/MiniCPM-V-4.6"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("✓ MiniCPM-V loaded\n")

def minicpm_chat(user_text: str, image: Image = None) -> str:
    """Send query to MiniCPM-V and get text response."""
    messages = []
    if image is None:
        messages.append({"role": "user", "content": user_text})
    else:
        messages.append({"role": "user", "content": f"<image>\n{user_text}"})

    response, _ = model.chat(
        image=image,
        msgs=messages,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
    )
    return response

# 2. Silero VAD Setup
print("Loading Silero VAD...")
from silero_vad import load_silero_vad, get_speech_timestamps

vad_model = load_silero_vad()

def record_when_speaking(
    silence_threshold_sec=0.8,
    max_duration_sec=15,
) -> np.ndarray | None:
    """
    Listens to microphone using zero-copy fast downsampling inside the callback.
    """
    print("🎤 Listening...")
    audio_chunks = []
    is_speaking = False
    silence_frames = 0
    
    # Calculate limits based on how many 512-frame hardware blocks we expect
    silence_limit = int(silence_threshold_sec * HARDWARE_SAMPLE_RATE / 512)

    def callback(indata, frames, time, status):
        nonlocal is_speaking, silence_frames, audio_chunks
        if status:
            print(f"Audio status flag triggered: {status}")

        # Deep copy the incoming raw data immediately to avoid buffer overwrite
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        audio_chunks.append(mono)

        # Process VAD on the last ~0.5 seconds of accumulated audio history
        context_blocks = int((HARDWARE_SAMPLE_RATE * 0.5) / 512)
        if len(audio_chunks) >= context_blocks:
            recent_hw = np.concatenate(audio_chunks[-context_blocks:])
            
            # FAST DOWNSAMPLING: Slice step approximation instead of heavy SciPy resample
            # 44100 / 16000 is roughly a step factor of 2.756
            step = HARDWARE_SAMPLE_RATE / TARGET_SAMPLE_RATE
            indices = np.arange(0, len(recent_hw), step).astype(np.int32)
            recent_16k = recent_hw[indices]
            
            # Run the VAD inference
            speech_ts = get_speech_timestamps(recent_16k, vad_model, sampling_rate=TARGET_SAMPLE_RATE)

            if len(speech_ts) > 0:
                is_speaking = True
                silence_frames = 0
            else:
                if is_speaking:
                    silence_frames += 1

    # Open stream channel
    with sd.InputStream(
        samplerate=HARDWARE_SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=callback,
        blocksize=512,
    ):
        while True:
            sd.sleep(100)
            if is_speaking and silence_frames >= silence_limit:
                break
            if len(audio_chunks) > (max_duration_sec * HARDWARE_SAMPLE_RATE // 512):
                break

    if not is_speaking or len(audio_chunks) < 2:
        print("No speech detected.")
        return None

    # Stitch the full hardware audio recording together
    full_audio = np.concatenate(audio_chunks)
    
    # Use standard SciPy resampling ONCE on the full file, safely outside the callback thread
    num_16k = int(len(full_audio) * TARGET_SAMPLE_RATE / HARDWARE_SAMPLE_RATE)
    full_audio_16k = resample(full_audio, num_16k)
    
    return full_audio_16k

print("✓ Silero VAD loaded with fast-slicing callbacks\n")

# 3. Whisper.cpp Setup
print(f"Setting up Whisper.cpp...")
print(f"  Binary: {WHISPER_BINARY}")
print(f"  Model: {WHISPER_MODEL}")

test_cmd = [str(WHISPER_BINARY), "--help"]
result = subprocess.run(test_cmd, capture_output=True)
if result.returncode != 0:
    raise RuntimeError(f"whisper.cpp not working: {result.stderr.decode()}")

def transcribe_audio(audio: np.ndarray) -> str:
    """Transcribe audio using native whisper.cpp (expects 16kHz audio)."""
    # Convert float32 to int16 PCM
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    raw_pcm = audio_int16.tobytes()
    
    cmd = [
        str(WHISPER_BINARY),
        "-m", str(WHISPER_MODEL),
        "-f", "raw",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-t", "4",
        "-l", "en",
        "--print-progress", "false",
        "--no-timestamps",
        "-"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            input=raw_pcm,
            capture_output=True,
            timeout=15,
            check=True
        )
        text = result.stdout.decode('utf-8').strip()
        text = text.replace('[BLANK_AUDIO]', '').strip()
        return text
    except subprocess.TimeoutExpired:
        print("⚠️  Transcription timed out")
        return ""
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown error"
        print(f"⚠️  Whisper error: {error_msg}")
        return ""

print("✓ Whisper.cpp ready\n")

# 4. VoxCPM Setup
print("Loading VoxCPM...")
from voxcpm import VoxCPM

tts_model_id = "openbmb/VoxCPM-0.5B"
tts_model = VoxCPM.from_pretrained(tts_model_id)

def speak(text: str):
    """Generates audio for input text, resamples to standard hardware rates, and plays it."""
    try:
        # Pre-warm runtime generation check
        wav = tts_model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
        
        # Audio array physical hardware interpolation to prevent ALSA rate crash
        native_rate = 16000
        target_rate = 44100
        num_samples = int(len(wav) * target_rate / native_rate)
        resampled_wav = resample(wav, num_samples)
        
        sd.play(resampled_wav, samplerate=target_rate)
        sd.wait()
    except Exception as e:
        print(f"⚠️ Audio playback engine error: {e}")

# 5. Camera Setup
print("Opening camera...")
camera = cv2.VideoCapture(0)
if camera.isOpened():
    print(f"✓ Camera ready\n")
else:
    print("⚠️  Warning: Could not open camera\n")

def should_use_camera(text: str) -> bool:
    """Check if user wants visual input."""
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in CAMERA_TRIGGERS)

def capture_frame():
    """Capture a frame from the webcam."""
    if not camera.isOpened():
        return None
    ret, frame = camera.read()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Main Loop
print("=" * 50)
print("  Voice + Vision Assistant Ready!")
print("  Speak to interact...")
print("  Say 'look' or 'see' to use the camera")
print("  Press Ctrl+C to exit")
print("=" * 50 + "\n")

try:
    while True:
        # 1. Listen for speech
        audio = record_when_speaking()
        if audio is None:
            continue
        
        # 2. Transcribe with native whisper.cpp
        print("📝 Transcribing...")
        user_text = transcribe_audio(audio)
        
        if not user_text:
            print("  (no speech detected or transcription failed)")
            continue
            
        print(f"👤 You: {user_text}")
        
        # 3. Check if we should capture an image
        image = None
        if should_use_camera(user_text):
            print("📸 Capturing image...")
            image = capture_frame()
            if image:
                print("  Image captured successfully")
            else:
                print("  ⚠️  Failed to capture image")
        
        # 4. Get response from MiniCPM-V
        print("🤔 Thinking...")
        t0 = time.time()
        try:
            response = minicpm_chat(user_text, image)
            elapsed = time.time() - t0
            print(f"🤖 Assistant ({elapsed:.1f}s): {response}")
            
            # 5. Speak the response (in thread for non-blocking)
            if response:
                threading.Thread(
                    target=speak,
                    args=(response,),
                    daemon=True
                ).start()
        except Exception as e:
            print(f"❌ Error generating response: {e}")
        
        print()
        
except KeyboardInterrupt:
    print("\n\nShutting down...")
    
finally:
    camera.release()
    sd.stop()
    print("Goodbye!")