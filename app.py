import io
import os
import time
import base64
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample
import scipy.io.wavfile as wavfile
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

# Configuration
WHISPER_CPP_DIR = Path("/home/cheng/Downloads/AI/whisper.cpp")
WHISPER_BINARY = WHISPER_CPP_DIR / "build/bin/whisper-cli"
WHISPER_MODEL = WHISPER_CPP_DIR / "models/ggml-base.en.bin"
TARGET_SAMPLE_RATE = 16000

# Create directory for saving temp files
TEMP_DIR = Path("/home/cheng/Downloads/AI/data/messages")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
print(f"Temp files will be saved to: {TEMP_DIR}")

# GPU Optimization
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

app = FastAPI(title="Panamera AI Assistant")

# 1. MiniCPM-V Setup
print("Loading MiniCPM-V (4-bit)...")
model_id = "openbmb/MiniCPM-V-4.6"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Try to load processor, but don't fail if not available
try:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("✓ Processor loaded")
except Exception as e:
    print(f"⚠️ Could not load AutoProcessor: {e}")
    processor = None

print("✓ MiniCPM-V loaded\n")

# 2. VoxCPM Setup
print("Loading VoxCPM...")
try:
    from voxcpm import VoxCPM
    tts_model_id = "openbmb/VoxCPM-0.5B"
    tts_model = VoxCPM.from_pretrained(tts_model_id)
    print("✓ VoxCPM loaded\n")
except Exception as e:
    print(f"⚠️ Could not load VoxCPM: {e}")
    tts_model = None

# Setup static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def read_root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Panamera Web UI</h1><p>Static directory is empty. Please add index.html</p>")

@app.get("/api/status")
def get_status():
    return {
        "status": "ready",
        "vision_model": "loaded" if model is not None else "loading",
        "tts_model": "loaded" if tts_model is not None else "loading"
    }

def save_audio_debug_copy(audio_data: np.ndarray, sample_rate: int, prefix: str) -> str:
    """Save a debug copy of audio data for analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = TEMP_DIR / f"{prefix}_{timestamp}.wav"
    
    # Convert to int16 for saving
    if audio_data.dtype != np.int16:
        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        audio_int16 = audio_data
    
    wavfile.write(filename, sample_rate, audio_int16)
    print(f"📁 Saved debug audio: {filename}")
    return str(filename)

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe uploaded audio bytes using whisper.cpp with temp file."""
    print(f"Received audio upload: {len(audio_bytes)} bytes")
    
    temp_wav_file = None
    debug_original_file = None
    debug_resampled_file = None
    
    try:
        # Read WAV from bytes
        sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
        print(f"WAV Info - Sample Rate: {sample_rate}, Shape: {audio_data.shape}, Dtype: {audio_data.dtype}")

        # Save original audio for debugging
        debug_original_file = save_audio_debug_copy(audio_data, sample_rate, "original")

        # Convert to float32 mono
        if audio_data.dtype != np.float32:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data
            
        if len(audio_float.shape) > 1:
            audio_float = audio_float[:, 0]  # Take first channel
            
        # Resample to 16000Hz if necessary
        if sample_rate != TARGET_SAMPLE_RATE:
            print(f"Resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz...")
            num_samples = int(len(audio_float) * TARGET_SAMPLE_RATE / sample_rate)
            audio_16k = resample(audio_float, num_samples)
        else:
            audio_16k = audio_float

        # Convert to int16 for WAV file
        audio_int16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)
        
        # Create temporary WAV file in the specified directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        temp_wav_file = TEMP_DIR / f"whisper_input_{timestamp}.wav"
        
        # Write WAV file at 16kHz
        wavfile.write(temp_wav_file, TARGET_SAMPLE_RATE, audio_int16)
        print(f"Created temp WAV file: {temp_wav_file}")
        
        # Save resampled audio for debugging
        debug_resampled_file = save_audio_debug_copy(audio_int16, TARGET_SAMPLE_RATE, "resampled")
        
        # Call whisper.cpp with the WAV file
        cmd = [
            str(WHISPER_BINARY),
            "-m", str(WHISPER_MODEL),
            "-f", str(temp_wav_file),
            "-t", "4",
            "-l", "en",
            "--no-timestamps",
            "--print-progress", "false"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            check=True
        )
        
        text = result.stdout.decode('utf-8').strip()
        print(f"Whisper output: '{text}'")
        
        if result.stderr:
            stderr_text = result.stderr.decode('utf-8').strip()
            if stderr_text:
                print(f"Whisper stderr: {stderr_text}")
        
        # Clean up the text
        text = text.replace('[BLANK_AUDIO]', '').strip()
        
        # Save transcription result
        transcription_file = TEMP_DIR / f"transcription_{timestamp}.txt"
        with open(transcription_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"📝 Saved transcription: {transcription_file}")
        
        return text
        
    except subprocess.CalledProcessError as e:
        print(f"Whisper process error: {e}")
        print(f"Stderr: {e.stderr.decode('utf-8') if e.stderr else 'None'}")
        return ""
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Transcription error: {e}")
        return ""
    finally:
        # Note: We're NOT deleting the temp file anymore
        # They will be saved permanently in TEMP_DIR
        if temp_wav_file and temp_wav_file.exists():
            print(f"📁 Temp file preserved at: {temp_wav_file}")
        if debug_original_file:
            print(f"📁 Original audio preserved at: {debug_original_file}")
        if debug_resampled_file:
            print(f"📁 Resampled audio preserved at: {debug_resampled_file}")

def generate_tts_audio_base64(text: str) -> str:
    """Generate audio using VoxCPM."""
    if not tts_model:
        return ""
    try:
        wav = tts_model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
        audio_int16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        
        # Save TTS output for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        tts_file = TEMP_DIR / f"tts_output_{timestamp}.wav"
        wavfile.write(tts_file, TARGET_SAMPLE_RATE, audio_int16)
        print(f"🎵 Saved TTS audio: {tts_file}")
        
        # Also return as base64 for immediate playback
        wav_io = io.BytesIO()
        wavfile.write(wav_io, TARGET_SAMPLE_RATE, audio_int16)
        return base64.b64encode(wav_io.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ TTS error: {e}")
        return ""

def generate_response(user_text: str, pil_image=None) -> str:
    """
    Generate response using MiniCPM-V's chat method.
    """
    try:
        # Try different approaches based on what's available
        
        # Approach 1: Use model.chat() if available (older versions)
        if hasattr(model, 'chat'):
            messages = [{"role": "user", "content": user_text}]
            response, _ = model.chat(
                image=pil_image,
                msgs=messages,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7
            )
            
            # Save response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            response_file = TEMP_DIR / f"response_{timestamp}.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"User: {user_text}\n\nAssistant: {response}")
            print(f"💾 Saved response: {response_file}")
            
            return response
        
        # Approach 2: Use model.generate() with prompt formatting
        elif processor is not None:
            # Build conversation
            if pil_image:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": user_text}
                        ]
                    }
                ]
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_file = TEMP_DIR / f"image_{timestamp}.jpg"
                pil_image.save(image_file)
                print(f"🖼️ Saved image: {image_file}")
            else:
                messages = [{"role": "user", "content": user_text}]
            
            # Apply chat template
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # Tokenize
            inputs = processor(
                text=prompt,
                images=pil_image if pil_image else None,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = processor.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Save response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            response_file = TEMP_DIR / f"response_{timestamp}.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"User: {user_text}\n\nAssistant: {response}")
            print(f"💾 Saved response: {response_file}")
            
            return response
        
        # Approach 3: Simple text-only generation fallback
        else:
            inputs = tokenizer(user_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove input text from response
            if response.startswith(user_text):
                response = response[len(user_text):].strip()
            
            # Save response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            response_file = TEMP_DIR / f"response_{timestamp}.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"User: {user_text}\n\nAssistant: {response}")
            print(f"💾 Saved response: {response_file}")
            
            return response
            
    except Exception as e:
        print(f"MiniCPM generation error: {e}")
        import traceback
        traceback.print_exc()
        return f"I encountered an error: {str(e)}"

@app.post("/api/interact")
async def interact(
    text: str = Form(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(None)
):
    user_text = text or ""
    
    # Handle audio transcription
    if audio:
        audio_bytes = await audio.read()
        transcribed_text = transcribe_audio_bytes(audio_bytes)
        if transcribed_text:
            user_text = transcribed_text
            
    if not user_text:
        raise HTTPException(status_code=400, detail="No input text or audio provided.")

    print(f"👤 User: {user_text}")
    
    # Process image
    pil_image = None
    if image:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("📸 Image attached")

    # Generate response
    print("🤔 Thinking...")
    t0 = time.time()
    response = generate_response(user_text, pil_image)
    elapsed = time.time() - t0
    print(f"🤖 Assistant ({elapsed:.1f}s): {response}")

    # Generate TTS
    audio_base64 = generate_tts_audio_base64(response) if response else ""

    return {
        "user_text": user_text,
        "assistant_response": response,
        "audio": audio_base64,
        "elapsed_seconds": round(elapsed, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)