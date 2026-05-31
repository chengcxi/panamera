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