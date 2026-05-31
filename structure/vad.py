import numpy as np
import sounddevice as sd
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

vad_model = load_silero_vad()
SAMPLE_RATE = 16000          # VAD and Whisper both like 16k

def record_when_speaking(
    silence_threshold_sec=0.8,
    max_duration_sec=15,
) -> np.ndarray | None:
    """
    Listens to the microphone. As soon as speech is detected, records until
    there is at least `silence_threshold_sec` of silence.
    Returns the whole utterance as a float32 numpy array, or None if no speech.
    """
    print("Listening...")
    audio_chunks = []
    is_speaking = False
    silence_frames = 0
    silence_limit = int(silence_threshold_sec * SAMPLE_RATE / 512)  # 512 frames per block

    def callback(indata, frames, time, status):
        nonlocal is_speaking, silence_frames, audio_chunks
        if status:
            print(status)

        mono = indata[:, 0] if indata.ndim > 1 else indata
        audio_chunks.append(mono.copy())

        # Check VAD on the latest ~1 second of audio
        recent = np.concatenate(audio_chunks[-int(SAMPLE_RATE/512):])  # 1 sec context
        speech_ts = get_speech_timestamps(recent, vad_model, sampling_rate=SAMPLE_RATE)

        if len(speech_ts) > 0:
            is_speaking = True
            silence_frames = 0
        else:
            if is_speaking:
                silence_frames += 1

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=callback,
        blocksize=512,
    ):
        while True:
            sd.sleep(100)
            # Start recording only after we actually have some audio
            if is_speaking and silence_frames >= silence_limit:
                break
            # Safety timeout
            if len(audio_chunks) > (max_duration_sec * SAMPLE_RATE // 512):
                break

    if not is_speaking:
        print("No speech detected.")
        return None

    full_audio = np.concatenate(audio_chunks)
    return full_audio