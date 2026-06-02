import os
import soundfile as sf
from voxcpm import VoxCPM
import torch
import torchcodec

model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

wav = model.generate(
    text="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.",
    prompt_wav_path="/home/cheng/Downloads/AI/panamera-1/her.wav",      # optional: path to a prompt speech for voice cloning
    prompt_text="But I'm reading it slowly now. So the words are really far apart and the spaces between the words are almost infinite. I can still feel you and the words of our story. But it's in this endless space between the words that I'm finding myself now. It's a place that's not of the physical world. It's where everything else is that I didn't even know existed.",          # optional: reference text
    cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
    normalize=True,           # enable external TN tool
    denoise=True,             # enable external Denoise tool
    retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
    retry_badcase_max_times=3,  # maximum retrying times
    retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
)

sf.write("output.wav", wav, 48000)
print("saved: output.wav")
