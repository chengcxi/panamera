import sounddevice as sd
import torch
from scipy.signal import resample
from voxcpm import VoxCPM

torch.set_float32_matmul_precision('high')

# Load the model
vox_model_id = "openbmb/VoxCPM-0.5B"
model = VoxCPM.from_pretrained(vox_model_id)

text_to_speak = "Playing audio directly through your speakers using sounddevice."

# Generate the audio array
print("Generating audio...")
wav = model.generate(
    text=text_to_speak,
    cfg_value=2.0,
    inference_timesteps=10
)

# Native VoxCPM output properties
native_rate = 16000
target_rate = 48000  # A globally supported hardware standard sample rate

print(f"Resampling audio from {native_rate}Hz to {target_rate}Hz...")
# Calculate the target number of samples needed
num_samples = int(len(wav) * target_rate / native_rate)
resampled_wav = resample(wav, num_samples)

print("Playing...")
# Play back using the safe, resampled rate
sd.play(resampled_wav.astype('float32'), 48000, device=13)
sd.wait()
print("Playback finished!")