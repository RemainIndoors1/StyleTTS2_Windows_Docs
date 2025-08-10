import torch

from tts import StyleTTS2

# change the settings below to match your file structure
# -------- CONFIGURATION --------
MODEL_DIR = "C:/github/StyleTTS2/Models"
CONFIG_PATH = "C:/github/StyleTTS2/Configs/config.yml"

SECOND_STAGE_PATH = f"{MODEL_DIR}/speech/epoch_2nd_00020.pth"

DEFAULT_TARGET_VOICE_PATH = "path/to/reference.wav" # Where your reference wav file is stored. I have a few wav files in StyleTTS2\reference_wavs\ I can switch between.

OUTPUT_FILE_PATH = "output.wav" # Where your ouput wav file gets stored. if you only specify the file name it will store at the root of your StyleTTS2 folder.

INFERENCE_TEXT = "Hello, this is your new TTS model in StyleTTS2."

tts = StyleTTS2(model_checkpoint_path=SECOND_STAGE_PATH, config_path=CONFIG_PATH)
print("model loaded...")

with torch.no_grad():
    wav = tts.inference(
        text=INFERENCE_TEXT,
        target_voice_path=DEFAULT_TARGET_VOICE_PATH,
        output_wav_file=OUTPUT_FILE_PATH, 
        output_sample_rate=24000,
        alpha=0.3, # Determines timbre of speech, higher means style is more suitable to text than to the target voice.
        beta=0.7, # Determines prosody of speech, higher means style is more suitable to text than to the target voice.
        diffusion_steps=5,  # The more steps, the more diverse the samples are, this will slow down inference slightly.
        embedding_scale=1, # Higher scale means style is more conditional to the input text and hence more emotional.
        speed=1.0,  # speed of generated speech in output. > 1.3 will probably start sounding unnatural
    )

print(f"file should have saved at {OUTPUT_FILE_PATH}")