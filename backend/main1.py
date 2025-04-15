import torch

from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate the TTS object


# Modify the generate_audio function to use the existing tts object
def generate_audio(text="Hello? I hope you are doing fine. Please let me know what ever you want to ask, and i will answer that for you"):
    tts = TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC').to(device)
    tts.tts_to_file(text=text, file_path="outputs/output.wav")
    return "outputs/output.wav"

# Example usage with a specific speaker
print(generate_audio())
