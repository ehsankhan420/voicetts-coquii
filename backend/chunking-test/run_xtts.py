from TTS.api import TTS
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
import os
import nltk
nltk.download("punkt_tab")


# Make sure NLTK is ready
import nltk
nltk.download('punkt')

# Load XTTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",gpu=True, progress_bar=True)
tts.to("cuda")

# Input text (long enough to test naturalness)
input_text = """
Medicare is a federal health insurance program in the United States, primarily designed for people aged 65 and older, although it also serves younger individuals with certain disabilities or medical conditions. Medicare is divided into several parts, each serving a unique purpose. Part A covers hospital insurance. That means if you’re admitted to a hospital, a skilled nursing facility, or even need hospice care, Part A helps with those costs. Then there’s Part B, which is more about everyday health needs like doctor visits, outpatient care, preventive services, and even some medical equipment.
Together, Parts A and B make up what we call Original Medicare. Many people opt for Medicare Part C, also known as Medicare Advantage. These are plans offered by private insurers that combine Parts A and B, and sometimes Part D, into a single plan, often with added benefits like vision, dental, or wellness programs. Speaking of Part D, that’s the prescription drug coverage portion. It helps beneficiaries pay for the medications they need on a daily basis.
Without it, out-of-pocket costs for prescriptions can be significantly higher. While Medicare offers substantial coverage, it doesn’t pay for everything. For instance, long-term care, dental procedures, hearing aids, and eyeglasses often fall outside its scope.
"""


sentences = sent_tokenize(input_text)
os.makedirs("chunks", exist_ok=True)
all_chunks = []


for i, sentence in enumerate(sentences):
    out_path = f"chunks/chunk_{i}.wav"
    tts.tts_to_file(
        text=sentence,
        file_path=out_path,
        speaker_wav="samples/sales_voice.wav",  
        language="en"
    )
    all_chunks.append(out_path)


final_audio = AudioSegment.empty()
for path in all_chunks:
    final_audio += AudioSegment.from_wav(path)

final_audio.export("final_output.wav", format="wav")
print("✅ Synthesis complete! Check: final_output.wav")
