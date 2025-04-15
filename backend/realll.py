import os
import time
import re
import torch
import speech_recognition as sr
from groq import Groq
from faster_whisper import WhisperModel
from TTS.api import TTS

# 🗣️ Voice Wake Word
wake_word = "hello"

# 🧠 AI Client (Groq)
groq_client = Groq(api_key="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")

# 🏗️ Load Whisper Model (Speech-to-Text)
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

# 🎤 Speech Recognition Setup
r = sr.Recognizer()
try:
    source = sr.Microphone()
    print("🎤 Microphone detected successfully!")
except OSError:
    print("❌ No microphone found. Check your input devices.")
    exit()

# 🔊 Text-to-Speech (TTS) Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC').to(device)

def generate_audio(text):
    """Convert AI-generated text to speech and save as audio file."""
    # Debugging: Print the length of the input text
    print(f"Input text length: {len(text)}")
    output_path = "outputs/output.wav"
    tts_model.tts_to_file(text=text, file_path=output_path)
    return output_path

def groq_prompt(prompt):
    """Get AI response from Groq API."""
    convo = [{'role': 'user', 'content': prompt}]
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def function_call(prompt):
    """Determine if a function should be executed based on user prompt."""
    sys_msg = (
        "You are an AI function-calling model. You will determine whether extracting the user's clipboard content, "
        "taking a screenshot, capturing the webcam, or calling no functions is best for a voice assistant to respond "
        "to the user's prompt. The webcam can be assumed to be a normal laptop webcam facing the user. "
        "You will respond with only one selection from this list: ['extract clipboard', 'take screenshot', 'capture webcam', 'None']. "
        "Do not respond with anything but the most logical selection from that list with no explanations. "
        "Format the function call name exactly as listed."
    )

    function_convo = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': prompt}
    ]

    chat_completion = groq_client.chat.completions.create(
        messages=function_convo,
        model='llama3-70b-8192'
    )
    response = chat_completion.choices[0].message
    return response.content

def wav_to_text(audio_path):
    """Convert WAV file to transcribed text using Whisper."""
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def extract_prompt(transcribed_text, wake_word):
    """Extract user prompt from detected speech using the wake word."""
    pattern = rf'\b{re.escape(wake_word)}[\s,?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def callback(recognizer, audio):
    """Process detected speech, generate AI response, and convert to audio output."""
    print("🔄 Detected sound... Processing")
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    print("📢 Converting speech to text...")
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        print(f'🗣️ USER: {clean_prompt}')
        call = function_call(clean_prompt)
        response = groq_prompt(clean_prompt)
        print(f'🤖 ASSISTANT: {response}')

        # Convert AI response to speech
        audio_output_path = generate_audio(response)
        print(f'🔊 Audio saved at: {audio_output_path}')
    else:
        print("❌ No valid command detected.")

def start_listening():
    """Continuously listen for voice commands."""
    with source as s:
        r.adjust_for_ambient_noise(s, duration=4)

    print('\nSay "', wake_word, '" followed by your prompt. Listening continuously...\n')

    listener = r.listen_in_background(source, callback)

    while True:
        time.sleep(0.5)

# Start the voice assistant
start_listening()
