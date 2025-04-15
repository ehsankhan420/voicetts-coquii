from groq import Groq
import os
from faster_whisper import WhisperModel
import speech_recognition as sr
import time
import re

wake_word = "hello"
groq_client = Groq(api_key="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")

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

r = sr.Recognizer()

# 🔍 Detect mic index if needed
try:
    source = sr.Microphone()
    print("🎤 Microphone detected successfully!")
except OSError:
    print("❌ No microphone found. Check your input devices.")
    exit()

def groq_prompt(prompt):
    convo = [{'role': 'user', 'content': prompt}]
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def function_call(prompt):
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
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def callback(recognizer, audio):
    print("🔄 Detected sound... Processing")
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    print("📢 Converting speech to text...")
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)
        response = groq_prompt(prompt=clean_prompt)
        print(f'ASSISTANT: {response}')
    else:
        print("❌ No valid command detected.")

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=4)

    print('\nSay "', wake_word, '" followed by your prompt. Listening continuously...\n')

    listener = r.listen_in_background(source, callback)

    while True:
        time.sleep(0.5)

start_listening()
