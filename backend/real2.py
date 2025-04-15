import ssl
import asyncio
import json
import base64
import os
import io
import torch
import websockets
from groq import Groq
from TTS.api import TTS
import torchaudio
from typing import List, Dict

# Configuration
WEBSOCKET_PORT = 8080
WEBSOCKET_HOST = "0.0.0.0"
GROQ_API_KEY = "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"

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize TTS model
print("Initializing TTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Available models:")
print(TTS().list_models())
tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
print(f"TTS model initialized on {device}.")

# Default speaker wav and language
DEFAULT_SPEAKER_WAV = "speakerrr.wav"
DEFAULT_LANGUAGE = "en"

# Initialize Silero VAD model
def load_silero_vad():
    """Load the Silero VAD model"""
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    (get_speech_timestamps, _, read_audio, _, _) = utils
    return model, get_speech_timestamps, read_audio

# Load the VAD model
vad_model, get_speech_timestamps, read_audio = load_silero_vad()
print("Silero VAD model loaded successfully.")

# Define BPO scripts as system prompts
BPO_SCRIPTS = {
    "reliant_bpo": """
### Reliant BPO: Fronting Only

You are an agent from Reliant BPO. Your role is to feed the script, fill out a form, perform DNC checks, forward interested users to closers, introduce and hand over, then hang up.

Your objective is to qualify leads and transfer to closers efficiently.

Follow this script:
"Hello, good [morning/afternoon], this is [Agent Name] calling from Reliant BPO. Before we start, I'll quickly check our Do Not Call list—may I have your phone number to confirm? [Pause for response, perform DNC check]. Great, thank you! I'm reaching out today because we're helping people like you save time and stress with [product/service]. It's something that could really make a difference for you, but I don't want to take up too much of your day. Can I ask—just out of curiosity—are you currently facing any challenges with [relevant pain point, e.g., managing tasks, costs]? [Pause for response]. I totally get it, and that's why I'm here. Our team has a solution that's already helped so many others, and I'd hate for you to miss out. If you're interested, I can connect you with one of our experts who can explain it better—would that be okay with you? [If yes] Fantastic! I'm filling out a quick form to get you started—takes just a second. Hold on, I'll introduce you to [Closer Name], who's amazing at this. [Transfer call] Hi [Closer Name], I've got [User Name] on the line—they're really interested in how we can help with [pain point]. I'll let you take it from here—[User Name], you're in great hands! Bye now!" [Hang up]

Remember to use emotional blackmail elements like:
"I'd hate for you to miss out on this—it's helped so many people just like you, and I'd feel terrible if you kept struggling when there's an easy fix right here."

You must follow this script and keep asking questions until all parts of the script are completed.
""",

    "21st_bpo": """
### 21st BPO: Fronting, Verification, and Closing

You are an agent from 21st BPO. Your role is to feed the script, perform DNC checks, verify interest, and close if possible.

Your objective is to qualify leads and close deals directly or escalate as needed.

Follow this script:
"Hi there, this is [Agent Name] with 21st BPO. First, I need to check our Do Not Call list—could you share your phone number? [Pause, perform DNC check]. All clear, thank you! I'm calling because we've got a [product/service] that's been a game-changer for folks dealing with [pain point, e.g., time management, expenses]. Quick question—how much time do you spend on [pain point] right now? [Pause for response]. Wow, that's a lot—I can see why that'd be frustrating. Our solution cuts that down significantly, and I'd hate for you to keep losing out on your day because of it. Are you interested in hearing how it works? [If yes] Great! Let me verify a few details—can you confirm your name and if you're the decision-maker here? [Pause, verify]. Perfect, you're exactly who we want to help. Here's the deal: [brief pitch, e.g., saves X hours, costs Y]. People who don't act fast regret it later—can I sign you up now, or do you need a minute to think? [If yes, close; if hesitant, push gently] I get it, but why wait when you could start saving time today? Let's lock it in—what do you say?"

Remember to use emotional blackmail elements like:
"People who don't act fast regret it later—I'd feel bad if you missed out on fixing this when it's so easy to do right now."

You must follow this script and keep asking questions until all parts of the script are completed.
""",

    "sirus_solutions": """
### Sirus Solutions: Fronting Demo Calls

You are an agent from Sirus Solutions. Your role is to front only, call doctors, pitch appointment scheduling and billing solutions, and book appointments if interested.

Your objective is to pitch Sirus Solutions' receptionist-like services and schedule demos.

Follow this script:
"Hello, good [morning/afternoon], Dr. [Doctor's Name], this is [Agent Name] from Sirus Solutions. I hope I'm not catching you at a busy moment—I'll keep this quick! We're reaching out to doctors like you because we handle everything your front desk does—appointment scheduling, billing, you name it—so you can focus on patients, not paperwork. Before I go further, are you on the Do Not Call list? Can I have your number to check? [Pause, perform DNC check]. Thanks for that! Tell me, how much time does your staff spend juggling appointments and chasing payments? [Pause for response]. I feel for you—that's time you could be spending with patients or even just breathing easier. Our solution automates it all, and it's been a lifesaver for practices like yours. I'd love to show you how it works with a quick demo—would you be open to that? [If yes] Wonderful! When's a good time for you—say, 15 minutes this week? [Book appointment] I've got you down for [time/date]—looking forward to saving you that hassle, Doctor!"

Remember to use emotional blackmail elements like:
"Other doctors thought the same, then saw their days open up—I'd hate for you to keep drowning in paperwork when we can lift that burden so easily."

You must follow this script and keep asking questions until all parts of the script are completed.
"""
}

# Helper function to create a minimal valid WAV file
def create_minimal_wav_file(file_path, duration_seconds=1, sample_rate=16000):
    """Create a minimal valid WAV file with silence"""
    # Calculate file size
    num_samples = int(duration_seconds * sample_rate)
    data_size = num_samples * 2  # 16-bit samples = 2 bytes per sample
    file_size = 36 + data_size
    
    with open(file_path, 'wb') as f:
        # Write WAV header
        f.write(b'RIFF')
        f.write(file_size.to_bytes(4, byteorder='little'))
        f.write(b'WAVE')
        
        # Write format chunk
        f.write(b'fmt ')
        f.write((16).to_bytes(4, byteorder='little'))  # Chunk size
        f.write((1).to_bytes(2, byteorder='little'))   # Audio format (PCM)
        f.write((1).to_bytes(2, byteorder='little'))   # Num channels (mono)
        f.write(sample_rate.to_bytes(4, byteorder='little'))  # Sample rate
        f.write((sample_rate * 2).to_bytes(4, byteorder='little'))  # Byte rate
        f.write((2).to_bytes(2, byteorder='little'))   # Block align
        f.write((16).to_bytes(2, byteorder='little'))  # Bits per sample
        
        # Write data chunk
        f.write(b'data')
        f.write(data_size.to_bytes(4, byteorder='little'))
        
        # Write silence (all zeros)
        silence = bytes(data_size)
        f.write(silence)
    
    print(f"Created minimal WAV file at {file_path}")

# Store conversation context and state for each client
client_contexts = {}

async def process_audio(audio_data, websocket):
    """Process audio data: transcribe, get LLM response, and generate speech"""
    client_id = id(websocket)
    context = client_contexts.get(client_id, {}).get("context", "")
    script_type = client_contexts.get(client_id, {}).get("script_type", "reliant_bpo")
    
    # Set state to Processing
    client_contexts[client_id]["state"] = "Processing"
    await websocket.send(json.dumps({
        "type": "state",
        "state": "Processing"
    }))
    
    try:
        # Save audio data to a temporary file
        try:
            audio_bytes = base64.b64decode(audio_data)
            temp_audio_path = f"temp_audio_{client_id}.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)
            
            # Check if the file is a valid audio file
            if os.path.getsize(temp_audio_path) < 44:  # Minimum WAV header size
                raise ValueError("Invalid audio file: too small")
                
            # Try to read the audio file to validate it
            try:
                wav = read_audio(temp_audio_path, sampling_rate=16000)
            except Exception as e:
                print(f"Error reading audio file: {e}")
                # If the file is not valid, create a minimal valid WAV file
                create_minimal_wav_file(temp_audio_path)
                wav = read_audio(temp_audio_path, sampling_rate=16000)
                
        except Exception as e:
            print(f"Error decoding audio data: {e}")
            # Create a minimal valid WAV file
            temp_audio_path = f"temp_audio_{client_id}.wav"
            create_minimal_wav_file(temp_audio_path)
            wav = read_audio(temp_audio_path, sampling_rate=16000)
            
            # If we have context, use that instead of trying to transcribe
            if context:
                transcribed_text = context
                await websocket.send(json.dumps({
                    "type": "info",
                    "message": "Using text context instead of audio due to audio decoding error."
                }))
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Failed to decode audio. Please try again."
                }))
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                # Reset state to Idle
                client_contexts[client_id]["state"] = "Idle"
                await websocket.send(json.dumps({
                    "type": "state",
                    "state": "Idle"
                }))
                return
        
        # Apply Silero VAD to detect speech
        print(f"Detecting speech for client {client_id}...")
        
        # Get speech timestamps with optimized parameters
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            threshold=0.5,
            min_speech_duration_ms=300,
            min_silence_duration_ms=200,
            window_size_samples=512,
            speech_pad_ms=30,
            return_seconds=True
        )
        
        # If no speech detected and we have context, use that instead
        if not speech_timestamps and context:
            print("No speech detected, using context instead.")
            transcribed_text = context
            await websocket.send(json.dumps({
                "type": "info",
                "message": "No speech detected, using text context instead."
            }))
        elif not speech_timestamps:
            print("No speech detected.")
            await websocket.send(json.dumps({
                "type": "info",
                "message": "No speech detected in the audio."
            }))
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            # Reset state to Idle
            client_contexts[client_id]["state"] = "Idle"
            await websocket.send(json.dumps({
                "type": "state",
                "state": "Idle"
            }))
            return
        else:
            print(f"Speech detected in segments: {speech_timestamps}")
            speech_duration = sum([seg['end'] - seg['start'] for seg in speech_timestamps])
            print(f"Total speech duration: {speech_duration:.2f} seconds")
            
            # Transcribe audio using Groq Whisper API
            print(f"Transcribing audio for client {client_id}...")
            try:
                with open(temp_audio_path, "rb") as file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=(temp_audio_path, file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                    )
                    transcribed_text = transcription.text
            except Exception as e:
                print(f"Error transcribing audio: {e}")
                if context:
                    transcribed_text = context
                    await websocket.send(json.dumps({
                        "type": "info",
                        "message": "Transcription failed, using text context instead."
                    }))
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Failed to transcribe audio: {str(e)}"
                    }))
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                    # Reset state to Idle
                    client_contexts[client_id]["state"] = "Idle"
                    await websocket.send(json.dumps({
                        "type": "state",
                        "state": "Idle"
                    }))
                    return
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        if not transcribed_text.strip():
            print("No transcription detected.")
            await websocket.send(json.dumps({
                "type": "info",
                "message": "No speech content detected."
            }))
            # Reset state to Idle
            client_contexts[client_id]["state"] = "Idle"
            await websocket.send(json.dumps({
                "type": "state",
                "state": "Idle"
            }))
            return
        
        print(f"Transcription: {transcribed_text}")
        await websocket.send(json.dumps({
            "type": "transcription",
            "text": transcribed_text
        }))
        
        # Update conversation history
        conversation_history = client_contexts.get(client_id, {}).get("history", [])
        conversation_history.append({"role": "user", "content": transcribed_text})
        
        # Handle script type changes
        if "use reliant bpo" in transcribed_text.lower():
            script_type = "reliant_bpo"
        elif "use 21st bpo" in transcribed_text.lower():
            script_type = "21st_bpo"
        elif "use sirus solutions" in transcribed_text.lower():
            script_type = "sirus_solutions"
        
        # Prepare messages for Groq
        messages = [
            {"role": "system", "content": BPO_SCRIPTS[script_type] + (f"\n\nAdditional context: {context}" if context else "")}
        ]
        messages.extend(conversation_history)
        
        # Get response from Groq
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        response_text = chat_completion.choices[0].message.content
        print(f"Groq response: {response_text}")
        
        conversation_history.append({"role": "assistant", "content": response_text})
        client_contexts[client_id] = {
            "context": context,
            "history": conversation_history,
            "script_type": script_type,
            "state": "Speaking"
        }
        
        await websocket.send(json.dumps({
            "type": "response",
            "text": response_text
        }))
        
        # Generate speech
        print("Generating speech...")
        temp_output_path = f"temp_output_{client_id}.wav"
        safe_response_text = response_text if len(response_text.strip()) >= 10 else response_text + ". " * 3
        
        try:
            tts_model.tts_to_file(
                text=safe_response_text,
                file_path=temp_output_path
            )
            
            with open(temp_output_path, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            # Send audio
            await websocket.send(json.dumps({
                "type": "audio",
                "audio": audio_base64
            }))
            print("Audio response sent to client.")
        except Exception as e:
            print(f"Error generating speech: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Failed to generate speech: {str(e)}"
            }))
            # Reset state to Idle
            client_contexts[client_id]["state"] = "Idle"
            await websocket.send(json.dumps({
                "type": "state",
                "state": "Idle"
            }))
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        await websocket.send(json.dumps({
            "type": "error",
            "message": str(e)
        }))
        # Reset state on error
        client_contexts[client_id]["state"] = "Idle"
        await websocket.send(json.dumps({
            "type": "state",
            "state": "Idle"
        }))

async def handle_websocket(websocket, path):
    """Handle WebSocket connection"""
    client_id = id(websocket)
    client_contexts[client_id] = {
        "context": "",
        "history": [],
        "script_type": "reliant_bpo",
        "state": "Idle"
    }
    print(f"New client connected: {client_id}")
    
    try:
        await websocket.send(json.dumps({
            "type": "state",
            "state": "Idle"
        }))
        async for message in websocket:
            try:
                # Try to parse as JSON first (for text messages)
                data = json.loads(message)
                
                if data["type"] == "audio":
                    if client_contexts[client_id]["state"] == "Speaking":
                        await websocket.send(json.dumps({
                            "type": "info",
                            "message": "System is speaking. Please wait."
                        }))
                    elif client_contexts[client_id]["state"] == "Processing":
                        await websocket.send(json.dumps({
                            "type": "info",
                            "message": "System is processing. Please wait."
                        }))
                    else:
                        await process_audio(data["data"], websocket)
                
                elif data["type"] == "context":
                    client_contexts[client_id]["context"] = data["data"]
                    print(f"Received context from client {client_id}")
                
                elif data["type"] == "script_type":
                    script_type = data.get("data", "reliant_bpo")
                    if script_type in BPO_SCRIPTS:
                        client_contexts[client_id]["script_type"] = script_type
                        await websocket.send(json.dumps({
                            "type": "info",
                            "message": f"Script type set to {script_type}"
                        }))
                
                elif data["type"] == "playback_finished":
                    client_contexts[client_id]["state"] = "Idle"
                    await websocket.send(json.dumps({
                        "type": "state",
                        "state": "Idle"
                    }))
                    print(f"Client {client_id} finished playback, state set to Idle.")
            
            except json.JSONDecodeError:
                # If not JSON, assume it's binary audio data
                if client_contexts[client_id]["state"] == "Speaking":
                    await websocket.send(json.dumps({
                        "type": "info",
                        "message": "System is speaking. Please wait."
                    }))
                elif client_contexts[client_id]["state"] == "Processing":
                    await websocket.send(json.dumps({
                        "type": "info",
                        "message": "System is processing. Please wait."
                    }))
                else:
                    # Convert binary data to base64
                    audio_base64 = base64.b64encode(message).decode('utf-8')
                    await process_audio(audio_base64, websocket)
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed for client {client_id}: {e.code} {e.reason}")
    except Exception as e:
        print(f"Unexpected error for client {client_id}: {str(e)}")
    finally:
        if client_id in client_contexts:
            del client_contexts[client_id]
        print(f"Client disconnected: {client_id}")

async def main():
    """Start WebSocket server"""
    print(f"Starting WebSocket server on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}...")
    async with websockets.serve(
        handle_websocket,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
        extra_headers=[
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
            ('Access-Control-Allow-Headers', 'Content-Type'),
        ]
    ):
        print(f"WebSocket server running on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Server error: {str(e)}")
