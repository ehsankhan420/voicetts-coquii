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
from pydub import AudioSegment  # For audio format conversion

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

# Initialize XTTS model
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"TTS model initialized on {device}.")

# Default speaker wav and language
DEFAULT_SPEAKER_WAV = "voice.wav"
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
    "reliant_bpo": """...""",  # (Your existing script content)
    "21st_bpo": """...""",     # (Your existing script content)
    "sirus_solutions": """..."""  # (Your existing script content)
}

# Store conversation context for each client
client_contexts = {}

def ensure_wav_format(audio_path: str) -> str:
    """Ensure the audio file is in proper WAV format"""
    try:
        # Try to load with torchaudio first
        waveform, sample_rate = torchaudio.load(audio_path)
        return audio_path
    except Exception as e:
        print(f"Standard WAV load failed, converting audio: {e}")
        try:
            # Try to convert using pydub
            audio = AudioSegment.from_file(audio_path)
            converted_path = audio_path + ".converted.wav"
            audio.export(converted_path, format="wav")
            return converted_path
        except Exception as conv_e:
            print(f"Audio conversion failed: {conv_e}")
            raise ValueError("Could not convert audio to proper WAV format")

async def process_audio(audio_data, websocket):
    """Process audio data: transcribe, get LLM response, and generate speech"""
    client_id = id(websocket)
    context = client_contexts.get(client_id, {}).get("context", "")
    script_type = client_contexts.get(client_id, {}).get("script_type", "reliant_bpo")
    
    try:
        # Save audio data to a temporary file
        audio_bytes = base64.b64decode(audio_data)
        temp_audio_path = f"temp_audio_{client_id}.wav"
        
        # Write the audio data to file
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        
        # Verify the file exists and has content
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            raise ValueError("Failed to create valid audio file")
        
        # Ensure proper WAV format
        try:
            converted_audio_path = ensure_wav_format(temp_audio_path)
            if converted_audio_path != temp_audio_path:
                # Replace original with converted file
                os.remove(temp_audio_path)
                temp_audio_path = converted_audio_path
        except Exception as conv_e:
            print(f"Audio conversion error: {conv_e}")
            raise ValueError("Could not process audio file format")
        
        # Apply Silero VAD to detect speech segments
        print(f"Detecting speech segments using Silero VAD for client {client_id}...")
        try:
            wav = read_audio(temp_audio_path)
        except Exception as e:
            print(f"Error reading audio file with Silero: {e}")
            # Fallback to torchaudio
            try:
                waveform, sample_rate = torchaudio.load(temp_audio_path)
                wav = waveform.numpy().squeeze()
                print("Successfully loaded audio with torchaudio fallback")
            except Exception as fallback_e:
                print(f"Fallback audio loading failed: {fallback_e}")
                raise
        
        # Get speech timestamps in seconds
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            return_seconds=True
        )
        
        # Check if speech was detected
        if not speech_timestamps:
            print("No speech detected in the audio.")
            await websocket.send(json.dumps({
                "type": "info",
                "message": "No speech detected in the audio."
            }))
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return
        
        print(f"Speech detected: {speech_timestamps}")
        
        # Continue with transcription since speech was detected
        print(f"Transcribing audio for client {client_id} using Groq Whisper API...")
        
        # Transcribe audio using Groq Whisper API
        with open(temp_audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
            transcribed_text = transcription.text
        
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        if not transcribed_text.strip():
            print("No transcription detected.")
            return
        
        print(f"Transcription: {transcribed_text}")
        
        # Send transcription back to client
        await websocket.send(json.dumps({
            "type": "transcription",
            "text": transcribed_text
        }))
        
        # Prepare conversation history
        conversation_history = client_contexts.get(client_id, {}).get("history", [])
        conversation_history.append({"role": "user", "content": transcribed_text})
        
        # Check if the user is trying to change the script type
        if "use reliant bpo" in transcribed_text.lower():
            script_type = "reliant_bpo"
            print(f"Switching to Reliant BPO script for client {client_id}")
        elif "use 21st bpo" in transcribed_text.lower():
            script_type = "21st_bpo"
            print(f"Switching to 21st BPO script for client {client_id}")
        elif "use sirus solutions" in transcribed_text.lower():
            script_type = "sirus_solutions"
            print(f"Switching to Sirus Solutions script for client {client_id}")
        
        # Prepare system message with context and selected BPO script
        messages = []
        system_message = BPO_SCRIPTS.get(script_type, BPO_SCRIPTS["reliant_bpo"])
        
        if context:
            system_message += f"\n\nAdditional context for this conversation: {context}"
        
        system_message += "\n\nWhile following the script, you should also be able to answer general questions, but always return to the script flow afterward. Keep asking questions from the script until all parts are completed."
        
        messages.append({"role": "system", "content": system_message})
        messages.extend(conversation_history)
        
        print(f"Getting response from Groq using {script_type} script...")
        
        # Get response from Groq using Llama model
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        
        response_text = chat_completion.choices[0].message.content
        print(f"Groq response: {response_text}")
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        # Update client context
        client_contexts[client_id] = {
            "context": context,
            "history": conversation_history,
            "script_type": script_type
        }
        
        # Send text response to client
        await websocket.send(json.dumps({
            "type": "response",
            "text": response_text
        }))
        
        print("Generating speech from response...")
        temp_output_path = f"temp_output_{client_id}.wav"
        safe_response_text = response_text
        
        if len(safe_response_text.strip()) < 10:
            safe_response_text = safe_response_text + ". " * 3
        
        try:
            tts_model.tts_to_file(
                text=safe_response_text, 
                speaker_wav=DEFAULT_SPEAKER_WAV,
                language=DEFAULT_LANGUAGE, 
                file_path=temp_output_path
            )
        except Exception as e:
            print(f"TTS generation error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Speech generation failed: {str(e)}"
            }))
            return
        
        # Read the audio file and convert to base64
        with open(temp_output_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Clean up temporary file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
        print("Sending audio response to client...")
        await websocket.send(json.dumps({
            "type": "audio",
            "audio": audio_base64
        }))
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        await websocket.send(json.dumps({
            "type": "error",
            "message": str(e)
        }))

async def handle_websocket(websocket, path):
    """Handle WebSocket connection"""
    client_id = id(websocket)
    client_contexts[client_id] = {
        "context": "", 
        "history": [],
        "script_type": "reliant_bpo"
    }
    
    print(f"New client connected: {client_id}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data["type"] == "audio":
                    await process_audio(data["data"], websocket)
                    
                elif data["type"] == "context":
                    client_contexts[client_id]["context"] = data["data"]
                    print(f"Received context from client {client_id}")
                
                elif data["type"] == "script_type":
                    script_type = data.get("data", "reliant_bpo")
                    if script_type in BPO_SCRIPTS:
                        client_contexts[client_id]["script_type"] = script_type
                        print(f"Set script type to {script_type} for client {client_id}")
                        
                        await websocket.send(json.dumps({
                            "type": "info",
                            "message": f"Script type set to {script_type}"
                        }))
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Invalid script type: {script_type}"
                        }))
                    
            except json.JSONDecodeError:
                print("Error: Invalid JSON")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
                
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
        process_request=lambda path, request_headers: None,
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
