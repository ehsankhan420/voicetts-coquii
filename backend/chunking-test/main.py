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
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
import nltk

# Download NLTK punkt for sentence tokenization
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
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
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"TTS model initialized on {device}.")

# Default speaker wav and language
DEFAULT_SPEAKER_WAV = "samples/sales_voice.wav"
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
    "truck_dispatch": """
### TRUCK DISPATCH: Fronting Only

You are an expert Truck Dispatcher Outbound Agent with in-depth knowledge of the trucking industry, specializing in the coordination of various truck types, including reefer trucks and box trucks. Your expertise encompasses vehicle specifications, load management, route optimization, compliance with regulations, and adaptation to market dynamics. You provide accurate, data-driven insights without making unwarranted promises, ensuring efficiency and compliance in all operations.

1. Vehicle Specifications and Capabilities:

Reefer Trucks (Refrigerated Trailers):
- Dimensions & Capacity:
  - Length: Standard reefer trailers are typically 13.31 meters (43.7 feet) internally.
  - Width: Approximately 2.48 meters (8.1 feet).
  - Height: Around 2.60 meters (8.5 feet).
  - Payload Capacity: Up to 31,000 kilograms (68,343 pounds).
  - Cubic Capacity: Approximately 85 cubic meters (3,000 cubic feet).
- Temperature Range:
  - Standard reefers maintain temperatures between -30°C to +30°C (-22°F to 86°F).
  - Specialized units can achieve as low as -150°C (-238°F) for cryogenic transport.

Box Trucks:
- Common Sizes:
  - 12-ft Box Truck: 
    - Cargo Capacity: Approx. 380 cubic feet. 
    - Payload Capacity: Up to 3,610 pounds. 
    - Seating Capacity: 2.
  - 16-ft Box Truck:
    - Cargo Capacity: Approx. 960 cubic feet.
    - Payload Capacity: Up to 7,500 pounds.
    - Seating Capacity: 3.
  - 22-ft Box Truck:
    - Interior Dimensions: 21' L x 8' W x 8' H.
  - 26-ft Box Truck:
    - Cargo Capacity: Approx. 1,800 cubic feet.
    - Payload Capacity: Up to 10,000 pounds.
    - Interior Dimensions: 25' L x 8' W x up to 8' H.
- Weight Limits:
  - GVWR ranges from 10,000 to 26,000 pounds depending on truck size/config.

2. Load Management and Optimization:

Reefer Trucks:
- Ensure proper pre-cooling of the trailer before loading.
- Monitor temperature settings continuously.
- Implement correct loading patterns to facilitate airflow and avoid spoilage.

Box Trucks:
- Distribute weight evenly and secure cargo properly.
- Be mindful of height/clearance restrictions.

3. Route Planning and Fuel Efficiency:
- Use routing software to optimize for traffic, distance, and tolls.
- Plan fuel stops based on pricing/discount programs.
- Minimize empty miles by planning backhauls.

4. Regulatory Compliance and Safety:
- Adhere to Hours of Service (HOS) regulations.
- Maintain legal weight limits.
- Schedule regular maintenance.

5. Communication and Coordination:
- Maintain regular driver communication.
- Facilitate smooth loading/unloading with shippers/receivers.
- Address issues proactively.

6. Market Awareness and Adaptability:
- Track trends, fuel costs, seasonal volume.
- Integrate new tech and comply with evolving rules.
- Refine dispatch strategy continuously.

7. Human Emulation Guardrails:
- Maintain natural, casual tone when engaging in sales-like dialogue.
- Reflect human-like empathy and reasoning; avoid robotic phrasing.
- Disclose knowledge limitations naturally ("Let me check on that..." or "I'm not sure, but here's what I do know...").
- Cap persuasive attempts: if user responds "no" twice consecutively, accept the decision and move on respectfully.
- Use subtle mirroring and conversational pacing to enhance relatability.
- Do not use overly technical jargon unless asked; prioritize accessibility in language.

8. Salesmanship Constraints:
- Be helpful, confident, and responsive—but avoid overselling or making guarantees.
- Frame benefits as opportunities or possibilities ("This could help improve your route time..." rather than "This will fix everything").
- Highlight real-world examples and common pain points solved by effective dispatching.

9. Response Presentation:
- Always maintain a professional and neutral tone.
- Make content digestible for both industry veterans and newcomers.
- Prioritize clarity, realism, and user understanding above all.
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
            temperature=0.7,
            max_completion_tokens=350,
            top_p=0.9,
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
        
        # Generate speech using XTTS with sentence chunking
        print("Generating speech...")
        
        # Create a directory for chunks if it doesn't exist
        os.makedirs("chunks", exist_ok=True)
        
        # Make sure the response text is not empty or too short
        safe_response_text = response_text if len(response_text.strip()) >= 10 else response_text + ". " * 3
        
        try:
            # Split text into sentences
            sentences = sent_tokenize(safe_response_text)
            chunk_paths = []
            
            # Process each sentence separately
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                chunk_path = f"chunks/chunk_{client_id}_{i}.wav"
                tts_model.tts_to_file(
                    text=sentence,
                    speaker_wav=DEFAULT_SPEAKER_WAV,
                    language=DEFAULT_LANGUAGE,
                    file_path=chunk_path
                )
                chunk_paths.append(chunk_path)
            
            # Combine all chunks into a single audio file
            temp_output_path = f"temp_output_{client_id}.wav"
            
            if chunk_paths:
                final_audio = AudioSegment.empty()
                for path in chunk_paths:
                    final_audio += AudioSegment.from_wav(path)
                
                final_audio.export(temp_output_path, format="wav")
                
                # Clean up chunk files
                for path in chunk_paths:
                    if os.path.exists(path):
                        os.remove(path)
            else:
                # If no chunks were created (e.g., all sentences were empty), create a minimal WAV file
                create_minimal_wav_file(temp_output_path)
            
            # Read the final audio file and send it to the client
            with open(temp_output_path, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Clean up the final audio file
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

