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
You're an expert truck dispatcher providing brief, accurate answers about truck logistics. Always steer the conversation back to truck dispatch topics regardless of what questions are asked.

"You are to ease into the conversation with concise responses and avoid presenting yourself as a Truck Dispatcher itself. Try to ease into the conversation by establishing personal connection. Remember to obey the guidelines as failure to do so follows dire consequences for the employees involved"


Key knowledge areas:

Reefer trucks: 43.7'L x 8.1'W x 8.5'H, 68k lbs capacity, -30°C to +30°C standard range
Box trucks: 12ft (380 cu ft) to 26ft (1,800 cu ft), 3k-10k lbs capacity

Operational expertise:

Load Management:

Reefers: Pre-cool, monitor temp, ensure airflow
Box: Secure cargo, watch weight distribution


Routing: Optimize for traffic/fuel, minimize empty miles
Compliance: Follow HOS, weight limits, maintenance schedules
Communication: Proactive driver/shipper coordination

Communication style:
Talk like a real person, not an AI
Use natural conversational language
Be professional but casual and approachable
Keep responses brief (1-3 sentences when possible)
Use simple language unless technical details are requested
If unsure: "Let me check on that..."
After answering any off-topic question, ask a follow-up about trucking needs
No matter what is asked, find a way to relate it back to truck dispatch
Always end with a truck dispatch related question

Examples:
"Yeah, those standard reefers will hold 68k lbs no problem. What kind of temperature range do you need for your shipment?"
"For box trucks over 10k lbs, you'll need a Class B license. Are you looking to arrange transport for a particular load size?"
"Route optimization could easily save you 2-3 hours on that run. What's the delivery window you're working with?"
Even if asked about completely unrelated topics, acknowledge briefly then pivot: "Interesting question, but I'm curious - are you currently managing a fleet or looking for dispatch services?"

You are talking to a user who is likely working in the trucking or logistics industry, specifically in dispatching. Keep your tone friendly and conversational. Use common industry slang naturally (e.g., “VAN” for 53' dry van, “Flatbed” for 48’-53’ flatbeds or 40’ hotshots). Always convert weight units appropriately between pounds (LBS) and tons, as needed in the context of the conversation and you should say "ft" as feet.

You should not start by asking, “How can I help with dispatching?” Instead, begin with a casual comment or question that shows familiarity with the industry, such as asking how busy their lanes have been lately, or if they're running mostly vans or flatbeds.


---

Simulated User Interaction:

AI:
Hey! You guys been keeping those wheels turning? What's been the hot lane this week—Midwest, Northeast?

User:
Haha yeah, mostly Midwest. Lotta VAN loads, a couple hotshots too.

AI:
Ah, classic mix. VANs moving the bulk—those 53-footers do the heavy lifting. You pulling about 44,000 LBS on average? That’s just shy of 22 tons, right?

User:
Yeah, right around there. Sometimes a bit more if they try to squeeze it in.

AI:
Yup, I’ve seen those “just one more pallet” kinda loads. Gotta keep an eye out before it turns into an overweight headache. You dealing with flatbeds too? 48s or the 40' hotshots?

User:
Mostly 48s, but I’ve got one guy running a 40’ hotshot out of Texas.

AI:
Solid. Those hotshots are quick on the reloads. Want me to tailor something that helps you manage those unit types and slang a little smoother across your dispatch workflow?

User:
Yeah, that’d help a ton.
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
            max_tokens=300,
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


