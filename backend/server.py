import asyncio
import json
import base64
import os
import io
import torch
import websockets
from groq import Groq
from TTS.api import TTS

# Configuration
WEBSOCKET_PORT = 8000
WEBSOCKET_HOST = "0.0.0.0"  # Listen on all interfaces
GROQ_API_KEY = "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"  # Replace with your actual API key

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize TTS model
print("Initializing TTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2').to(device)
print(f"TTS model initialized on {device}.")

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

# Store conversation context for each client
client_contexts = {}

async def process_audio(audio_data, websocket):
    """Process audio data: transcribe, get LLM response, and generate speech"""
    client_id = id(websocket)
    context = client_contexts.get(client_id, {}).get("context", "")
    script_type = client_contexts.get(client_id, {}).get("script_type", "reliant_bpo")  # Default to reliant_bpo
    
    try:
        # Save audio data to a temporary file
        audio_bytes = base64.b64decode(audio_data)
        temp_audio_path = f"temp_audio_{client_id}.wav"
        
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        
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
        
        # Add user message to history
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
        
        # Add the BPO script as the primary system message
        system_message = BPO_SCRIPTS.get(script_type, BPO_SCRIPTS["reliant_bpo"])
        
        # Add additional context if provided
        if context:
            system_message += f"\n\nAdditional context for this conversation: {context}"
        
        # Add general instruction to allow answering general questions
        system_message += "\n\nWhile following the script, you should also be able to answer general questions, but always return to the script flow afterward. Keep asking questions from the script until all parts are completed."
        
        messages.append({
            "role": "system", 
            "content": system_message
        })
        
        # Add conversation history
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
            "script_type": script_type  # Save the current script type
        }
        
        # Send text response to client
        await websocket.send(json.dumps({
            "type": "response",
            "text": response_text
        }))
        
        print("Generating speech from response...")
        # Generate speech from response using the original TTS implementation
        temp_output_path = f"temp_output_{client_id}.wav"
        
        # Ensure the response is long enough to avoid the kernel size error
        # If the response is too short, add padding text that won't be noticeable
        safe_response_text = response_text
        
        # Check if the response is too short (less than 10 characters)
        if len(safe_response_text.strip()) < 10:
            # Add padding that won't affect the meaning
            safe_response_text = safe_response_text + ". " * 3
        
        try:
            # Try to generate speech with the original or padded text
            tts_model.tts_to_file(text=safe_response_text, file_path=temp_output_path)
        except RuntimeError as e:
            if "Kernel size can't be greater than actual input size" in str(e):
                # If we still get the kernel size error, use a fallback message
                print("TTS error with kernel size. Using fallback approach.")
                fallback_text = safe_response_text + " I'm processing your request."
                try:
                    # Try with a longer fallback message
                    tts_model.tts_to_file(text=fallback_text, file_path=temp_output_path)
                except Exception as fallback_error:
                    print(f"Fallback TTS also failed: {fallback_error}")
                    # If all TTS attempts fail, send a pre-recorded or synthesized "I understand" audio
                    # For now, we'll just inform the client about the error
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Unable to generate speech for this response."
                    }))
                    return
            else:
                # For other TTS errors, re-raise
                raise
        
        # Read the audio file and convert to base64
        with open(temp_output_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Clean up temporary file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
        print("Sending audio response to client...")
        # Send audio response to client
        await websocket.send(json.dumps({
            "type": "audio",
            "audio": audio_base64
        }))
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        # Send error message to client
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
        "script_type": "reliant_bpo"  # Default script type
    }
    
    print(f"New client connected: {client_id}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data["type"] == "audio":
                    # Process audio data
                    await process_audio(data["data"], websocket)
                    
                elif data["type"] == "context":
                    # Store context for this client
                    client_contexts[client_id]["context"] = data["data"]
                    print(f"Received context from client {client_id}")
                
                elif data["type"] == "script_type":
                    # Allow client to set script type directly
                    script_type = data.get("data", "reliant_bpo")
                    if script_type in BPO_SCRIPTS:
                        client_contexts[client_id]["script_type"] = script_type
                        print(f"Set script type to {script_type} for client {client_id}")
                        
                        # Send confirmation to client
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
        # Clean up client context when connection is closed
        if client_id in client_contexts:
            del client_contexts[client_id]
        print(f"Client disconnected: {client_id}")

async def main():
    """Start WebSocket server"""
    print(f"Starting WebSocket server on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}...")
    
    # Create a WebSocket server with CORS support
    async with websockets.serve(
        handle_websocket, 
        WEBSOCKET_HOST, 
        WEBSOCKET_PORT,
        # Add CORS headers
        process_request=lambda path, request_headers: None,
        extra_headers=[
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
            ('Access-Control-Allow-Headers', 'Content-Type'),
        ]
    ):
        print(f"WebSocket server running on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Server error: {str(e)}")
