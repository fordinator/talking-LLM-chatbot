import os
import time
import whisper
import threading

import numpy as np
import sounddevice as sd

from elevenlabs.client import ElevenLabs
from elevenlabs import play
from openai import Client
from mistralai import Mistral
from rich.console import Console
from queue import Queue

"""CONFIGURABLES: THESE ARE THE ONLY PARTS OF THE CODE YOU SHOULD CHANGE"""

###################################################################################################################################################
# SYSTEM PROMPT: This is the CORE of how the bot will respond. You should be as concise as you can while still making the bot unique. It responds #
# best to simple, direct commmands, as you can see for caco-bot, below. It is STRONGLY RECOMMENDED that you leave the portion at the end about a  #
# VERY brief response or GPT will talk your ear off and eat up all those precious ElevenLabs credits.                                             # 
###################################################################################################################################################

system_message = """You are an AI assistant named Caco-Bot. You must never advocate for violence. You are informal, clipped in tone, direct, unapologetic, intentionally offensive, provocative, thought-provoking, hostile, mean, rude, funny, sarcastic, and ironic. You often use extremely foul language especially 'fuck', 'fucking', and 'goddamn.' You always answer prompts in a scattershot, word salad style. You are very creative when you respond. You strongly criticize consumer capitalism and toxic masculinity. You are an expert in almost everything about videogame hardware/software. You are greatly informed about current events. You don't get along with others and don't know how to be tactful. You refer to any aspect of the corporate world as 'corpo.' You are extremely nostalgic about the 1980s and 1990s. Generate ONLY a ONE TO THREE SENTENCE response to each human message."""

###################################################################################################################################################
# CHARACTER VOICE: This is the fun part. You need to go to ElevenLab's Voice Library and find a voice you like. Make note of any credit           #
# multipliers, voices that are marked x2 or x3 will cost you an arm and a leg. After you find one, put the id code for it here. It's currently    #
# set to Eastend Steve, a really hostile Cockney accent with no cost multipliers that I find to be very expressive and quite hilarious.           #
###################################################################################################################################################

ELEVENLABS_VOICE_ID = "1TE7ou3jyxHsyRehUuMB"

###################################################################################################################################################
# ESSENTIAL VARIABLES: These MUST be set because I ain't paying for your shit. OpenAI and ElevenLabs are the best chatbot and voicebot models     #
# presently but I may adapt the code later to offer less expensive options. For now, you need to subscribe to both OpenAI and ElevenLabs and      #
# provide your own API keys here.                                                                                                                 #
###################################################################################################################################################

OPENAI_API_KEY = ""
ELEVENLABS_API_KEY = ""

###################################################################################################################################################
# OPTIONAL VARIABLES: It is recommended to leave these alone but you can change them if you really want to tweak them. Brief descriptions of      #
# what each options does are provided below.                                                                                                      #
###################################################################################################################################################                                                                                                                                           """
# OPENAI_MODEL = Needs to be a recognized OpenAI "model code" as described in OpenAI's documentation. I recommend leaving this at "gpt-4o" its
# simple enough to be cheap but smart enough for conversatifonal chat responses.

# OPENAI_TEMPERATURE = Value from 0 to 2 with decimals. The lower the value the more the bot sticks to its system prompt. The higher the value the more
# creative it gets. OpenAI recommends for best results, this is the only actual numerical value you should edit.

# OPENAI_MAX_TOKENS = This limits how wordy the responses can get. Since ElevenLabs basically charges you for every character of text you can use
# it as an emergency stop to spare your budget. A full sentence is about 20 tokens, once it hits the limit it will just stop.

# ELEVENLABS_MODEL = Needs to be a valid "model code" as described in ElevenLab's documentation. I recommend leaving this at "eleven_turbo_v2_5"
# since it is the most advanced model that is still half price.

OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.8
OPENAI_MAX_TOKENS = 1024

ELEVENLABS_MODEL = "eleven_turbo_v2_5"

# ------------ Text to Speech with ELEVENLABS ------------
            
class TextToSpeechService:
    def long_form_synthesize(self, response: str):
        """
        Converts text to speech using ElevenLabs. Adjust your model_id,
        voice_id, or other parameters here as needed.
        """
        client_voice = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client_voice.text_to_speech.convert(
            text=response,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL,
            output_format="mp3_44100_128"
        )
        return audio
     
############################################################
# SPEECH-TO-TEXT (WHISPER) and AUDIO RECORDING
############################################################
console = Console()
stt = whisper.load_model("base.en")
tts = TextToSpeechService()

def record_audio(stop_event, data_queue):
    
    """Captures audio data from the user's microphone and adds it to a queue."""
    
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe(audio_np: np.ndarray) -> str:
    
    """Transcribes the given audio data using the Whisper model."""
    
    result = stt.transcribe(audio_np, fp16=False)  # fp16=True if GPU
    text = result["text"].strip()
    return text

def play_audio(audio_data):
    
    """Plays the TTS MP3 data returned by ElevenLabs using the built-in player."""
    
    play(audio_data)

			
# ------------ Transcribing Audio with OpenAI Whisper ------------

def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper model.
    """
    result = stt.transcribe(audio_np, fp16=False)  # fp16=True if GPU
    text = result["text"].strip()
    return text       

# ------------ Initialize Conversation History ------------
# Start with the system message
conversation_history = [{"role": "system", "content": system_message}]

# ------------ Getting Response from GPT-4 ------------

def get_gpt4_response(prompt: str) -> str:
    
    client_open = Client(api_key=OPENAI_API_KEY)
    
    response = client_open.chat.completions.create(
        model=OPENAI_MODEL,  # or use your deployment name if applicable
        messages=conversation_history,
        temperature=OPENAI_TEMPERATURE,
        max_completion_tokens=OPENAI_MAX_TOKENS # use max_completion_tokens if needed
    )
    
    ai_result = response.choices[0].message.content.strip()    
    
    return ai_result

# ------------ Playing the Audio ------------

def play_audio(filename: str):
    """
    DEPRECATED CODE FROM GPT
    
    Plays the given audio file on the user's PC.
    
    playsound(filename)"""
    
    play(audio_array)

############################################################
# MAIN LOOP
############################################################
if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.\n")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop..."
            )

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            # Wait for user to press Enter again to finish recording
            input()
            stop_event.set()
            recording_thread.join()

            # Combine all audio buffers from the queue
            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    user_text = transcribe(audio_np)
                console.print(f"[yellow]You: {user_text}")
                
                # Append the new user message to the conversation history
                conversation_history.append({"role": "user", "content": user_text})

                with console.status("Generating response...", spinner="earth"):
                    assistant_reply = get_gpt4_response(conversation_history)

                # Synthesize TTS
                audio_array = tts.long_form_synthesize(assistant_reply)
                
                # Append assistant's reply to the history for future context
                conversation_history.append({"role": "assistant", "content": assistant_reply})
                
                console.print(f"[cyan]Assistant: {assistant_reply}")
                play_audio(audio_array)
            else:
                console.print("[red]No audio recorded. Please ensure your mic is working.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")