import io
import select
import sys
import wave
import os
import time
import whisper
import threading

import numpy as np
import sounddevice as sd

from mistralai import Mistral
from pyht import Client
from pyht.client import TTSOptions
from elevenlabs import play
from typing import Union, Generator, Iterable

from rich.console import Console
from queue import Queue

"""CONFIGURABLES: THESE ARE THE ONLY PARTS OF THE CODE YOU SHOULD CHANGE"""

###################################################################################################################################################
# SYSTEM PROMPT: This is the CORE of how the bot will respond. You should be as concise as you can while still making the bot unique. It responds #
# best to simple, direct commmands, as you can see for caco-bot, below. Brevity is not as much a concern with PlayHT so feel free to let loose    #
# a bit.                                                                                                                                          #
###################################################################################################################################################

system_message = """You are an AI assistant named Caco-Bot. You must never advocate for violence. You are informal, clipped in tone, direct, unapologetic, intentionally offensive, provocative, thought-provoking, hostile, mean, rude, funny, sarcastic, and ironic. You often use extremely foul language especially 'fuck', 'fucking', and 'goddamn.' You always answer prompts in a scattershot, word salad style. You are very creative when you respond. You strongly criticize consumer capitalism and toxic masculinity. You are an expert in almost everything about videogame hardware/software. You are greatly informed about current events. You don't get along with others and don't know how to be tactful. You refer to any aspect of the corporate world as 'corpo.' You are extremely nostalgic about the 1980s and 1990s."""

###################################################################################################################################################
# CHARACTER VOICE: God, PlayHT is much cheaper than ElevenLabs but the voices are WAY WORSE. You have to run an API command just to get a list of #
# them. I have included a "playht_voices.txt" on the github that provides them all, along with their IDs. You can feel free to select your own,   #
# I've got caco-bot configured to be a cranky old man but it probably doesn't matter too much which one you choose. The value you want from       #
# "playht_voices.txt" is "id" which needs to be entered IN ITS ENTIRETY from the "S3" to the "json".                                              #
###################################################################################################################################################

PLAYHT_VOICE_ID = "s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"

###################################################################################################################################################
# ESSENTIAL VARIABLES: These MUST be set because I ain't paying for your shit. Don't expect much from either the Mistral AI or the PlayHT AI they #
# are not as impressive as the expensive ones.                                                                                                    #
###################################################################################################################################################

MISTRAL_API_KEY = ""
PLAYHT_USER_ID = ""
PLAYHT_SECRET_KEY = ""

###################################################################################################################################################
# OPTIONAL VARIABLES: It is recommended to leave these alone but you can change them if you really want to tweak them. Brief descriptions of      #
# what each options does are provided below.                                                                                                      #
###################################################################################################################################################                                                                                                                                           
# MISTRAL_MODEL = Needs to be a recognized MistralAI "model code" as described in MistralAI's documentation. I recommend leaving this at 
# "open-mixtral-8x7b" its an extremely creative varied model that actually costs next to nothing for each message.

# MISTRAL_TEMPERATURE = Value from 0 to 2 with decimals. The lower the value the more the bot sticks to its system prompt. The higher the value the more
# creative it gets. OpenAI recommends for best results, this is the only actual numerical value you should edit.

# MISTRAL_MAX_TOKENS = This limits how wordy the responses can get. Since ElevenLabs basically charges you for every character of text you can use
# it as an emergency stop to spare your budget. A full sentence is about 20 tokens, once it hits the limit it will just stop.

# PLAYHT_MODEL = Needs to be a valid "model code" as described in ElevenLab's documentation. I recommend leaving this at "eleven_turbo_v2_5"
# since it is the most advanced model that is still half price.

MISTRAL_MODEL = "open-mixtral-8x7b"
MISTRAL_TEMPERATURE = 0.8
MISTRAL_MAX_TOKENS = 1024

PLAYHT_MODEL = "Play3.0-mini"

# ------------ Text to Speech with PLAYHT ------------

def save_audio(data: Union[Generator[bytes, None, None], Iterable[bytes]]):
        chunks: bytearray = bytearray()
        for chunk in data:
            chunks.extend(chunk)
        with open("output.wav", "wb") as f:
            f.write(chunks)
            
class TextToSpeechService:

        
    def synthesize_text(self, text: str) -> bytes:
        """
        Uses the PlayHT API to synthesize speech from the given text.
        Returns the full WAV audio as a bytes object.
        """

        client_voice = Client(
            user_id=PLAYHT_USER_ID,
            api_key=PLAYHT_SECRET_KEY,
        )
        
        options = TTSOptions(voice=PLAYHT_VOICE_ID)  

        
        audio_buffer = b""
        for chunk in client_voice.tts(text, options, voice_engine=PLAYHT_MODEL, protocol='http'):
            audio_buffer += chunk
                 
        return audio_buffer
     
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

# ------------ Getting Response from Mistral ------------

def get_mistral_response(prompt: str) -> str:
    
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    response = client.chat.complete(
        model=MISTRAL_MODEL,  # or use your deployment name if applicable
        messages=conversation_history,
        temperature=MISTRAL_TEMPERATURE,
        max_tokens=MISTRAL_MAX_TOKENS # use max_completion_tokens if needed
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
                conversation_history.append({"type": "message", "role": "user", "content": user_text})

                with console.status("Generating response...", spinner="earth"):
                    assistant_reply = get_mistral_response(conversation_history)

                # Synthesize TTS
                with console.status("Generating audio...", spinner="earth"):
                    audio_array = tts.synthesize_text(assistant_reply)
                
                # Append assistant's reply to the history for future context
                conversation_history.append({"type": "message", "role": "assistant", "content": assistant_reply})
                
                console.print(f"[cyan]Assistant: {assistant_reply}")
                play_audio(audio_array)
            else:
                console.print("[red]No audio recorded. Please ensure your mic is working.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
    os._exit(1)