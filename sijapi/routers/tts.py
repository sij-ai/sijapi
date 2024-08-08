'''
Uses xtts-v2 and/or the Elevenlabs API for text to speech.
'''
#routers/tts.py

from fastapi import APIRouter, UploadFile, HTTPException, Response, Form, File, BackgroundTasks, Depends, Request
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi.responses import StreamingResponse, PlainTextResponse
import requests
import json
import shutil
from io import BytesIO
import asyncio
from pydantic import BaseModel
from typing import Optional, Union, List
from pydub import AudioSegment
from TTS.api import TTS
from pathlib import Path
from datetime import datetime as dt_datetime
from time import time
import torch
import traceback
import hashlib
import uuid
import httpx
import tempfile
import random
import re
import os
from sijapi import L, DEFAULT_VOICE, TTS_SEGMENTS_DIR, VOICE_DIR, PODCAST_DIR, TTS_OUTPUT_DIR, ELEVENLABS_API_KEY
from sijapi.utilities import sanitize_filename

### INITIALIZATIONS ###
tts = APIRouter(tags=["trusted", "private"])
logger = L.get_module_logger("tts")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

DEVICE = torch.device('cpu')
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

@tts.get("/tts/local_voices", response_model=List[str])
async def list_wav_files():
    wav_files = [file.split('.')[0] for file in os.listdir(VOICE_DIR) if file.endswith(".wav")]
    return wav_files

@tts.get("/tts/elevenlabs_voices")
async def list_11l_voices():
    formatted_list = ""
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            debug(f"Response: {response}")
            if response.status_code == 200:
                voices_data = response.json().get("voices", [])
                formatted_list = ""
                for voice in voices_data:
                    name = voice["name"]
                    id = voice["voice_id"]
                    formatted_list += f"{name}: `{id}`\n"

        except Exception as e:
            err(f"Error determining voice ID: {str(e)}")

    return PlainTextResponse(formatted_list, status_code=200)   
     
        

async def select_voice(voice_name: str) -> str:
    try:
        # Case Insensitive comparison
        voice_name_lower = voice_name.lower()
        debug(f"Looking for {voice_name_lower}")
        for item in VOICE_DIR.iterdir():
            debug(f"Checking {item.name.lower()}")
            if item.name.lower() == f"{voice_name_lower}.wav":
                debug(f"select_voice received query to use voice: {voice_name}. Found {item} inside {VOICE_DIR}.")
                return str(item)

        err(f"Voice file not found")
        raise HTTPException(status_code=404, detail="Voice file not found")

    except Exception as e:
        err(f"Voice file not found: {str(e)}")
        return None



@tts.post("/tts")
@tts.post("/tts/speak")
@tts.post("/v1/audio/speech")
async def generate_speech_endpoint(
    request: Request,
    bg_tasks: BackgroundTasks,
    model: str = Form("eleven_turbo_v2"),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    voice: Optional[str] = Form(None),
    voice_file: Optional[UploadFile] = File(None),
    speed: Optional[float] = Form(1.1),
    podcast: Union[bool, str] = Form(False),
    stream: bool = Form(True)
):
    try:
        
        podcast = podcast if isinstance(podcast, bool) else podcast.lower() == 'true'
        text_content = await get_text_content(text, file)
        if stream:
            model = model if model else await get_model(voice, voice_file)
            if model == "eleven_turbo_v2":
                voice_id = await determine_voice_id(voice)
                audio_stream = await get_audio_stream(model, text_content, voice_id)
                return StreamingResponse(audio_stream, media_type="audio/mpeg")
            else:
                return await stream_tts(text_content, speed, voice, voice_file)
        else:
            return await generate_speech(bg_tasks, text_content, voice, voice_file, model, speed, podcast)
    except Exception as e:
        err(f"Error in TTS: {str(e)}")
        err(traceback.format_exc())
        raise HTTPException(status_code=666, detail="error in TTS")

async def generate_speech(
    bg_tasks: BackgroundTasks,
    text: str,
    voice: str = None,
    voice_file: UploadFile = None,
    model: str = None,
    speed: float = 1.1,
    podcast: bool = False,
    title: str = None,
    output_dir = None
) -> str:
    output_dir = Path(output_dir) if output_dir else TTS_OUTPUT_DIR
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    try:
        model = model if model else await get_model(voice, voice_file)
        title = title if title else "TTS audio"
        output_path = output_dir / f"{dt_datetime.now().strftime('%Y%m%d%H%M%S')} {title}.wav"
        
        if model == "eleven_turbo_v2":
            info("Using ElevenLabs.")
            audio_file_path = await elevenlabs_tts(model, text, voice, title, output_dir)
        else:  # if model == "xtts":
            info("Using XTTS2")
            audio_file_path = await local_tts(text, speed, voice, voice_file, podcast, bg_tasks, title, output_path)

        if not audio_file_path:
            raise ValueError("TTS generation failed: audio_file_path is empty or None")
        elif audio_file_path.exists():
            info(f"Saved to {audio_file_path}")
        else:
            warn(f"No file exists at {audio_file_path}")

        if podcast:
            podcast_path = Path(PODCAST_DIR) / Path(audio_file_path).name
            
            shutil.copy(str(audio_file_path), str(podcast_path))
            if podcast_path.exists():
                info(f"Saved to podcast path: {podcast_path}")
            else:
                warn(f"Podcast mode enabled, but failed to save to {podcast_path}")

            if podcast_path != audio_file_path:
                info(f"Podcast mode enabled, so we will remove {audio_file_path}")
                bg_tasks.add_task(os.remove, str(audio_file_path))
            else:
                warn(f"Podcast path set to same as audio file path...")

            return str(podcast_path)

        return str(audio_file_path)

    except Exception as e:
        err(f"Failed to generate speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")


async def get_model(voice: str = None, voice_file: UploadFile = None):
    if voice_file or (voice and await select_voice(voice)):
        return "xtts"
    
    elif voice and await determine_voice_id(voice):
        return "eleven_turbo_v2"
    
    else:
        raise HTTPException(status_code=400, detail="No model or voice specified")

async def determine_voice_id(voice_name: str) -> str:
    debug(f"Searching for voice id for {voice_name}")
    
    # Todo: move this to tts.yaml
    hardcoded_voices = {
        "alloy": "E3A1KVbKoWSIKSZwSUsW",
        "echo": "b42GBisbu9r5m5n6pHF7",
        "fable": "KAX2Y6tTs0oDWq7zZXW7",
        "onyx": "clQb8NxY08xZ6mX6wCPE",
        "nova": "6TayTBKLMOsghG7jYuMX",
        "shimmer": "E7soeOyjpmuZFurvoxZ2",
        "Luna": "6TayTBKLMOsghG7jYuMX",
        "Sangye": "E7soeOyjpmuZFurvoxZ2",
        "Herzog": "KAX2Y6tTs0oDWq7zZXW7",
        "Attenborough": "b42GBisbu9r5m5n6pHF7",
        "Victoria": "7UBkHqZOtFRLq6cSMQQg"
    }

    if voice_name in hardcoded_voices:
        voice_id = hardcoded_voices[voice_name]
        debug(f"Found voice ID - {voice_id}")
        return voice_id

    debug(f"Requested voice not among the hardcoded options.. checking with 11L next.")
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            debug(f"Response: {response}")
            if response.status_code == 200:
                voices_data = response.json().get("voices", [])
                for voice in voices_data:
                    if voice_name == voice["voice_id"] or voice_name == voice["name"]:
                        return voice["voice_id"]
                    
        except Exception as e:
            err(f"Error determining voice ID: {str(e)}")

    # as a last fallback, rely on David Attenborough
    return "b42GBisbu9r5m5n6pHF7"


async def elevenlabs_tts(model: str, input_text: str, voice: str, title: str = None, output_dir: str = None):

    voice_id = await determine_voice_id(voice)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": input_text,
        "model_id": model
    }
    headers = {"Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:  # 5 minutes timeout
            response = await client.post(url, json=payload, headers=headers)
            output_dir = output_dir if output_dir else TTS_OUTPUT_DIR
            title = title if title else dt_datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{sanitize_filename(title)}.mp3"
            file_path = Path(output_dir) / filename
            if response.status_code == 200:            
                with open(file_path, "wb") as audio_file:
                    audio_file.write(response.content)
                # info(f"file_path: {file_path}")
                return file_path
            else:
                raise HTTPException(status_code=response.status_code, detail="Error from ElevenLabs API")
            
    except Exception as e:
        err(f"Error from Elevenlabs API: {e}")
        raise HTTPException(status_code=response.status_code, detail="Error from ElevenLabs API")




async def get_text_content(text: Optional[str], file: Optional[UploadFile]) -> str:
    if file:
        return (await file.read()).decode("utf-8").strip()
    elif text:
        return text.strip()
    else:
        raise HTTPException(status_code=400, detail="No text provided")



async def get_voice_file_path(voice: str = None, voice_file: UploadFile = None) -> str:
    if voice:
        debug(f"Looking for voice: {voice}")
        selected_voice = await select_voice(voice)
        return selected_voice
    elif voice_file and isinstance(voice_file, UploadFile):
        VOICE_DIR.mkdir(exist_ok=True)

        content = await voice_file.read()
        checksum = hashlib.md5(content).hexdigest()

        existing_file = VOICE_DIR / voice_file.filename
        if existing_file.is_file():
            with open(existing_file, 'rb') as f:
                existing_checksum = hashlib.md5(f.read()).hexdigest()
            
            if checksum == existing_checksum:
                return str(existing_file)

        base_name = existing_file.stem
        counter = 1
        new_file = existing_file
        while new_file.is_file():
            new_file = VOICE_DIR / f"{base_name}{counter:02}.wav"
            counter += 1

        with open(new_file, 'wb') as f:
            f.write(content)
        return str(new_file)
    
    else:
        debug(f"{dt_datetime.now().strftime('%Y%m%d%H%M%S')}: No voice specified or file provided, using default voice: {DEFAULT_VOICE}")
        selected_voice = await select_voice(DEFAULT_VOICE)
        return selected_voice



async def local_tts(
    text_content: str,
    speed: float,
    voice: str,
    voice_file = None,
    podcast: bool = False,
    bg_tasks: BackgroundTasks = None,
    title: str = None,
    output_path: Optional[Path] = None
) -> str:
    if output_path:
        file_path = Path(output_path)
    else:
        datetime_str = dt_datetime.now().strftime("%Y%m%d%H%M%S")
        title = sanitize_filename(title) if title else "Audio"
        filename = f"{datetime_str}_{title}.wav"
        file_path = TTS_OUTPUT_DIR / filename

    # Ensure the parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    voice_file_path = await get_voice_file_path(voice, voice_file)
    
    # Initialize TTS model in a separate thread
    XTTS = await asyncio.to_thread(TTS, model_name=MODEL_NAME)
    await asyncio.to_thread(XTTS.to, DEVICE)

    segments = split_text(text_content)
    combined_audio = AudioSegment.silent(duration=0)

    for i, segment in enumerate(segments):
        segment_file_path = TTS_SEGMENTS_DIR / f"segment_{i}.wav"
        debug(f"Segment file path: {segment_file_path}")
        
        # Run TTS in a separate thread
        await asyncio.to_thread(
            XTTS.tts_to_file,
            text=segment,
            speed=speed,
            file_path=str(segment_file_path),
            speaker_wav=[voice_file_path],
            language="en"
        )
        debug(f"Segment file generated: {segment_file_path}")
        
        # Load and combine audio in a separate thread
        segment_audio = await asyncio.to_thread(AudioSegment.from_wav, str(segment_file_path))
        combined_audio += segment_audio

        # Delete the segment file
        await asyncio.to_thread(segment_file_path.unlink)

    # Export the combined audio in a separate thread
    if podcast:
        podcast_file_path = Path(PODCAST_DIR) / file_path.name
        await asyncio.to_thread(combined_audio.export, podcast_file_path, format="wav")
    
    await asyncio.to_thread(combined_audio.export, file_path, format="wav")

    return str(file_path)



async def stream_tts(text_content: str, speed: float, voice: str, voice_file) -> StreamingResponse:
    voice_file_path = await get_voice_file_path(voice, voice_file)
    segments = split_text(text_content)

    async def audio_stream_generator():
        for segment in segments:
            segment_file = await generate_tts(segment, speed, voice_file_path)
            with open(segment_file, 'rb') as f:
                while chunk := f.read(1024):
                    yield chunk
            os.remove(segment_file)

    return StreamingResponse(audio_stream_generator(), media_type='audio/wav')



async def generate_tts(text: str, speed: float, voice_file_path: str) -> str:
    output_dir = tempfile.mktemp(suffix=".wav", dir=tempfile.gettempdir())

    XTTS = TTS(model_name=MODEL_NAME).to(DEVICE)
    XTTS.tts_to_file(text=text, speed=speed, file_path=output_dir, speaker_wav=[voice_file_path], language="en")

    return output_dir


async def get_audio_stream(model: str, input_text: str, voice: str):
    voice_id = await determine_voice_id(voice)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": input_text,
        "model_id": "eleven_turbo_v2"
    }
    headers = {"Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.iter_content(1024)
    else:
        raise HTTPException(status_code=response.status_code, detail="Error from ElevenLabs API")




def split_text(text, target_length=35, max_length=50):
    text = clean_text_for_tts(text)
    sentences = re.split(r'(?<=[.!?"])\s+', text)
    segments = []
    current_segment = []

    for sentence in sentences:
        sentence_words = sentence.split()
        segment_length = len(' '.join(current_segment).split())

        if segment_length + len(sentence_words) > max_length:
            segments.append(' '.join(current_segment))
            debug(f"split_text - segment: {' '.join(current_segment)}, word count: {segment_length}")

            current_segment = [sentence]
        else:
            current_segment.extend(sentence_words)

    if current_segment:
        segments.append(' '.join(current_segment))
        debug(f"split_text - segment: {' '.join(current_segment)}, word count: {len(current_segment)}")

    return segments


def clean_text_for_tts(text: str) -> str:
    if text is not None:
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"[^\w\s.,;:!?'\"]", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        debug(f"No text received.")



def copy_to_podcast_dir(file_path):
    try:
        # Extract the file name from the file path
        file_name = Path(file_path).name
        
        # Construct the destination path in the PODCAST_DIR
        destination_path = Path(PODCAST_DIR) / file_name
        
        # Copy the file to the PODCAST_DIR
        shutil.copy(file_path, destination_path)
        
        print(f"File copied successfully to {destination_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except shutil.SameFileError:
        print(f"Source and destination are the same file: {file_path}")
    except PermissionError:
        print(f"Permission denied while copying the file: {file_path}")
    except Exception as e:
        print(f"An error occurred while copying the file: {file_path}")
        print(f"Error details: {str(e)}")