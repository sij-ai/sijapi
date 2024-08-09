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
from sijapi import L, API, Dir, Tts, TTS_SEGMENTS_DIR, VOICE_DIR, TTS_OUTPUT_DIR
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

@tts.get("/tts/local_voices", response_model=List[str])
async def list_wav_files():
    wav_files = [file.split('.')[0] for file in os.listdir(VOICE_DIR) if file.endswith(".wav")]
    return wav_files

@tts.get("/tts/elevenlabs_voices")
async def list_11l_voices():
    formatted_list = ""
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": Tts.elevenlabs.key}
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
    debug(f"Entering generate_speech function")
    debug(f"API.EXTENSIONS: {API.EXTENSIONS}")
    debug(f"Type of API.EXTENSIONS: {type(API.EXTENSIONS)}")
    debug(f"Dir of API.EXTENSIONS: {dir(API.EXTENSIONS)}")
    debug(f"Tts config: {Tts}")
    debug(f"Type of Tts: {type(Tts)}")
    debug(f"Dir of Tts: {dir(Tts)}")

    output_dir = Path(output_dir) if output_dir else TTS_OUTPUT_DIR
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    try:
        model = model if model else await get_model(voice, voice_file)
        title = title if title else "TTS audio"
        output_path = output_dir / f"{dt_datetime.now().strftime('%Y%m%d%H%M%S')} {title}.wav"
        
        debug(f"Model: {model}")
        debug(f"API.EXTENSIONS.elevenlabs: {getattr(API.EXTENSIONS, 'elevenlabs', None)}")
        debug(f"API.EXTENSIONS.xtts: {getattr(API.EXTENSIONS, 'xtts', None)}")

        if model == "eleven_turbo_v2" and getattr(API.EXTENSIONS, 'elevenlabs', False):
            info("Using ElevenLabs.")
            audio_file_path = await elevenlabs_tts(model, text, voice, title, output_dir)
        elif getattr(API.EXTENSIONS, 'xtts', False):
            info("Using XTTS2")
            audio_file_path = await local_tts(text, speed, voice, voice_file, podcast, bg_tasks, title, output_path)
        else:
            err(f"No TTS module enabled!")
            raise ValueError("No TTS module enabled")

        if not audio_file_path:
            raise ValueError("TTS generation failed: audio_file_path is empty or None")
        elif audio_file_path.exists():
            info(f"Saved to {audio_file_path}")
        else:
            warn(f"No file exists at {audio_file_path}")

        if podcast:
            podcast_path = Path(Dir.PODCAST) / Path(audio_file_path).name
            
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
        err(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")



async def get_model(voice: str = None, voice_file: UploadFile = None):
    if (voice_file or (voice and await select_voice(voice))) and API.EXTENSIONS.xtts:
        return "xtts"
    
    elif voice and await determine_voice_id(voice) and API.EXTENSIONS.elevenlabs:
        return "eleven_turbo_v2"
    
    else:
        err(f"No model or voice specified, or no TTS module loaded")
        raise HTTPException(status_code=400, detail="No model or voice specified, or no TTS module loaded")

async def determine_voice_id(voice_name: str) -> str:
    debug(f"Searching for voice id for {voice_name}")
    debug(f"Tts.elevenlabs.voices: {Tts.elevenlabs.voices}")
    
    # Check if the voice is in the configured voices
    if voice_name in Tts.elevenlabs.voices:
        voice_id = Tts.elevenlabs.voices[voice_name]
        debug(f"Found voice ID in config - {voice_id}")
        return voice_id
    
    debug(f"Requested voice not among the voices specified in config/tts.yaml. Checking with ElevenLabs API using api_key: {Tts.elevenlabs.key}.")
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": Tts.elevenlabs.key}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            debug(f"Response status: {response.status_code}")
            if response.status_code == 200:
                voices_data = response.json().get("voices", [])
                for voice in voices_data:
                    if voice_name == voice["voice_id"] or voice_name.lower() == voice["name"].lower():
                        debug(f"Found voice ID from API - {voice['voice_id']}")
                        return voice["voice_id"]
            else:
                err(f"Failed to get voices from ElevenLabs API. Status code: {response.status_code}")
                err(f"Response content: {response.text}")
        except Exception as e:
            err(f"Error determining voice ID: {str(e)}")
    
    warn(f"Voice '{voice_name}' not found; using the default specified in config/tts.yaml: {Tts.elevenlabs.default}")
    if Tts.elevenlabs.default in Tts.elevenlabs.voices:
        return Tts.elevenlabs.voices[Tts.elevenlabs.default]
    else:
        err(f"Default voice '{Tts.elevenlabs.default}' not found in configuration. Using first available voice.")
        return next(iter(Tts.elevenlabs.voices.values()))



async def elevenlabs_tts(model: str, input_text: str, voice: str, title: str = None, output_dir: str = None):
    # Debug logging
    debug(f"API.EXTENSIONS: {API.EXTENSIONS}")
    debug(f"API.EXTENSIONS.elevenlabs: {getattr(API.EXTENSIONS, 'elevenlabs', None)}")
    debug(f"Tts config: {Tts}")
    
    if getattr(API.EXTENSIONS, 'elevenlabs', False):
        voice_id = await determine_voice_id(voice)
    
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "text": input_text,
            "model_id": model
        }
        # Make sure this is the correct way to access the API key
        headers = {"Content-Type": "application/json", "xi-api-key": Tts.elevenlabs.key}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                response = await client.post(url, json=payload, headers=headers)
                output_dir = output_dir if output_dir else TTS_OUTPUT_DIR
                title = title if title else dt_datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{sanitize_filename(title)}.mp3"
                file_path = Path(output_dir) / filename
                if response.status_code == 200:            
                    with open(file_path, "wb") as audio_file:
                        audio_file.write(response.content)
                    return file_path
                else:
                    raise HTTPException(status_code=response.status_code, detail="Error from ElevenLabs API")
                
        except Exception as e:
            err(f"Error from Elevenlabs API: {e}")
            raise HTTPException(status_code=500, detail=f"Error from ElevenLabs API: {str(e)}")
    
    else:
        warn(f"elevenlabs_tts called but ElevenLabs module is not enabled in config.")
        raise HTTPException(status_code=400, detail="ElevenLabs TTS is not enabled")



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
        debug(f"No voice specified or file provided, using default voice: {Tts.xtts.default}")
        selected_voice = await select_voice(Tts.xtts.default)
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

    if API.EXTENSIONS.xtts:
        from TTS.api import TTS
        
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
        XTTS = await asyncio.to_thread(TTS, model_name=Tts.xtts.model)
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
            podcast_file_path = Path(Dir.PODCAST) / file_path.name
            await asyncio.to_thread(combined_audio.export, podcast_file_path, format="wav")
        
        await asyncio.to_thread(combined_audio.export, file_path, format="wav")
    
        return str(file_path)
        
    else:
        warn(f"local_tts called but xtts module disabled!")
        return None



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
    
    if API.EXTENSIONS.xtts:
        from TTS.api import TTS
        
        output_dir = tempfile.mktemp(suffix=".wav", dir=tempfile.gettempdir())
    
        XTTS = TTS(model_name=Tts.xtts.model).to(DEVICE)
        XTTS.tts_to_file(text=text, speed=speed, file_path=output_dir, speaker_wav=[voice_file_path], language="en")

        return output_dir
        
    else:
        warn(f"generate_tts called but xtts module disabled!")
        return None


async def get_audio_stream(model: str, input_text: str, voice: str):
    voice_id = await determine_voice_id(voice)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": input_text,
        "model_id": "eleven_turbo_v2"
    }
    headers = {"Content-Type": "application/json", "xi-api-key": Tts.elevenlabs.key}
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
        
        # Construct the destination path in the podcast folder
        destination_path = Path(Dir.PODCAST) / file_name
        
        # Copy the file to the podcast folder
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