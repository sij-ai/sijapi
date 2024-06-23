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
from datetime import datetime
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
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi import HOME_DIR, DATA_DIR, DEFAULT_VOICE, TTS_DIR, TTS_SEGMENTS_DIR, VOICE_DIR, PODCAST_DIR, TTS_OUTPUT_DIR, ELEVENLABS_API_KEY
from sijapi.utilities import sanitize_filename


### INITIALIZATIONS ###
tts = APIRouter(tags=["trusted", "private"])

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
            DEBUG(f"Response: {response}")
            if response.status_code == 200:
                voices_data = response.json().get("voices", [])
                formatted_list = ""
                for voice in voices_data:
                    name = voice["name"]
                    id = voice["voice_id"]
                    formatted_list += f"{name}: `{id}`\n"

        except Exception as e:
            ERR(f"Error determining voice ID: {str(e)}")

    return PlainTextResponse(formatted_list, status_code=200)   
     
        


def select_voice(voice_name: str) -> str:
    try:
        voice_file = VOICE_DIR / f"{voice_name}.wav"
        DEBUG(f"select_voice received query to use voice: {voice_name}. Looking for {voice_file} inside {VOICE_DIR}.")

        if voice_file.is_file():
            return str(voice_file)
        else:
            raise HTTPException(status_code=404, detail="Voice file not found")
    except Exception as e:
        ERR(f"Voice file not found: {str(e)}")
        ERR(traceback.format_exc())
        raise HTTPException(status_code=404, detail="Voice file not found")



@tts.post("/tts/speak")
@tts.post("/v1/audio/speech")
async def generate_speech_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
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
            return await generate_speech(background_tasks, text_content, voice, voice_file, model, speed, podcast)
    except Exception as e:
        ERR(f"Error in TTS: {str(e)}")
        ERR(traceback.format_exc())
        raise HTTPException(status_code=666, detail="error in TTS")


async def generate_speech(
    background_tasks: BackgroundTasks,
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

        if model == "eleven_turbo_v2":
            INFO(f"Using ElevenLabs.")
            audio_file_path = await elevenlabs_tts(model, text, voice, title, output_dir)
            return str(audio_file_path)
        
        elif model == "xtts":
            INFO(f"Using XTTS2")
            final_output_dir = await local_tts(text, speed, voice, voice_file, podcast, background_tasks, title, output_dir)
            background_tasks.add_task(os.remove, str(final_output_dir))
            return str(final_output_dir)
        else:
            raise HTTPException(status_code=400, detail="Invalid model specified")
    except HTTPException as e:
        ERR(f"HTTP error: {e}")
        ERR(traceback.format_exc())
        raise e
    except Exception as e:
        ERR(f"Error: {e}")
        ERR(traceback.format_exc())
        raise e



async def get_model(voice: str = None, voice_file: UploadFile = None):
    if voice_file or (voice and select_voice(voice)):
        return "xtts"
    elif voice and await determine_voice_id(voice):
        return "eleven_turbo_v2"
    else:
        raise HTTPException(status_code=400, detail="No model or voice specified")

async def determine_voice_id(voice_name: str) -> str:
    hardcoded_voices = {
        "alloy": "E3A1KVbKoWSIKSZwSUsW",
        "echo": "b42GBisbu9r5m5n6pHF7",
        "fable": "KAX2Y6tTs0oDWq7zZXW7",
        "onyx": "clQb8NxY08xZ6mX6wCPE",
        "nova": "6TayTBKLMOsghG7jYuMX",
        "shimmer": "E7soeOyjpmuZFurvoxZ2",
        DEFAULT_VOICE: "6TayTBKLMOsghG7jYuMX",
        "Sangye": "E7soeOyjpmuZFurvoxZ2",
        "Herzog": "KAX2Y6tTs0oDWq7zZXW7",
        "Attenborough": "b42GBisbu9r5m5n6pHF7"
    }

    if voice_name in hardcoded_voices:
        voice_id = hardcoded_voices[voice_name]
        DEBUG(f"Found voice ID - {voice_id}")
        return voice_id

    DEBUG(f"Requested voice not among the hardcoded options.. checking with 11L next.")
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            DEBUG(f"Response: {response}")
            if response.status_code == 200:
                voices_data = response.json().get("voices", [])
                for voice in voices_data:
                    if voice_name == voice["voice_id"] or voice_name == voice["name"]:
                        return voice["voice_id"]
        except Exception as e:
            ERR(f"Error determining voice ID: {str(e)}")

    return "6TayTBKLMOsghG7jYuMX" 


async def elevenlabs_tts(model: str, input_text: str, voice: str, title: str = None, output_dir: str = None):

    voice_id = await determine_voice_id(voice)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": input_text,
        "model_id": model
    }
    headers = {"Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        output_dir = output_dir if output_dir else TTS_OUTPUT_DIR
        title = title if title else datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{sanitize_filename(title)}.mp3"
        file_path = Path(output_dir) / filename
        if response.status_code == 200:            
            with open(file_path, "wb") as audio_file:
                audio_file.write(response.content)
            return file_path
        else:
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
        return select_voice(voice)
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
        DEBUG(f"{datetime.now().strftime('%Y%m%d%H%M%S')}: No voice specified or file provided, using default voice: {DEFAULT_VOICE}")
        return select_voice(DEFAULT_VOICE)


async def local_tts(text_content: str, speed: float, voice: str, voice_file = None, podcast: bool = False, background_tasks: BackgroundTasks = None, title: str = None, output_path: Optional[Path] = None) -> str:
    if output_path:
        file_path = Path(output_path)
    else:
        datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
        title = sanitize_filename(title) if title else "Audio"
        filename = f"{datetime_str}_{title}.wav"
        file_path = TTS_OUTPUT_DIR / filename

    # Ensure the parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    voice_file_path = await get_voice_file_path(voice, voice_file)
    XTTS = TTS(model_name=MODEL_NAME).to(DEVICE)
    segments = split_text(text_content)
    combined_audio = AudioSegment.silent(duration=0)

    for i, segment in enumerate(segments):
        segment_file_path = TTS_SEGMENTS_DIR / f"segment_{i}.wav"
        DEBUG(f"Segment file path: {segment_file_path}")
        segment_file = await asyncio.to_thread(XTTS.tts_to_file, text=segment, speed=speed, file_path=str(segment_file_path), speaker_wav=[voice_file_path], language="en")
        DEBUG(f"Segment file generated: {segment_file}")
        combined_audio += AudioSegment.from_wav(str(segment_file))
        # Delete the segment file immediately after adding it to the combined audio
        segment_file_path.unlink()

    if podcast:
        podcast_file_path = PODCAST_DIR / file_path.name
        combined_audio.export(podcast_file_path, format="wav")

    combined_audio.export(file_path, format="wav")
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
            DEBUG(f"split_text - segment: {' '.join(current_segment)}, word count: {segment_length}")

            current_segment = [sentence]
        else:
            current_segment.extend(sentence_words)

    if current_segment:
        segments.append(' '.join(current_segment))
        DEBUG(f"split_text - segment: {' '.join(current_segment)}, word count: {len(current_segment)}")

    return segments


def clean_text_for_tts(text: str) -> str:
    if text is not None:
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"[^\w\s.,;:!?'\"]", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        DEBUG(f"No text received.")



def copy_to_podcast_dir(file_path):
    try:
        # Extract the file name from the file path
        file_name = Path(file_path).name
        
        # Construct the destination path in the PODCAST_DIR
        destination_path = PODCAST_DIR / file_name
        
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