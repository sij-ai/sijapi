'''
Automatic Speech Recognition module relying on the `whisper_cpp` implementation of OpenAI's Whisper model.
Depends on:
  LOGGER, ASR_DIR, WHISPER_CPP_MODELS, GARBAGE_COLLECTION_INTERVAL, GARBAGE_TTL, WHISPER_CPP_DIR
Notes: 
  Performs exceptionally well on Apple Silicon. Other devices will benefit from future updates to optionally use `faster_whisper`, `insanely_faster_whisper`, and/or `whisper_jax`.
'''

from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional
import tempfile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, HttpUrl
from whisperplus.pipelines import mlx_whisper
from youtube_dl import YoutubeDL
from urllib.parse import unquote
import subprocess
import os
import uuid
from threading import Thread
import multiprocessing
import asyncio
import subprocess
import tempfile

from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL, ASR_DIR, WHISPER_CPP_MODELS, GARBAGE_COLLECTION_INTERVAL, GARBAGE_TTL, WHISPER_CPP_DIR, MAX_CPU_CORES


asr = APIRouter()

class TranscribeParams(BaseModel):
    model: str = Field(default="small")
    output_srt : Optional[bool] = Field(default=False)
    language : Optional[str] = Field(None)
    split_on_word : Optional[bool] = Field(default=False)
    temperature : Optional[float] = Field(default=0)
    temp_increment : Optional[int] = Field(None)
    translate : Optional[bool] = Field(default=False)
    diarize : Optional[bool] = Field(default=False)
    tiny_diarize : Optional[bool] = Field(default=False)
    no_fallback : Optional[bool] = Field(default=False)
    output_json : Optional[bool] = Field(default=False)
    detect_language : Optional[bool] = Field(default=False)
    dtw : Optional[str] = Field(None)
    threads : Optional[int] = Field(None)

from urllib.parse import unquote
import json

@asr.post("/asr")
@asr.post("/transcribe")
@asr.post("/v1/audio/transcription")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    params: str = Form(...)
):
    try:
        # Decode the URL-encoded string
        decoded_params = unquote(params)
        
        # Parse the JSON string
        parameters_dict = json.loads(decoded_params)
        
        # Create TranscribeParams object
        parameters = TranscribeParams(**parameters_dict)
    except json.JSONDecodeError as json_err:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(json_err)}")
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Error parsing parameters: {str(err)}")
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    transcription = await transcribe_audio(file_path=temp_file_path, params=parameters)
    return transcription

async def transcribe_audio(file_path, params: TranscribeParams):

    file_path = convert_to_wav(file_path)
    model = params.model if params.model in WHISPER_CPP_MODELS else 'small' 
    model_path = WHISPER_CPP_DIR / 'models' / f'ggml-{model}.bin'
    command = [str(WHISPER_CPP_DIR / 'build' / 'bin' / 'main')]
    command.extend(['-m', str(model_path)]) 
    command.extend(['-t', str(max(1, min(params.threads or MAX_CPU_CORES, MAX_CPU_CORES)))])
    command.extend(['-np'])  # Always enable no-prints

    if params.split_on_word:
        command.append('-sow')
    if params.temperature > 0:
        command.extend(['-tp', str(params.temperature)])
    if params.temp_increment:
        command.extend(['-tpi', str(params.temp_increment)])
    if params.language:
        command.extend(['-l', params.language])
    elif params.detect_language:
        command.append('-dl')
    if params.translate:
        command.append('-tr')
    if params.diarize:
        command.append('-di')
    if params.tiny_diarize:
        command.append('-tdrz')
    if params.no_fallback:
        command.append('-nf')
    if params.output_srt:
        command.append('-osrt')
    elif params.output_json:
        command.append('-oj')
    else:
        command.append('-nt')
    if params.dtw:
        command.extend(['--dtw', params.dtw])

    command.extend(['-f', file_path])
  
    DEBUG(f"Command: {command}")
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise Exception(f"Error running command: {stderr.decode()}")
    
    result = stdout.decode().strip()
    DEBUG(f"Result: {result}")
    return result


def convert_to_wav(file_path: str):
    wav_file_path = os.path.join(ASR_DIR, f"{uuid.uuid4()}.wav")
    subprocess.run(["ffmpeg", "-y", "-i", file_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_file_path], check=True)
    return wav_file_path
def download_from_youtube(url: str):
    temp_file = os.path.join(ASR_DIR, f"{uuid.uuid4()}.mp3")
    ytdl_opts = {
        'outtmpl': temp_file,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'nooverwrites': True
    }
    with YoutubeDL(ytdl_opts) as ydl:
        ydl.download([url])
    return convert_to_wav(temp_file)

def format_srt_timestamp(seconds: float):
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def write_srt(segments: list, output_file: str):
    with open(output_file, 'w') as f:
        for i, segment in enumerate(segments, start=1):
            start = format_srt_timestamp(segment['start'])
            end = format_srt_timestamp(segment['end'])
            text = segment['text']
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
