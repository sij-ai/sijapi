import os
import sys
import uuid
import json
import asyncio
import tempfile
import subprocess
from urllib.parse import unquote
from fastapi import APIRouter, HTTPException, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL, ASR_DIR, WHISPER_CPP_MODELS, GARBAGE_COLLECTION_INTERVAL, GARBAGE_TTL, WHISPER_CPP_DIR, MAX_CPU_CORES

asr = APIRouter()

class TranscribeParams(BaseModel):
    model: str = Field(default="small")
    output_srt: Optional[bool] = Field(default=False)
    language: Optional[str] = Field(None)
    split_on_word: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=0)
    temp_increment: Optional[int] = Field(None)
    translate: Optional[bool] = Field(default=False)
    diarize: Optional[bool] = Field(default=False)
    tiny_diarize: Optional[bool] = Field(default=False)
    no_fallback: Optional[bool] = Field(default=False)
    output_json: Optional[bool] = Field(default=False)
    detect_language: Optional[bool] = Field(default=False)
    dtw: Optional[str] = Field(None)
    threads: Optional[int] = Field(None)

# Global dictionary to store transcription results
transcription_results = {}

@asr.post("/asr")
@asr.post("/transcribe")
@asr.post("/v1/audio/transcription")
async def transcribe_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    params: str = Form(...)
):
    try:
        decoded_params = unquote(params)
        parameters_dict = json.loads(decoded_params)
        parameters = TranscribeParams(**parameters_dict)
    except json.JSONDecodeError as json_err:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(json_err)}")
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Error parsing parameters: {str(err)}")
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    transcription_job = await transcribe_audio(file_path=temp_file_path, params=parameters, background_tasks=background_tasks)
    job_id = transcription_job["job_id"]

    # Poll for completion
    max_wait_time = 600  # 10 minutes
    poll_interval = 2  # 2 seconds
    elapsed_time = 0

    while elapsed_time < max_wait_time:
        if job_id in transcription_results:
            result = transcription_results[job_id]
            if result["status"] == "completed":
                return JSONResponse(content={"status": "completed", "result": result["result"]})
            elif result["status"] == "failed":
                return JSONResponse(content={"status": "failed", "error": result["error"]}, status_code=500)
        
        await asyncio.sleep(poll_interval)
        elapsed_time += poll_interval

    # If we've reached this point, the transcription has taken too long
    return JSONResponse(content={"status": "timeout", "message": "Transcription is taking longer than expected. Please check back later."}, status_code=202)

async def transcribe_audio(file_path, params: TranscribeParams, background_tasks: BackgroundTasks):
    file_path = await convert_to_wav(file_path)
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

    # Create a unique ID for this transcription job
    job_id = str(uuid.uuid4())

    # Store the job status
    transcription_results[job_id] = {"status": "processing", "result": None}

    # Run the transcription in a background task
    background_tasks.add_task(process_transcription, command, file_path, job_id)

    return {"job_id": job_id}

async def process_transcription(command, file_path, job_id):
    try:
        result = await run_transcription(command, file_path)
        transcription_results[job_id] = {"status": "completed", "result": result}
    except Exception as e:
        transcription_results[job_id] = {"status": "failed", "error": str(e)}
    finally:
        # Clean up the temporary file
        os.remove(file_path)

async def run_transcription(command, file_path):
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"Error running command: {stderr.decode()}")
    return stdout.decode().strip()

async def convert_to_wav(file_path: str):
    wav_file_path = os.path.join(ASR_DIR, f"{uuid.uuid4()}.wav")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", file_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"Error converting file to WAV: {stderr.decode()}")
    return wav_file_path

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
