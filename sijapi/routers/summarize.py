from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import filetype
import shutil
import os
import re
from os.path import basename, splitext
from datetime import datetime
from typing import Optional, Union, List
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
import pytesseract
from pdf2image import convert_from_path
import asyncio
import html2text
import markdown
from ollama import Client, AsyncClient
from docx import Document

from sijapi.routers.tts import generate_speech
from sijapi.routers.asr import transcribe_audio
from sijapi.utilities import sanitize_filename, ocr_pdf, clean_text, should_use_ocr, extract_text_from_pdf, extract_text_from_docx, read_text_file, str_to_bool, get_extension, f
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi import DEFAULT_VOICE, SUMMARY_INSTRUCT, SUMMARY_CHUNK_SIZE, SUMMARY_TPW, SUMMARY_CHUNK_OVERLAP, SUMMARY_LENGTH_RATIO, SUMMARY_TOKEN_LIMIT, SUMMARY_MIN_LENGTH, SUMMARY_MIN_LENGTH, SUMMARY_MODEL

summarize = APIRouter(tags=["trusted", "private"])

@summarize.get("/summarize")
async def summarize_get(text: str = Form(None), instruction: str = Form(SUMMARY_INSTRUCT)):
    summarized_text = await summarize_text(text, instruction)
    return summarized_text

@summarize.post("/summarize")
async def summarize_post(file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), instruction: str = Form(SUMMARY_INSTRUCT)):
    text_content = text if text else await extract_text(file)
    summarized_text = await summarize_text(text_content, instruction)
    return summarized_text

@summarize.post("/speaksummary")
async def summarize_tts_endpoint(background_tasks: BackgroundTasks, instruction: str = Form(SUMMARY_INSTRUCT), file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), voice: Optional[str] = Form(DEFAULT_VOICE), speed: Optional[float] = Form(1.2), podcast: Union[bool, str] = Form(False)):
    
    podcast = str_to_bool(str(podcast))  # Proper boolean conversion
    text_content = text if text else extract_text(file)
    final_output_path = await summarize_tts(text_content, instruction, voice, speed, podcast)
    return FileResponse(path=final_output_path, filename=os.path.basename(final_output_path), media_type='audio/wav')
    

async def summarize_tts(
    text: str,
    instruction: str = SUMMARY_INSTRUCT,
    voice: Optional[str] = DEFAULT_VOICE,
    speed: float = 1.1,
    podcast: bool = False,
    LLM: AsyncClient = None
):
    LLM = LLM if LLM else AsyncClient()
    summarized_text = await summarize_text(text, instruction, LLM=LLM)
    filename = await summarize_text(summarized_text, "Provide a title for this summary no longer than 4 words")
    filename = sanitize_filename(filename)
    filename = ' '.join(filename.split()[:5])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}{filename}.wav" 
    
    background_tasks = BackgroundTasks()
    final_output_path = await generate_speech(background_tasks, summarized_text, voice, "xtts", speed=speed, podcast=podcast, title=filename)
    DEBUG(f"summary_tts completed with final_output_path: {final_output_path}")
    return final_output_path
    


async def get_title(text: str, LLM: AsyncClient() = None):
    LLM = LLM if LLM else AsyncClient()
    title = await process_chunk("Generate a title for this text", text, 1, 1, 12, LLM)
    title = sanitize_filename(title)
    return title

def split_text_into_chunks(text: str) -> List[str]:
    """
    Splits the given text into manageable chunks based on predefined size and overlap.
    """
    words = text.split()
    adjusted_chunk_size = max(1, int(SUMMARY_CHUNK_SIZE / SUMMARY_TPW))  # Ensure at least 1
    adjusted_overlap = max(0, int(SUMMARY_CHUNK_OVERLAP / SUMMARY_TPW))  # Ensure non-negative
    chunks = []
    for i in range(0, len(words), adjusted_chunk_size - adjusted_overlap):
        DEBUG(f"We are on iteration # {i} if split_text_into_chunks.")
        chunk = ' '.join(words[i:i + adjusted_chunk_size])
        chunks.append(chunk)
    return chunks


def calculate_max_tokens(text: str) -> int:
    tokens_count = max(1, int(len(text.split()) * SUMMARY_TPW))  # Ensure at least 1
    return min(tokens_count // 4, SUMMARY_CHUNK_SIZE)


async def extract_text(file: Union[UploadFile, bytes, bytearray, str, Path], background_tasks: BackgroundTasks = None) -> str:
    if isinstance(file, UploadFile):
        file_extension = get_extension(file)
        temp_file_path = tempfile.mktemp(suffix=file_extension)
        with open(temp_file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_path = temp_file_path
    elif isinstance(file, (bytes, bytearray)):
        temp_file_path = tempfile.mktemp()
        with open(temp_file_path, 'wb') as buffer:
            buffer.write(file)
        file_path = temp_file_path
    elif isinstance(file, (str, Path)):
        file_path = str(file)
    else:
        raise ValueError("Unsupported file type")

    _, file_ext = os.path.splitext(file_path)
    file_ext = file_ext.lower()
    text_content = ""

    if file_ext == '.pdf':
        text_content = await extract_text_from_pdf(file_path)
    elif file_ext in ['.wav', '.m4a', '.m4v', '.mp3', '.mp4']:
        text_content = await transcribe_audio(file_path=file_path)
    elif file_ext == '.md':
        text_content = await read_text_file(file_path)
        text_content = markdown.markdown(text_content)
    elif file_ext == '.html':
        text_content = await read_text_file(file_path)
        text_content = html2text.html2text(text_content)
    elif file_ext in ['.txt', '.csv', '.json']:
        text_content = await read_text_file(file_path)
    elif file_ext == '.docx':
        text_content = await extract_text_from_docx(file_path)

    if background_tasks and 'temp_file_path' in locals():
        background_tasks.add_task(os.remove, temp_file_path)
    elif 'temp_file_path' in locals():
        os.remove(temp_file_path)

    return text_content

async def summarize_text(text: str, instruction: str = SUMMARY_INSTRUCT, length_override: int = None, length_quotient: float = SUMMARY_LENGTH_RATIO, LLM: AsyncClient = None):
    """
    Process the given text: split into chunks, summarize each chunk, and
    potentially summarize the concatenated summary for long texts.
    """
    LLM = LLM if LLM else AsyncClient()

    chunked_text = split_text_into_chunks(text)
    total_parts = max(1, len(chunked_text))  # Ensure at least 1

    total_words_count = len(text.split())
    total_tokens_count = max(1, int(total_words_count * SUMMARY_TPW))  # Ensure at least 1
    total_summary_length = length_override if length_override else total_tokens_count // length_quotient
    corrected_total_summary_length = min(total_summary_length, SUMMARY_TOKEN_LIMIT)
    individual_summary_length = max(1, corrected_total_summary_length // total_parts)  # Ensure at least 1

    DEBUG(f"Text split into {total_parts} chunks.")
    summaries = await asyncio.gather(*[
        process_chunk(instruction, chunk, i+1, total_parts, individual_summary_length, LLM) for i, chunk in enumerate(chunked_text)
    ])
    
    concatenated_summary = ' '.join(summaries)
    
    if total_parts > 1:
        concatenated_summary = await process_chunk(instruction, concatenated_summary, 1, 1)
    
    return concatenated_summary

async def process_chunk(instruction: str, text: str, part: int, total_parts: int, max_tokens: Optional[int] = None, LLM: AsyncClient = None) -> str:
    """
    Process a portion of text using the ollama library asynchronously.
    """

    LLM = LLM if LLM else AsyncClient()

    words_count = max(1, len(text.split()))  # Ensure at least 1
    tokens_count = max(1, int(words_count * SUMMARY_TPW))  # Ensure at least 1
    fraction_tokens = max(1, tokens_count // SUMMARY_LENGTH_RATIO)  # Ensure at least 1
    if max_tokens is None:
        max_tokens = min(fraction_tokens, SUMMARY_CHUNK_SIZE // max(1, total_parts))  # Ensure at least 1
        max_tokens = max(max_tokens, SUMMARY_MIN_LENGTH)  # Ensure a minimum token count to avoid tiny processing chunks
    
    DEBUG(f"Summarizing part {part} of {total_parts}: Max_tokens: {max_tokens}")
    
    if part and total_parts > 1:
        prompt = f"{instruction}. Part {part} of {total_parts}:\n{text}"
    else:
        prompt = f"{instruction}:\n\n{text}"
    
    DEBUG(f"Starting LLM.generate for part {part} of {total_parts}")
    response = await LLM.generate(
        model=SUMMARY_MODEL, 
        prompt=prompt,
        stream=False,
        options={'num_predict': max_tokens, 'temperature': 0.6}
    )
    
    text_response = response['response']
    DEBUG(f"Completed LLM.generate for part {part} of {total_parts}")
    
    return text_response

async def title_and_summary(extracted_text: str):
    title = await get_title(extracted_text)
    processed_title = title.split("\n")[-1]
    processed_title = processed_title.split("\r")[-1]
    processed_title = sanitize_filename(processed_title)
    summary = await summarize_text(extracted_text)

    return processed_title, summary