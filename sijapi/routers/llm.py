#routers/llm.py
from fastapi import APIRouter, HTTPException, Request, Response, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from datetime import datetime as dt_datetime
from dateutil import parser
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, root_validator, ValidationError
import aiofiles
import os 
import glob
import chromadb
from openai import OpenAI
import uuid
import json
import base64
from pathlib import Path
import ollama
from ollama import AsyncClient as Ollama, list as OllamaList
import time
import asyncio
import tempfile
import shutil
import html2text
import markdown
from sijapi import LLM_SYS_MSG, DEFAULT_LLM, DEFAULT_VISION, REQUESTS_DIR, OBSIDIAN_CHROMADB_COLLECTION, OBSIDIAN_VAULT_DIR, DOC_DIR, OPENAI_API_KEY, DEBUG, INFO, WARN, ERR, CRITICAL, DEFAULT_VOICE, SUMMARY_INSTRUCT, SUMMARY_CHUNK_SIZE, SUMMARY_TPW, SUMMARY_CHUNK_OVERLAP, SUMMARY_LENGTH_RATIO, SUMMARY_TOKEN_LIMIT, SUMMARY_MIN_LENGTH, SUMMARY_MODEL
from sijapi.utilities import convert_to_unix_time, sanitize_filename, ocr_pdf, clean_text, should_use_ocr, extract_text_from_pdf, extract_text_from_docx, read_text_file, str_to_bool, get_extension
from sijapi.routers.tts import generate_speech
from sijapi.routers.asr import transcribe_audio


llm = APIRouter()

# Initialize chromadb client
client = chromadb.Client()
OBSIDIAN_CHROMADB_COLLECTION = client.create_collection("obsidian")

# Function to read all markdown files in the folder
def read_markdown_files(folder: Path):
    file_paths = glob.glob(os.path.join(folder, "*.md"))
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents, file_paths

# Read markdown files and generate embeddings
documents, file_paths = read_markdown_files(DOC_DIR)
for i, doc in enumerate(documents):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=doc)
    embedding = response["embedding"]
    OBSIDIAN_CHROMADB_COLLECTION.add(
        ids=[file_paths[i]],
        embeddings=[embedding],
        documents=[doc]
    )

# Function to retrieve the most relevant document given a prompt
@llm.get("/retrieve_document/{prompt}")
async def retrieve_document(prompt: str):
    response = ollama.embeddings(
        prompt=prompt,
        model="mxbai-embed-large"
    )
    results = OBSIDIAN_CHROMADB_COLLECTION.query(
        query_embeddings=[response["embedding"]],
        n_results=1
    )
    return {"document": results['documents'][0][0]}

# Function to generate a response using RAG
@llm.get("/generate_response/{prompt}")
async def generate_response(prompt: str):
    data = await retrieve_document(prompt)
    output = ollama.generate(
        model="llama2",
        prompt=f"Using this data: {data['document']}. Respond to this prompt: {prompt}"
    )
    return {"response": output['response']}


async def query_ollama(usr: str, sys: str = LLM_SYS_MSG, model: str = DEFAULT_LLM, max_tokens: int = 200):
    messages = [{"role": "system", "content": sys},
                {"role": "user", "content": usr}]
    LLM = Ollama()
    response = await LLM.chat(model=model, messages=messages, options={"num_predict": max_tokens})

    DEBUG(response)
    if "message" in response:
        if "content" in response["message"]:
            content = response["message"]["content"]
            return content
    else:
        DEBUG("No choices found in response")
        return None

def is_vision_request(content):
    return False

@llm.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    timestamp = dt_datetime.now().strftime("%Y%m%d_%H%M%S%f")
    filename = REQUESTS_DIR / f"request_{timestamp}.json"

    async with aiofiles.open(filename, mode='w') as file:
        await file.write(json.dumps(body, indent=4))

    messages = body.get('messages')
    if not messages:
        raise HTTPException(status_code=400, detail="Message data is required in the request body.")

    requested_model = body.get('model', 'default-model')
    DEBUG(f"Requested model: {requested_model}")
    stream = body.get('stream')
    token_limit = body.get('max_tokens') or body.get('num_predict')

    # Check if the most recent message contains an image_url
    recent_message = messages[-1]
    if recent_message.get('role') == 'user' and is_vision_request(recent_message.get('content')):
        DEBUG("Processing as a vision request")
        model = "llava"
        DEBUG(f"Using model: {model}")
        return StreamingResponse(stream_messages_with_vision(recent_message, model, token_limit), media_type="application/json")
    else:
        DEBUG("Processing as a standard request")
        model = requested_model
        DEBUG(f"Using model: {model}")
        if stream:
            return StreamingResponse(stream_messages(messages, model, token_limit), media_type="application/json")
        else:
            response_data = await generate_messages(messages, model)
            return JSONResponse(response_data, media_type="application/json")

async def stream_messages(messages: list, model: str = "llama3", num_predict: int = 300):
    async with Ollama() as async_client:
        try:
            index = 0
            async for part in async_client.chat(model=model, messages=messages, stream=True, options={'num_predict': num_predict}):
                yield "data: " + json.dumps({
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "system_fingerprint": "fp_44709d6fcb",
                    "choices": [{
                        "index": index,
                        "delta": {"role": "assistant", "content": part['message']['content']},
                        "logprobs": None,
                        "finish_reason": None if 'finish_reason' not in part else part['finish_reason']
                    }]
                }) + "\n\n"
                index += 1
        except Exception as e:
            yield "data: " + json.dumps({"error": f"Error: {str(e)}"}) + "\n\n"
        yield "data: [DONE]\n\n"



async def stream_messages_with_vision(message: dict, model: str, num_predict: int = 300):
    async with Ollama() as async_client:
        try:
            if isinstance(message.get('content'), list):
                content = message['content']
                for part in content:
                    if part['type'] == 'image_url' and 'url' in part['image_url']:
                        image_url = part['image_url']['url']
                        if image_url.startswith('data:image'):
                            # Convert base64 to bytes
                            image_data = base64.b64decode(image_url.split('base64,')[1])
                            response_generator = await async_client.generate(
                                model=model, 
                                prompt='explain this image:', 
                                images=[image_data], 
                                stream=True,
                                options={'num_predict': num_predict}
                            )
                            index = 0
                            async for response in response_generator:
                                yield "data: " + json.dumps({
                                    "id": "chatcmpl-123",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "system_fingerprint": "fp_44709d6fcb",
                                    "choices": [{
                                        "index": index,
                                        "delta": {"role": "assistant", "content": response['response']},
                                        "logprobs": None,
                                        "finish_reason": None if 'finish_reason' not in response else response['finish_reason']
                                    }]
                                }) + "\n\n"
                                index += 1
        except Exception as e:
            yield "data: " + json.dumps({"error": f"Error: {str(e)}"}) + "\n\n"
        yield "data: [DONE]\n\n"

        
def get_appropriate_model(requested_model):
    if requested_model == "gpt-4-vision-preview":
        return DEFAULT_VISION
    elif not is_model_available(requested_model):
        return DEFAULT_LLM
    else:
        return requested_model

def is_vision_request(content):
    if isinstance(content, list):
        return any(isinstance(msg, dict) and msg.get('type') == 'image_url' for msg in content)
    return False


@llm.get("/v1/models")
async def get_models():
    model_data = OllamaList() 
    formatted_models = []

    for model in model_data['models']:
        model_id = model['name'].split(':')[0]  
        formatted_models.append({
            "id": model_id,
            "object": "model",
            "created": convert_to_unix_time(model['modified_at']),
            "owned_by": "sij"
        })

    return JSONResponse({
        "object": "list",
        "data": formatted_models
    })

async def generate_messages(messages: list, model: str = "llama3"):
    async_client = Ollama()
    try:
        response = await async_client.chat(model=model, messages=messages, stream=False)
        return {
            "model": model,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response['message']['content']
                }
            }]
        }
    except Exception as e:
        return {"error": f"Error: {str(e)}"}



def is_model_available(model_name):
    model_data = OllamaList()
    available_models = [model['name'] for model in model_data['models']]
    DEBUG(f"Available models: {available_models}")  # Log using the configured LOGGER

    matching_models = [model for model in available_models if model.startswith(model_name + ':') or model == model_name]
    if len(matching_models) == 1:
        DEBUG(f"Unique match found: {matching_models[0]}")
        return True
    elif len(matching_models) > 1:
        ERR(f"Ambiguous match found, models: {matching_models}")
        return True
    else:
        ERR(f"No match found for model: {model_name}")
    return False


@llm.options("/chat/completions")
@llm.options("/v1/chat/completions")
async def chat_completions_options(request: Request):
    return JSONResponse(
        content={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "To use the chat completions endpoint, make a POST request to /v1/chat/completions with a JSON payload containing the 'messages' array. Each message should have a 'role' (either 'system', 'user', or 'assistant') and 'content' (the message text). You can optionally specify the 'model' to use. The response will be a JSON object containing the generated completions."
                    },
                    "finish_reason": "stop"
                }
            ],
            "created": int(time.time()),
            "id": str(uuid.uuid4()),
            "model": DEFAULT_LLM,
            "object": "chat.completion.chunk",
        },
        status_code=200,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Allow": "OPTIONS, POST",
        },
    )

#### EMBEDDINGS

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str], None] = None
    prompt: Union[str, List[str], None] = None

    @root_validator(pre=True)
    def ensure_list(cls, values):
        input_value = values.get('input')
        prompt_value = values.get('prompt')

        if input_value and isinstance(input_value, str):
            values['input'] = [input_value]

        if prompt_value and isinstance(prompt_value, str):
            values['prompt'] = [prompt_value]

        if input_value and not prompt_value:
            values['prompt'] = values['input']
            values['input'] = None

        return values

class EmbeddingResponse(BaseModel):
    object: str
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

@llm.post("/api/embeddings", response_model=EmbeddingResponse)
@llm.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    try:
        combined_input = " ".join(request.prompt)
        response = ollama.embeddings(model=request.model, prompt=combined_input)
        embedding_list = response.get("embedding", [])

        data = [{
            "object": "embedding",
            "index": 0,
            "embedding": embedding_list
        }]

        result = {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {"prompt_tokens": 5, "total_tokens": 5}  # Example token counts
        }

        return result
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@llm.options("/api/embeddings")
@llm.options("/v1/embeddings")
async def options_embedding():
    return JSONResponse(
        content={},
        headers={
            "Allow": "OPTIONS, POST",
            "Content-Type": "application/json",
            "Access-Control-Allow-Methods": "OPTIONS, POST",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )




###### PORTED FROM IGBOT, NEEDS TO BE UPDATED FOR THIS ENVIRONMENT AND MADE ASYNC: #####

def query_gpt4(llmPrompt: List = [], system_msg: str = "", user_msg: str = "", max_tokens: int = 150):
    messages = llmPrompt if llmPrompt else [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
    LLM = OpenAI(api_key=OPENAI_API_KEY) 
    response = LLM.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=max_tokens
    )
    if hasattr(response, "choices") and response.choices:  # Checks if 'choices' attribute exists and is not empty
        first_choice = response.choices[0]
        if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
            return first_choice.message.content
        else:
            DEBUG("No content attribute in the first choice's message")
            DEBUG(f"No content found in message string: {response.choices}")
            DEBUG("Trying again!")
            query_gpt4(messages, max_tokens)
    else:
        DEBUG(f"No content found in message string: {response}")
        return ""

def llava(image_base64, prompt):
    VISION_LLM = Ollama(host='http://localhost:11434')
    response = VISION_LLM.generate(
        model = 'llava',
        prompt = f"This is a chat between a user and an assistant. The assistant is helping the user to describe an image. {prompt}",
        images = [image_base64]
    )
    DEBUG(response)
    return "" if "pass" in response["response"].lower() else response["response"] 

def gpt4v(image_base64, prompt_sys: str, prompt_usr: str, max_tokens: int = 150):
    VISION_LLM = OpenAI(api_key=OPENAI_API_KEY)
    response_1 = VISION_LLM.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": f"This is a chat between a user and an assistant. The assistant is helping the user to describe an image. {prompt_sys}",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt_usr}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image_base64}"}}
                ],
            }
        ],
        max_tokens=max_tokens,
        stream=False
    )

    if response_1 and response_1.choices:
        if len(response_1.choices) > 0:
            first_choice = response_1.choices[0]
            if first_choice.message and first_choice.message.content:
                comment_content = first_choice.message.content
                if "PASS" in comment_content:
                    return ""
                DEBUG(f"Generated comment: {comment_content}")

                response_2 = VISION_LLM.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": f"This is a chat between a user and an assistant. The assistant is helping the user to describe an image. {prompt_sys}",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{prompt_usr}"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpg;base64,{image_base64}"
                                },
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": comment_content
                        },
                        {
                            "role": "user",
                            "content": "Please refine it, and remember to ONLY include the caption or comment, nothing else! That means no preface, no postscript, no notes, no reflections, and not even any acknowledgment of this follow-up message. I need to be able to use your output directly on social media. Do include emojis though."
                        }
                    ],
                    max_tokens=max_tokens,
                    stream=False
                )
                if response_2 and response_2.choices:
                    if len(response_2.choices) > 0:
                        first_choice = response_2.choices[0]
                        if first_choice.message and first_choice.message.content:
                            final_content = first_choice.message.content
                            DEBUG(f"Generated comment: {final_content}")
                            if "PASS" in final_content:
                                return ""
                            else:
                                return final_content


    DEBUG("Vision response did not contain expected data.")
    DEBUG(f"Vision response: {response_1}")
    asyncio.sleep(15)

    try_again = gpt4v(image_base64, prompt_sys, prompt_usr, max_tokens)
    return try_again


@llm.get("/summarize")
async def summarize_get(text: str = Form(None), instruction: str = Form(SUMMARY_INSTRUCT)):
    summarized_text = await summarize_text(text, instruction)
    return summarized_text

@llm.post("/summarize")
async def summarize_post(file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), instruction: str = Form(SUMMARY_INSTRUCT)):
    text_content = text if text else await extract_text(file)
    summarized_text = await summarize_text(text_content, instruction)
    return summarized_text

@llm.post("/speaksummary")
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
    LLM: Ollama = None
):
    LLM = LLM if LLM else Ollama()
    summarized_text = await summarize_text(text, instruction, LLM=LLM)
    filename = await summarize_text(summarized_text, "Provide a title for this summary no longer than 4 words")
    filename = sanitize_filename(filename)
    filename = ' '.join(filename.split()[:5])
    timestamp = dt_datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}{filename}.wav" 
    
    background_tasks = BackgroundTasks()
    final_output_path = await generate_speech(background_tasks, summarized_text, voice, "xtts", speed=speed, podcast=podcast, title=filename)
    DEBUG(f"summary_tts completed with final_output_path: {final_output_path}")
    return final_output_path
    

async def get_title(text: str, LLM: Ollama() = None):
    LLM = LLM if LLM else Ollama()
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

async def summarize_text(text: str, instruction: str = SUMMARY_INSTRUCT, length_override: int = None, length_quotient: float = SUMMARY_LENGTH_RATIO, LLM: Ollama = None):
    """
    Process the given text: split into chunks, summarize each chunk, and
    potentially summarize the concatenated summary for long texts.
    """
    LLM = LLM if LLM else Ollama()

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

async def process_chunk(instruction: str, text: str, part: int, total_parts: int, max_tokens: Optional[int] = None, LLM: Ollama = None) -> str:
    """
    Process a portion of text using the ollama library asynchronously.
    """

    LLM = LLM if LLM else Ollama()

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