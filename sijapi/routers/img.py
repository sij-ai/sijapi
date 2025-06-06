'''
Image generation module using StableDiffusion and similar models by way of ComfyUI.
'''
#routers/img.py

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse, RedirectResponse
from aiohttp import ClientSession
import aiofiles
from PIL import Image
from pathlib import Path
import uuid
import json
import yaml
import socket
import subprocess
import random
import os
import asyncio
from sijapi.routers.llm import query_ollama
from sijapi import Sys, COMFYUI_URL, COMFYUI_OUTPUT_DIR, IMG_CONFIG_PATH, IMG_DIR, IMG_WORKFLOWS_DIR
from sijapi.logs import get_logger
l = get_logger(__name__)

img = APIRouter()


CLIENT_ID = str(uuid.uuid4())

@img.post("/img")
@img.post("/v1/images/generations")
async def sd_endpoint(request: Request):
    request_data = await request.json()
    prompt = request_data.get("prompt")
    model = request_data.get("model")
    size = request_data.get("size")
    earlyurl = request_data.get("earlyurl", None)
    earlyout = "web" if earlyurl else None

    image_path = await workflow(prompt=prompt, scene=model, size=size, earlyout=earlyout)

    if earlyout == "web":
        return JSONResponse({"image_url": image_path})
        # return RedirectResponse(url=image_path, status_code=303)
    else:
        return JSONResponse({"image_url": image_path})
    
@img.get("/img")
@img.get("/v1/images/generations")
async def sd_endpoint(
    request: Request,
    prompt: str = Query(..., description="The prompt for image generation"),
    earlyout: str = Query("output", description="specify web for a redirect, or json for a json with the local path")
):
    image_path = await workflow(prompt=prompt, scene="wallpaper", earlyout=earlyout)
    web_path = get_web_path(image_path)

    if earlyout == "web":
        return RedirectResponse(url=web_path, status_code=303)
    else:
        return JSONResponse({"image_url": image_path})

async def workflow(prompt: str, scene: str = None, size: str = None, earlyout: str = None, destination_path: str = None, downscale_to_fit: bool = False):
    scene_data = get_scene(scene)
    if not scene_data:
        scene_data = get_matching_scene(prompt)
    prompt = scene_data.get('llm_pre_prompt') + prompt
    prompt_model = scene_data.get('prompt_model')
    image_concept = await query_ollama(usr=prompt, sys=scene_data.get('llm_sys_msg'), model=prompt_model, max_tokens=100)

    scene_workflow = random.choice(scene_data['workflows'])
    if size:
        l.debug(f"Specified size: {size}")

    size = size if size else scene_workflow.get('size', '1024x1024')
    
    width, height = map(int, size.split('x'))
    l.debug(f"Parsed width: {width}; parsed height: {height}")

    workflow_path = Path(IMG_WORKFLOWS_DIR) / scene_workflow['workflow']
    workflow_data = json.loads(workflow_path.read_text())

    post = {
        "API_PrePrompt": scene_data['API_PrePrompt'] + image_concept + ', '.join(f"; (({trigger}))" for trigger in scene_data['triggers']),
        "API_StylePrompt": scene_data['API_StylePrompt'],
        "API_NegativePrompt": scene_data['API_NegativePrompt'],
        "width": width,
        "height": height
    }

    saved_file_key = await update_prompt_and_get_key(workflow=workflow_data, post=post, positive=image_concept)
    l.info(f"Saved file key: {saved_file_key}")

    prompt_id = await queue_prompt(workflow_data)
    l.info(f"Prompt ID: {prompt_id}")

    max_size = max(width, height) if downscale_to_fit else None
    destination_path = Path(destination_path).with_suffix(".jpg") if destination_path else IMG_DIR / f"{prompt_id}.jpg"

    if earlyout:
        asyncio.create_task(generate_and_save_image(prompt_id, saved_file_key, max_size, destination_path))
        l.debug(f"Returning {destination_path}")
        return destination_path
    
    else:
        await generate_and_save_image(prompt_id, saved_file_key, max_size, destination_path)
        l.debug(f"Returning {destination_path}")
        return destination_path


async def generate_and_save_image(prompt_id, saved_file_key, max_size, destination_path):
    try:
        status_data = await poll_status(prompt_id)
        image_data = await get_image(status_data, saved_file_key)
        jpg_file_path = await save_as_jpg(image_data, prompt_id, quality=90, max_size=max_size, destination_path=destination_path)

        if Path(jpg_file_path) != Path(destination_path):
            l.error(f"Mismatch between jpg_file_path, {jpg_file_path}, and detination_path, {destination_path}")

    except Exception as e:
        l.error(f"Error in generate_and_save_image: {e}")
        return None
    
    
def get_web_path(file_path: Path) -> str:
    uri = file_path.relative_to(IMG_DIR)
    web_path = f"{Sys.URL}/img/{uri}"
    return web_path


async def poll_status(prompt_id):
    """Asynchronously poll the job status until it's complete and return the status data."""
    start_time = asyncio.get_event_loop().time()
    async with ClientSession() as session:
        while True:
            elapsed_time = int(asyncio.get_event_loop().time() - start_time)
            async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as response:
                if response.status != 200:
                    raise Exception("Failed to get job status")
                status_data = await response.json()
                job_data = status_data.get(prompt_id, {})
                if job_data.get("status", {}).get("completed", False):
                    l.info(f"{prompt_id} completed in {elapsed_time} seconds.")
                    return job_data
            await asyncio.sleep(1)


async def get_image(status_data, key):
    """Asynchronously extract the filename and subfolder from the status data and read the file."""
    try:
        outputs = status_data.get("outputs", {})
        images_info = outputs.get(key, {}).get("images", [])
        if not images_info:
            raise Exception("No images found in the job output.")

        image_info = images_info[0]
        filename = image_info.get("filename")
        subfolder = image_info.get("subfolder", "")
        file_path = os.path.join(COMFYUI_OUTPUT_DIR, subfolder, filename)

        async with aiofiles.open(file_path, 'rb') as file:
            return await file.read()
    except Exception as e:
        raise Exception(f"Failed to get image: {e}")


async def save_as_jpg(image_data, prompt_id, max_size = None, quality = 100, destination_path: Path = None):
    destination_path_png = (IMG_DIR / prompt_id).with_suffix(".png")
    destination_path_jpg = destination_path.with_suffix(".jpg") if destination_path else (IMG_DIR / prompt_id).with_suffix(".jpg")

    try:
        destination_path_png.parent.mkdir(parents=True, exist_ok=True)
        destination_path_jpg.parent.mkdir(parents=True, exist_ok=True)

        # Save the PNG
        async with aiofiles.open(destination_path_png, 'wb') as f:
            await f.write(image_data)

        # Open, possibly resize, and save as JPG
        with Image.open(destination_path_png) as img:
            if max_size and max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple([int(x * ratio) for x in img.size])
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            img.convert('RGB').save(destination_path_jpg, format='JPEG', quality=quality)

        # Optionally remove the PNG
        os.remove(destination_path_png)

        return str(destination_path_jpg)
    
    except Exception as e:
        l.error(f"Error processing image: {e}")
        return None



def set_presets(workflow_data, preset_values):
    if preset_values:
        preset_node = preset_values.get('node')
        preset_key = preset_values.get('key')
        values = preset_values.get('values')

        if preset_node and preset_key and values:
            preset_value = random.choice(values)
            if 'inputs' in workflow_data.get(preset_node, {}):
                workflow_data[preset_node]['inputs'][preset_key] = preset_value
            else:
                l.debug("Node not found in workflow_data")
        else:
            l.debug("Required data missing in preset_values")
    else:
        l.debug("No preset_values found")


def get_return_path(destination_path):
    sd_dir = Path(IMG_DIR)
    if destination_path.parent.samefile(sd_dir):
        return destination_path.name
    else:
        return str(destination_path)

def get_scene(scene):
    with open(IMG_CONFIG_PATH, 'r') as IMG_CONFIG_file:
        IMG_CONFIG = yaml.safe_load(IMG_CONFIG_file)
    for scene_data in IMG_CONFIG['scenes']:
        if scene_data['scene'] == scene:
            l.debug(f"Found scene for \"{scene}\".")
            return scene_data
    return None


# This returns the scene with the most trigger words present in the provided prompt,
# or otherwise if none match it returns the first scene in the array - 
# meaning the first should be considered the default scene.
def get_matching_scene(prompt):
    prompt_lower = prompt.lower()
    max_count = 0
    scene_data = None
    with open(IMG_CONFIG_PATH, 'r') as IMG_CONFIG_file:
        IMG_CONFIG = yaml.safe_load(IMG_CONFIG_file)
    for sc in IMG_CONFIG['scenes']:
        count = sum(1 for trigger in sc['triggers'] if trigger in prompt_lower)
        if count > max_count:
            max_count = count
            scene_data = sc
            if scene_data:
                l.debug(f"Found better-matching scene: the prompt contains {max_count} words that match triggers for {scene_data.get('name')}!")
    if scene_data:
        return scene_data
    else:
        l.debug(f"No matching scenes found, falling back to default scene.")
        return IMG_CONFIG['scenes'][0]



async def ensure_comfy(retries: int = 4, timeout: float = 6.0):
    """
    Ensures that ComfyUI is running, starting it if necessary.

    Args:
        retries (int): Number of connection attempts. Defaults to 3.
        timeout (float): Time to wait between attempts in seconds. Defaults to 5.0.

    Raises:
        RuntimeError: If ComfyUI couldn't be started or connected to after all retries.
    """
    for attempt in range(retries):
        try:
            with socket.create_connection(("127.0.0.1", 8188), timeout=2):
                l.info("ComfyUI is already running.")
                return
        except (socket.timeout, ConnectionRefusedError):
            if attempt == 0:  # Only try to start ComfyUI on the first failed attempt
                l.warning("ComfyUI is not running. Starting it now...")
                try:
                    tmux_command = (
                        "tmux split-window -h "
                        "\"source /Users/sij/.zshrc; cd /Users/sij/workshop/ComfyUI; "
                        "mamba activate comfyui && "
                        "python main.py; exec $SHELL\""
                    )
                    subprocess.Popen(tmux_command, shell=True)
                    l.info("ComfyUI started in a new tmux session.")
                except Exception as e:
                    raise RuntimeError(f"Error starting ComfyUI: {e}")
            
            l.warning(f"Attempt {attempt + 1}/{retries} failed. Waiting {timeout} seconds before retrying...")
            await asyncio.sleep(timeout)

    l.critical(f"Failed to ensure ComfyUI is running after {retries} attempts with {timeout} second intervals.")
    raise RuntimeError(f"Failed to ensure ComfyUI is running after {retries} attempts with {timeout} second intervals.")


@img.get("/image/{prompt_id}")
async def get_image_status(prompt_id: str):
    status_data = await poll_status(prompt_id)
    save_image_key = None
    for key, value in status_data.get("outputs", {}).items():
        if "images" in value:
            save_image_key = key
            break
    if save_image_key:
        image_data = await get_image(status_data, save_image_key)
        await save_as_jpg(image_data, prompt_id)
        external_url = f"https://api.lone.blue/img/{prompt_id}.jpg"
        return JSONResponse({"image_url": external_url})
    else:
        return JSONResponse(content={"status": "Processing", "details": status_data}, status_code=202)

@img.get("/image-status/{prompt_id}")
async def get_image_processing_status(prompt_id: str):
    try:
        status_data = await poll_status(prompt_id)
        return JSONResponse(content={"status": "Processing", "details": status_data}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



@img.options("/v1/images/generations", tags=["generations"])
async def get_generation_options():
    return {
        "model": {
            "description": "The model to use for image generation.",
            "type": "string",
            "example": "stable-diffusion"
        },
        "prompt": {
            "description": "The text prompt for the image generation.", 
            "type": "string",
            "required": True,
            "example": "A beautiful sunset over the ocean."
        },
        "n": {
            "description": "The number of images to generate.",
            "type": "integer",
            "default": 1,
            "example": 3
        },
        "size": {
            "description": "The size of the generated images in 'widthxheight' format.",
            "type": "string", 
            "default": "1024x1024",
            "example": "512x512"
        },
        "raw": {
            "description": "Whether to return raw image data or not.",
            "type": "boolean",
            "default": False
        },
        "earlyurl": {
            "description": "Whether to return the URL early or wait for the image to be ready.",
            "type": "boolean",
            "default": False
        }
    }


async def load_workflow(workflow_path: str, workflow:str):
    workflow_path = workflow_path if workflow_path else os.path.join(IMG_WORKFLOWS_DIR, f"{workflow}.json" if not workflow.endswith('.json') else workflow)
    with open(workflow_path, 'r') as file:
        return json.load(file)


async def update_prompt_and_get_key(workflow: dict, post: dict, positive: str):
    '''
Recurses through the workflow searching for and substituting the dynamic values for API_PrePrompt, API_StylePrompt, API_NegativePrompt, width, height, and seed (random integer).
Even more important, it finds and returns the key to the filepath where the file is saved, which we need to decipher status when generation is complete.
    '''
    found_key = [None]
    
    def update_recursive(workflow, path=None):
        if path is None:
            path = []
        
        if isinstance(workflow, dict):
            for key, value in workflow.items():
                current_path = path + [key]
                
                if isinstance(value, dict):
                    if value.get('class_type') == 'SaveImage' and value.get('inputs', {}).get('filename_prefix') == 'API_':
                        found_key[0] = key
                    update_recursive(value, current_path)
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        update_recursive(item, current_path + [str(index)])
                
                if value == "API_PrePrompt":
                    workflow[key] = post.get(value, "") + positive
                elif value in ["API_StylePrompt", "API_NegativePrompt"]:
                    workflow[key] = post.get(value, "")
                elif key in ["seed", "noise_seed"]:
                    workflow[key] = random.randint(1000000000000, 9999999999999)

                elif key in ["width", "max_width", "scaled_width", "height", "max_height", "scaled_height", "side_length", "size", "value", "dimension", "dimensions", "long", "long_side", "short", "short_side", "length"]:
                    l.debug(f"Got a hit for a dimension: {key} {value}")
                    if value == 1023:
                        workflow[key] = post.get("width", 1024)
                        l.debug(f"Set {key} to {workflow[key]}.")
                    elif value == 1025:
                        workflow[key] = post.get("height", 1024)
                        l.debug(f"Set {key} to {workflow[key]}.")

    update_recursive(workflow)
    return found_key[0]


async def queue_prompt(workflow_data):
    await ensure_comfy()
    async with ClientSession() as session:
        
        async with session.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow_data}) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('prompt_id')
            else:
                raise Exception(f"Failed to queue prompt. Status code: {response.status}")