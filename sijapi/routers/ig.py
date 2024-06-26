'''
IN DEVELOPMENT: Instagram AI bot module.
'''
from fastapi import APIRouter, UploadFile
import os
import io
import copy
import re
import jwt
import json
from tqdm import tqdm
import pyotp
import time
import pytz
import requests
import tempfile
import random
import subprocess
import urllib.request
import uuid

from fastapi import APIRouter
from time import sleep
from datetime import timedelta, datetime as date
from PIL import Image
from pydantic import BaseModel
from typing import Dict, List, Optional
import instagrapi
from instagrapi import Client as igClient
from instagrapi.types import UserShort
from urllib.parse import urlparse
from instagrapi.exceptions import LoginRequired as ClientLoginRequiredError
import json
from ollama import Client as oLlama
from sd import sd
from dotenv import load_dotenv
from sijapi import L, COMFYUI_DIR

import io
from io import BytesIO
import base64

ig = APIRouter()

class IG_Request(BaseModel):
    file: Optional[UploadFile] = None # upload a particular file to Instagram
    profile: Optional[str] = None # specify the profile account to use (uses the shortnames defined per folders and the config file)
    local_only: Optional[bool] = False # overrides all other settings to ensure images are generated locally and stay local
    openai: Optional[str] = None # OpenAI API key; if included, will rely on it for DALL-E, GPT-4, and GPT-4-Vision unless otherwise overridden
    llm: Optional[str] = "llama3" # if a valid OpenAI model name is provided, it will be used; otherwise it will attempt to match to an Ollama model (if one exists)
    i2t: Optional[str] = "llava" # set to GPT-4-Vision to use the OpenAI image-2-text model, otherwise this will attempt to match to a vision-capable Ollama model
    t2i: Optional[str] = None # set to DALL-E to use the OpenAI model, or use it to override the StableDiffusion workflow that's otherwise selected. Leave blank to use defaults per the config file
    ig_post: Optional[str] = True # if given a value, will use this as the category of post; if given no value, willuse all categories unless ig_comment_only is enabled
    ig_comment: Optional[str] = None # if given a value, will use this as the category of comment; if given no value, will use all categories unless ig_post_only is enabled
    ig_comment_user: Optional[str] = None # target a particular user for comments
    ig_comment_url: Optional[str] = None # target a particular ig url for comments
    ghost_post: Optional[bool] = True # enable posting to Ghost
    sleep_short: Optional[int] = 5 # average duration of short intervals (a few seconds is adequate; this is to simulate doomscrolling latency) 
    sleep_long: Optional[int] = 180 # agerage duration of long intervals (this should be about a minute at least; it simulates the time it takes to write a comment or prepare a post)

IG_PROFILE = os.getenv("IG_PROFILE")
IG_SHORT_SLEEP = int(os.getenv("IG_SHORT_SLEEP", 5))
IG_LONG_SLEEP = int(os.getenv("IG_LONG_SLEEP", 180))
IG_POST_GHOST = os.getenv("IG_POST_GHOST") 
IG_VISION_LLM = os.getenv("IG_VISION_LLM")
IG_PROMPT_LLM = os.getenv("IG_PROMPT_LLM")
IG_IMG_GEN = os.getenv("IG_IMG_GEN", "ComfyUI")
IG_OUTPUT_PLATFORMS = os.getenv("IG_OUTPUT_PLATFORMS", "ig,ghost,obsidian").split(',')
SD_WORKFLOWS_DIR = os.path.join(COMFYUI_DIR, 'workflows')
COMFYUI_OUTPUT_DIR = COMFYUI_DIR / 'output'
IG_PROFILES_DIR = os.path.join(BASE_DIR, 'profiles')
IG_PROFILE_DIR = os.path.join(IG_PROFILES_DIR, PROFILE)
IG_IMAGES_DIR = os.path.join(IG_PROFILE_DIR, 'images')
IG_PROFILE_CONFIG_PATH = os.path.join(IG_PROFILE_DIR, f'config.json')
IG_VIEWED_IMAGES_DIR = os.path.join(IG_PROFILE_DIR, 'downloads')

with open(IG_PROFILE_CONFIG_PATH, 'r') as config_file:
    PROFILE_CONFIG = json.load(config_file)

if not os.path.exists(IG_IMAGES_DIR):
    os.makedirs(IG_IMAGES_DIR )

OPENAI_API_KEY=PROFILE_CONFIG.get("openai_key")


###################
### VALIDATION ###
##################


if args.profile and args.posttype and not args.custompost and not args.posttype in PROFILE_CONFIG["posts"]:
    print ("ERROR: NO SUCH POST TYPE IS AVAILABLE FOR THIS PROFILE.")

if args.profile and args.commenttype and not args.commenttype in PROFILE_CONFIG["comments"]:
    print ("ERROR: NO SUCH COMMENT TYPE IS AVAILABLE FOR THIS PROFILE.")


####################
### CLIENT SETUP ###
####################

cl = igClient(request_timeout=1)


IMG_GEN = OpenAI(api_key=OPENAI_API_KEY)
IMG_MODEL = "dall-e-3"

COMFYUI_URL = "http://localhost:8188"
CLIENT_ID = str(uuid.uuid4())


###############################
### INSTAGRAM & GHOST SETUP ###
###############################
IG_USERNAME = PROFILE_CONFIG.get("ig_name")
IG_PASSWORD = PROFILE_CONFIG.get("ig_pass")
IG_SECRET_KEY = PROFILE_CONFIG.get("ig_2fa_secret")
IG_SESSION_PATH = os.path.join(IG_PROFILE_DIR, f'credentials.json')

GHOST_API_URL=PROFILE_CONFIG.get("ghost_admin_url")
GHOST_API_KEY=PROFILE_CONFIG.get("ghost_admin_api_key")
GHOST_CONTENT_KEY=PROFILE_CONFIG.get("ghost_content_key")

########################
### LLM PROMPT SETUP ###
########################
IMG_PROMPT_SYS = PROFILE_CONFIG.get("img_prompt_sys") 
IMG_DESCRIPTION_SYS = PROFILE_CONFIG.get("img_description_sys") 
COMMENT_PROMPT_SYS = PROFILE_CONFIG.get("img_comment_sys")
HASHTAGS = PROFILE_CONFIG.get("preferred_hashtags", [])
IMAGE_URL = args.image_url
rollover_time = 1702605780
COMPLETED_MEDIA_LOG = os.path.join(IG_PROFILE_DIR, f'completed-media.txt')
TOTP = pyotp.TOTP(IG_SECRET_KEY)
SHORT = args.shortsleep
LONG = args.longsleep


def follow_by_username(username) -> bool:
    """
    Follow a user, return true if successful false if not.
    """
    userid = cl.user_id_from_username(username)
    sleep(SHORT)
    return cl.user_follow(userid)

def unfollow_by_username(username) -> bool:
    """
    Unfollow a user, return true if successful false if not.
    """
    userid = cl.user_id_from_username(username)
    sleep(SHORT)
    return cl.user_unfollow(userid)

def get_poster_of_post(shortcode):
    media_info = cl.media_info_by_shortcode(shortcode)
    poster_username = media_info.user.username
    return(poster_username)


def get_followers(amount: int = 0) -> Dict[int, UserShort]:
    """
    Get followers, return  Dict of user_id and User object
    """
    return cl.user_followers(cl.user_id, amount=amount)


def get_followers_usernames(amount: int = 0) -> List[str]:
    """
    Get bot's followers usernames, return List of usernames
    """
    followers = cl.user_followers(cl.user_id, amount=amount)
    sleep(SHORT)
    return [user.username for user in followers.values()]

def get_following(amount: int = 0) -> Dict[int, UserShort]:
    """
    Get bot's followed users, return Dict of user_id and User object
    """
    sleep(SHORT)
    return cl.user_following(cl.user_id, amount=amount)


def get_user_media(username, amount=30):
    """
    Fetch recent media for a given username, return List of medias
    """

    L.DEBUG(f"Fetching recent media for {username}...")
    user_id = cl.user_id_from_username(username)
    medias = cl.user_medias(user_id, amount)
    final_medias = []
    for media in medias:
        sleep(SHORT)
        if media.media_type == 1:
            final_medias.append(media)
    return final_medias


def get_user_image_urls(username, amount=30) -> List[str]:
    """
    Fetch recent media URLs for a given username, return List of media URLs
    """
    L.DEBUG(f"Fetching recent media URLs for {username}...")
    user_id = cl.user_id_from_username(username)
    medias = cl.user_medias(user_id, amount)

    urls = []
    for media in medias:
        sleep(SHORT)
        if media.media_type == 1 and media.thumbnail_url:
            urls.append(media.thumbnail_url)

    return urls

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
        
def get_random_follower():
    followers = cl.get_followers_usernames()
    sleep(SHORT)
    return random.choice(followers)


def get_medias_by_hashtag(hashtag: str, days_ago_max:int = 14, ht_type:str = None, amount:int = args.count):
    if not ht_type:
        ht_type = args.commentmode
    L.DEBUG(f"Fetching {ht_type} media for hashtag: {hashtag}")
    ht_medias = []
    while True:
        sleep(SHORT)
        if ht_type == "top":
            ht_medias.extend(cl.hashtag_medias_top(name=hashtag, amount=amount*10))
        elif ht_type == "recent":
            ht_medias.extend(cl.hashtag_medias_recent(name=hashtag, amount=amount*10))
            
        filtered_medias = filter_medias(ht_medias, days_ago_max=days_ago_max)
        L.DEBUG(f"Filtered {ht_type} media count obtained for '#{hashtag}': {len(filtered_medias)}")
        
        if len(filtered_medias) >= amount:
            L.DEBUG(f"Desired amount of {amount} filtered media reached.")
            break
    
    return filtered_medias

def get_medias_from_all_hashtags(days_ago_max=14, ht_type:str = None, amount:int = args.count):
    if not ht_type:
        ht_type = args.commentmode
    L.DEBUG(f"Fetching {ht_type} media.")
    filtered_medias = []
    while len(filtered_medias) < amount:
        hashtag = random.choice(HASHTAGS)
        L.DEBUG(f"Using hashtag: {hashtag}")
        fetched_medias = []
        sleep(SHORT)
        if ht_type == "top":
            fetched_medias = cl.hashtag_medias_top(name=hashtag, amount=50)  # Fetch a large batch to filter from
        elif ht_type == "recent":
            fetched_medias = cl.hashtag_medias_recent(name=hashtag, amount=50)  # Same for recent
        
        current_filtered_medias = filter_medias(fetched_medias, days_ago_max=days_ago_max)
        filtered_medias.extend(current_filtered_medias)
        L.DEBUG(f"Filtered {ht_type} media count obtained for '#{hashtag}': {len(current_filtered_medias)}")
        
        # Trim the list if we've collected more than needed
        if len(filtered_medias) > amount:
            filtered_medias = filtered_medias[:amount]
            L.DEBUG(f"Desired amount of {amount} filtered media reached.")
            break
        else:
            L.DEBUG(f"Total filtered media count so far: {len(filtered_medias)}")
    
    return filtered_medias

def filter_medias(
    medias: List,
    like_count_min=None,
    like_count_max=None,
    comment_count_min=None,
    comment_count_max=None,
    days_ago_max=None,
):
    # Adjust to use your preferred timezone, for example, UTC
    days_back = date.now(pytz.utc) - timedelta(days=days_ago_max) if days_ago_max else None
    return [
        media for media in medias
        if (
            (like_count_min is None or media.like_count >= like_count_min) and
            (like_count_max is None or media.like_count <= like_count_max) and
            (comment_count_min is None or media.comment_count >= comment_count_min) and
            (comment_count_max is None or media.comment_count <= comment_count_max) and
            (days_ago_max is None or (media.taken_at and media.taken_at > days_back)) and not
            check_media_in_completed_lists(media)
        )
    ]

def add_media_to_completed_lists(media):
    """
    Add a media to the completed lists after interacting with it.
    """
    with open(COMPLETED_MEDIA_LOG, 'a') as file:
        file.write(f"{str(media.pk)}\n")


def check_media_in_completed_lists(media):
    """
    Check if a media is in the completed lists.
    """
    with open(COMPLETED_MEDIA_LOG, 'r') as file:
        completed_media = file.read().splitlines()
    return str(media.pk) in completed_media



def download_and_resize_image(url: str, download_path: str = None, max_dimension: int = 1200) -> str:
    if not isinstance(url, str):
        url = str(url)
    parsed_url = urlparse(url)

    if not download_path or not os.path.isdir(os.path.dirname(download_path)):
        _, temp_file_extension = os.path.splitext(parsed_url.path)
        if not temp_file_extension:
            temp_file_extension = ".jpg"  # Default extension if none is found
        download_path = tempfile.mktemp(suffix=temp_file_extension, prefix="download_")

    if url and parsed_url.scheme and parsed_url.netloc:
        try:
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            with requests.get(url) as response:
                response.raise_for_status()  # Raises an HTTPError if the response was an error
                image = Image.open(BytesIO(response.content))

                # Resize the image, preserving aspect ratio
                if max(image.size) > max_dimension:
                    image.thumbnail((max_dimension, max_dimension))

                # Save the image, preserving the original format if possible
                image_format = image.format if image.format else "jpg"
                image.save(download_path, image_format)
                
                return download_path
        except Exception as e:
            # Handle or log the error as needed
            L.DEBUG(f"Error downloading or resizing image: {e}")
    return None


def comment_on_user_media(user: str, comment_type: str = "default", amount=5):
    """
    Comment on a user's media.
    """
    comment_prompt_usr = PROFILE_CONFIG['comments'][comment_type]['img_comment_usr']
    medias = get_user_media(user, amount)
    for media in medias:
        if not check_media_in_completed_lists(media):
            sleep(SHORT)
            if media.thumbnail_url and is_valid_url(media.thumbnail_url):
                media_path = download_and_resize_image(media.thumbnail_url, f"{IG_VIEWED_IMAGES_DIR}/{media.pk}.jpg")
                if media_path is not None:
                    encoded_media = encode_image_to_base64(media_path)
                    comment_text = llava(encoded_media, COMMENT_PROMPT_SYS, comment_prompt_usr) if args.llava or not args.openai else gpt4v(encoded_media, COMMENT_PROMPT_SYS, comment_prompt_usr)
                    if comment_text:
                        cl.media_comment(media.pk, comment_text)
                        L.DEBUG(f"Commented on media: {media.pk}")
                    else:
                        L.DEBUG(f"Failed to generate comment for media: {media.pk}")
                    add_media_to_completed_lists(media)
                    sleep(SHORT)
                else:
                    L.DEBUG(f"We received a nonetype! {media_path}")
            else:
                L.DEBUG(f"URL for {media.pk} disappeared it seems...")
        else:
            L.DEBUG(f"Media already interacted with: {media.pk}")

def comment_on_hashtagged_media(comment_type: str = args.commenttype, amount=3, hashtag: str = None):
    """
    Comment on a hashtag's media.
    """
    if not hashtag:
        hashtag = random.choice(PROFILE_CONFIG['comments'][comment_type]['hashtags'])

    medias = get_medias_by_hashtag(hashtag=hashtag, days_ago_max=7, amount=amount)

    for media in medias:
        if not check_media_in_completed_lists(media):
            media_path = download_and_resize_image(media.thumbnail_url, f"{IG_VIEWED_IMAGES_DIR}/{media.pk}.jpg")
            comment_text = None

            if media_path and os.path.exists(media_path):
                encoded_media = encode_image_to_base64(media_path)
                comment_prompt_usr = PROFILE_CONFIG['comments'][comment_type]['img_comment_usr'] + " For reference, here is the description that was posted with this image: " + media.caption_text
                comment_text = llava(encoded_media, comment_prompt_usr) if args.llava or not args.openai else gpt4v(encoded_media, COMMENT_PROMPT_SYS, comment_prompt_usr)

            if (PROFILE_CONFIG['comments'][comment_type]['sentiment'] == "positive") and False is True:
                try:
                    like_result = cl.media_like(media)
                    if like_result:
                        L.DEBUG(f"Liked media: https://instagram.com/p/{media.pk}/")
                except instagrapi.exceptions.FeedbackRequired as e:
                    L.DEBUG(f"Cannot like media {media.pk}: {str(e)}")

            if comment_text:
                try:
                    cl.media_comment(media.pk, comment_text)
                    L.DEBUG(f"Commented on media: https://instagram.com/p/{media.pk}/")
                except instagrapi.exceptions.FeedbackRequired as e:
                    L.DEBUG(f"Cannot comment on media {media.pk}: {str(e)}")
            else:
                L.DEBUG(f"Failed to generate comment for media: https://instagram.com/p/{media.pk}")
            add_media_to_completed_lists(media)
            sleep(SHORT)
        else:
            L.DEBUG(f"Media already interacted with: {media.pk}")


def comment_on_specific_media(media_url, comment_type: str = "default"):
    """
    Comment on a specific media given its URL.
    """
    media_id = cl.media_pk_from_url(media_url)
    sleep(SHORT)
    media = cl.media_info(media_id)
    sleep(SHORT)

    media_path = download_and_resize_image(media.thumbnail_url, f"{IG_VIEWED_IMAGES_DIR}/{media.pk}.jpg")
    encoded_media = encode_image_to_base64(media_path)

    comment_prompt_usr = PROFILE_CONFIG['comments'][comment_type]['img_comment_usr'] + " For reference, here is the description that was posted with this image: " + media.caption_text
    comment_text = llava(encoded_media, comment_prompt_usr) if args.llava or not args.openai else gpt4v(encoded_media, COMMENT_PROMPT_SYS, comment_prompt_usr)

    if comment_text:
        try:
            cl.media_comment(media.pk, comment_text)
            L.DEBUG(f"Commented on specific media: https://instagram.com/p/{media.pk}/")
        except instagrapi.exceptions.FeedbackRequired as e:
            L.DEBUG(f"Failed to comment on specific media: https://instagram.com/p/{media.pk}/ due to error: {str(e)}")
    else:
        L.DEBUG(f"Failed to generate comment for specific media: https://instagram.com/p/{media.pk}/")



def get_image(status_data, key):
    """Extract the filename and subfolder from the status data and read the file."""
    try:
        outputs = status_data.get("outputs", {})
        images_info = outputs.get(key, {}).get("images", [])
        if not images_info:
            raise Exception("No images found in the job output.")

        image_info = images_info[0]  # Assuming the first image is the target
        filename = image_info.get("filename")
        subfolder = image_info.get("subfolder", "")  # Default to empty if not present
        file_path = os.path.join(COMFYUI_OUTPUT_DIR, subfolder, filename)

        with open(file_path, 'rb') as file:
            return file.read()
    except KeyError as e:
        raise Exception(f"Failed to extract image information due to missing key: {e}")
    except FileNotFoundError:
        raise Exception(f"File {filename} not found at the expected path {file_path}")
    

def update_prompt(workflow: dict, post: dict, positive: str, found_key=[None], path=None):
    if path is None:
        path = [] 

    try:
        if isinstance(workflow, dict):
            for key, value in workflow.items():
                current_path = path + [key]

                if isinstance(value, dict):
                    if value.get('class_type') == 'SaveImage' and value.get('inputs', {}).get('filename_prefix') == 'API_':
                        found_key[0] = key
                    update_prompt(value, post, positive, found_key, current_path)
                elif isinstance(value, list):
                    # Recursive call with updated path for each item in a list
                    for index, item in enumerate(value):
                        update_prompt(item, post, positive, found_key, current_path + [str(index)])

                if value == "API_PPrompt":
                    workflow[key] = post.get(value, "") + positive
                    L.DEBUG(f"Updated API_PPrompt to: {workflow[key]}")
                elif value == "API_SPrompt":
                    workflow[key] = post.get(value, "")
                    L.DEBUG(f"Updated API_SPrompt to: {workflow[key]}")
                elif value == "API_NPrompt":
                    workflow[key] = post.get(value, "")  
                    L.DEBUG(f"Updated API_NPrompt to: {workflow[key]}")
                elif key == "seed" or key == "noise_seed":
                    workflow[key] = random.randint(1000000000000, 9999999999999)
                    L.DEBUG(f"Updated seed to: {workflow[key]}")
                elif (key == "width" or key == "max_width" or key == "scaled_width" or key == "side_length") and (value == 1023 or value == 1025):
                    # workflow[key] = post.get(value, "")
                    workflow[key] = post.get("width", 1024)
                elif (key == "dimension" or key == "height" or key == "max_height" or key == "scaled_height") and (value == 1023 or value == 1025):
                    # workflow[key] = post.get(value, "")
                    workflow[key] = post.get("height", 1024)
    except Exception as e:
        L.DEBUG(f"Error in update_prompt at path {' -> '.join(path)}: {e}")
        raise

    return found_key[0]

def update_prompt_custom(workflow: dict, API_PPrompt: str, API_SPrompt: str, API_NPrompt: str, found_key=[None], path=None):
    if path is None:
        path = [] 

    try:
        if isinstance(workflow, dict):
            for key, value in workflow.items():
                current_path = path + [key]

                if isinstance(value, dict):
                    if value.get('class_type') == 'SaveImage' and value.get('inputs', {}).get('filename_prefix') == 'API_':
                        found_key[0] = key
                    update_prompt(value, API_PPrompt, API_SPrompt, API_NPrompt, found_key, current_path)
                elif isinstance(value, list):
                    # Recursive call with updated path for each item in a list
                    for index, item in enumerate(value):
                        update_prompt(item, API_PPrompt, API_SPrompt, API_NPrompt, found_key, current_path + [str(index)])

                if value == "API_PPrompt":
                    workflow[key] = API_PPrompt
                    L.DEBUG(f"Updated API_PPrompt to: {workflow[key]}")
                elif value == "API_SPrompt":
                    workflow[key] = API_SPrompt
                    L.DEBUG(f"Updated API_SPrompt to: {workflow[key]}")
                elif value == "API_NPrompt":
                    workflow[key] = API_NPrompt
                    L.DEBUG(f"Updated API_NPrompt to: {workflow[key]}")
                elif key == "seed" or key == "noise_seed":
                    workflow[key] = random.randint(1000000000000, 9999999999999)
                    L.DEBUG(f"Updated seed to: {workflow[key]}")
                elif (key == "width" or key == "max_width" or key == "scaled_width") and (value == 1023 or value == 1025):
                    workflow[key] = 1024
                elif (key == "dimension" or key == "height" or key == "max_height" or key == "scaled_height") and (value == 1023 or value == 1025):
                    workflow[key] = 1024
    except Exception as e:
        L.DEBUG(f"Error in update_prompt_custom at path {' -> '.join(path)}: {e}")
        raise

    return found_key[0]


##################################
### IMAGE GENERATION FUNCTIONS ###
##################################


def image_gen(prompt: str, model: str):

    response = IMG_GEN.images.generate(
    model=model,
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
    )

    image_url = response.data[0].url
    image_path = download_and_resize_image(image_url)
    return image_path
    

def queue_prompt(prompt: dict):
    response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": prompt, "client_id": CLIENT_ID})
    if response.status_code == 200:
        return response.json().get('prompt_id')
    else:
        raise Exception(f"Failed to queue prompt. Status code: {response.status_code}, Response body: {response.text}")

def poll_status(prompt_id):
    """Poll the job status until it's complete and return the status data."""
    start_time = time.time()  # Record the start time
    while True:
        elapsed_time = int(time.time() - start_time)  # Calculate elapsed time in seconds
        status_response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        # Use \r to return to the start of the line, and end='' to prevent newline
        L.DEBUG(f"\rGenerating {prompt_id}. Elapsed time: {elapsed_time} seconds", end='')
        if status_response.status_code != 200:
            raise Exception("Failed to get job status")
        status_data = status_response.json()
        job_data = status_data.get(prompt_id, {})
        if job_data.get("status", {}).get("completed", False):
            L.DEBUG()
            L.DEBUG(f"{prompt_id} completed in {elapsed_time} seconds.")
            return job_data
        time.sleep(1)

def poll_status(prompt_id):
    """Poll the job status until it's complete and return the status data."""
    start_time = time.time()  # Record the start time
    while True:
        elapsed_time = int(time.time() - start_time)  # Calculate elapsed time in seconds
        status_response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        # Use \r to return to the start of the line, and end='' to prevent newline
        L.DEBUG(f"\rGenerating {prompt_id}. Elapsed time: {elapsed_time} seconds", end='')
        if status_response.status_code != 200:
            raise Exception("Failed to get job status")
        status_data = status_response.json()
        job_data = status_data.get(prompt_id, {})
        if job_data.get("status", {}).get("completed", False):
            L.DEBUG()
            L.DEBUG(f"{prompt_id} completed in {elapsed_time} seconds.")
            return job_data
        time.sleep(1)

################################
### PRIMARY ACTIVE FUNCTIONS ###
################################
    
def load_post(chosen_post: str = "default"):
    if chosen_post in PROFILE_CONFIG['posts']:
        post = PROFILE_CONFIG['posts'][chosen_post]
        L.DEBUG(f"Loaded post for {chosen_post}")
    else:
        L.DEBUG(f"Unable to load post for {chosen_post}. Choosing a default post.")
        chosen_post = choose_post(PROFILE_CONFIG['posts'])
        post = PROFILE_CONFIG['posts'][chosen_post]
        L.DEBUG(f"Defaulted to {chosen_post}")

    return post

def handle_image_workflow(chosen_post=None):
    """
    Orchestrates the workflow from prompt update, image generation, to either saving the image and description locally
    or posting to Instagram based on the local flag.
    """
    if chosen_post is None:
        chosen_post = choose_post(PROFILE_CONFIG['posts'])
        
    post = load_post(chosen_post)

    workflow_name = args.workflow if args.workflow else random.choice(post['workflows'])

    L.DEBUG(f"Workflow name: {workflow_name}")

    L.DEBUG(f"Generating image concept for {chosen_post} and {workflow_name} now.")
    image_concept = query_ollama(llmPrompt = post['llmPrompt'], max_tokens = 180) if args.local or not args.openai else query_gpt4(llmPrompt = post['llmPrompt'], max_tokens = 180)
    
    L.DEBUG(f"Image concept for {chosen_post}: {image_concept}")

    workflow_data = None

    if args.fast:
        workflow_data = load_json(None, f"{workflow_name}_fast")
        
    if workflow_data is None:
        workflow_data = load_json(None, workflow_name)

    if args.dalle and not args.local:
        jpg_file_path = image_gen(image_concept, "dall-e-3")
    else:
        saved_file_key = update_prompt(workflow=workflow_data, post=post, positive=image_concept)
        L.DEBUG(f"Saved file key: {saved_file_key}")
        prompt_id = queue_prompt(workflow_data)
        L.DEBUG(f"Prompt ID: {prompt_id}")
        status_data = poll_status(prompt_id)
        image_data = get_image(status_data, saved_file_key)
        if chosen_post == "landscape":
            jpg_file_path = save_as_jpg(image_data, prompt_id, chosen_post, 2880, 100)
        else:
            jpg_file_path = save_as_jpg(image_data, prompt_id, chosen_post, 1440, 90)

    image_aftergen(jpg_file_path, chosen_post)

def handle_custom_image(custom_post: str):
    """
    Orchestrates the workflow from prompt update, image generation, to either saving the image and description locally
    or posting to Instagram based on the local flag.
    """
    if args.posttype:
        post = load_post(args.posttype)
        workflow_name = args.workflow if args.workflow else random.choice(post['workflows'])

    else:
        workflow_name = args.workflow if args.workflow else "selfie"
        post = {
            "API_PPrompt": "",
            "API_SPrompt": "; (((masterpiece))); (beautiful lighting:1), subdued, fine detail, extremely sharp, 8k, insane detail, dynamic lighting, cinematic, best quality, ultra detailed.",
            "API_NPrompt": "canvas frame, 3d, ((bad art)), illustrated, deformed, blurry, duplicate, bad art, bad anatomy, worst quality, low quality, watermark, FastNegativeV2, (easynegative:0.5), epiCNegative, easynegative, verybadimagenegative_v1.3",
            "Vision_Prompt": "Write an upbeat Instagram description with emojis to accompany this selfie!",
            "frequency": 2,
            "ghost_tags": [
                "aigenerated",
                "stablediffusion",
                "sdxl",
            ],
        }

    workflow_data = load_json(None, workflow_name)

    system_msg = "You are a helpful AI who assists in generating prompts that will be used to generate highly realistic images. Always use the most visually descriptive terms possible, and avoid any vague or abstract concepts. Do not include any words or descriptions based on other senses or emotions. Strive to show rather than tell. Space is limited, so be efficient with your words."
    image_concept = query_ollama(system_msg=system_msg, user_msg=custom_post, max_tokens = 180) if args.local or not args.openai else query_gpt4(system_msg=system_msg, user_msg=custom_post, max_tokens = 180)
    
    L.DEBUG(f"Image concept: {image_concept}")    

    if args.dalle and not args.local:
        jpg_file_path = image_gen(image_concept, "dall-e-3")

    else:  
        saved_file_key = update_prompt(workflow=workflow_data, post=post, positive=image_concept)
        L.DEBUG(f"Saved file key: {saved_file_key}")

        prompt_id = queue_prompt(workflow_data)
        L.DEBUG(f"Prompt ID: {prompt_id}")

        status_data = poll_status(prompt_id)
        image_data = get_image(status_data, saved_file_key)
        chosen_post = args.posttype if args.posttype else "custom"
        jpg_file_path = save_as_jpg(image_data, prompt_id, chosen_post, 1440, 90)

        encoded_string = encode_image_to_base64(jpg_file_path)
        vision_prompt = f"Write upbeat Instagram description accompany this image, which was created by AI using the following prompt: {image_concept}"
        instagram_description = llava(encoded_string, vision_prompt) if args.local or args.llava or not args.openai else gpt4v(encoded_string, vision_prompt, 150)
            

    image_aftergen(jpg_file_path, chosen_post, )


def image_aftergen(jpg_file_path: str, chosen_post: str = None, post: Dict = None, prompt: str = None):
    if chosen_post and not prompt:
        prompt = PROFILE_CONFIG['posts'][chosen_post]['Vision_Prompt']
    encoded_string = encode_image_to_base64(jpg_file_path)
    L.DEBUG(f"Image successfully encoded from {jpg_file_path}")
    instagram_description = llava(encoded_string, prompt) if args.local or args.llava or not args.openai else gpt4v(encoded_string, prompt, 150)
    instagram_description = re.sub(r'^["\'](.*)["\']$', r'\1', instagram_description)

    ghost_tags = post['ghost_tags'] if post else PROFILE_CONFIG['posts'][chosen_post]['ghost_tags']

    title_prompt = f"Generate a short 3-5 word title for this image, which already includes the following description: {instagram_description}"

    # Generate img_title based on the condition provided
    img_title = llava(encoded_string, title_prompt) if args.local or args.llava or not args.openai else gpt4v(encoded_string, title_prompt, 150)
    img_title = re.sub(r'^["\'](.*)["\']$', r'\1', img_title)

    # Save description to file and upload or save locally
    description_filename = jpg_file_path.rsplit('.', 1)[0] + ".txt"
    description_path = os.path.join(IG_IMAGES_DIR, description_filename)
    with open(description_path, "w") as desc_file:
        desc_file.write(instagram_description)

    # Initial markdown content creation
    markdown_filename = jpg_file_path.rsplit('.', 1)[0] + ".md"
    markdown_content = f"""# {img_title}

![{img_title}]({jpg_file_path})
---
{instagram_description}
---
Tags: {', '.join(ghost_tags)}
"""
    with open(markdown_filename, "w") as md_file:
        md_file.write(markdown_content)

    L.DEBUG(f"Markdown file created at {markdown_filename}")

    if args.wallpaper:
        change_wallpaper(jpg_file_path)
        L.DEBUG(f"Wallpaper changed.")


    if not args.local:
        ig_footer = ""
        if not args.noig:
            post_url = upload_photo(jpg_file_path, instagram_description)
            L.DEBUG(f"Image posted at {post_url}")
            ig_footer = f"\n<a href=\"{post_url}\">Instagram link</a>"

        if not args.noghost:
            ghost_text = f"{instagram_description}"
            ghost_url = post_to_ghost(img_title, jpg_file_path, ghost_text, ghost_tags)    
            L.DEBUG(f"Ghost post: {ghost_url}\n{ig_footer}")


def choose_post(posts):
    total_frequency = sum(posts[post_type]['frequency'] for post_type in posts)
    random_choice = random.randint(1, total_frequency)
    current_sum = 0
    
    for post_type, post_info in posts.items():
        current_sum += post_info['frequency']
        if random_choice <= current_sum:
            return post_type

def load_json(json_payload, workflow):
    if json_payload:
        return json.loads(json_payload)
    elif workflow:
        workflow_path = os.path.join(SD_WORKFLOWS_DIR, f"{workflow}.json" if not workflow.endswith('.json') else workflow)
        with open(workflow_path, 'r') as file:
            return json.load(file)
    else:
        raise ValueError("No valid input provided.")




def save_as_jpg(image_data, prompt_id, chosen_post:str = None, max_size=2160, quality=80):
    chosen_post = chosen_post if chosen_post else "custom"
    filename_png = f"{prompt_id}.png"
    category_dir = os.path.join(IG_IMAGES_DIR, chosen_post)
    image_path_png = os.path.join(category_dir, filename_png)

    try:
        # Ensure the directory exists
        os.makedirs(category_dir, exist_ok=True)

        # Save the raw PNG data to a file
        with open(image_path_png, 'wb') as file:
            file.write(image_data)

        # Open the PNG, resize it, and save it as jpg
        with Image.open(image_path_png) as img:
            # Resize image if necessary
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple([int(x * ratio) for x in img.size])
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Prepare the path for the converted image
            new_file_name = f"{prompt_id}.jpg"
            new_file_path = os.path.join(category_dir, new_file_name)

            # Convert to jpg and save
            img.convert('RGB').save(new_file_path, format='JPEG', quality=quality)

        # Optionally, delete the temporary PNG file
        os.remove(image_path_png)
        
        return new_file_path
    except Exception as e:
        L.DEBUG(f"Error processing image: {e}")
        return None


def upload_photo(path, caption, title: str=None):
    L.DEBUG(f"Uploading photo from {path}...")
    media = cl.photo_upload(path, caption)
    post_url = f"https://www.instagram.com/p/{media.code}/"
    return post_url

def format_duration(seconds):
    """Return a string representing the duration in a human-readable format."""
    if seconds < 120:
        return f"{int(seconds)} sec"
    elif seconds < 6400:
        return f"{int(seconds // 60)} min"
    else:
        return f"{seconds / 3600:.2f} hr"

########################
### HELPER FUNCTIONS ###
########################

import subprocess

def change_wallpaper(image_path):
    command = """
    osascript -e 'tell application "Finder" to set desktop picture to POSIX file "{}"'
    """.format(image_path)
    subprocess.run(command, shell=True)


def sleep(seconds):
    """Sleep for a random amount of time, approximately the given number of seconds."""
    sleepupto(seconds*0.66, seconds*1.5)

def sleepupto(min_seconds, max_seconds=None):
    interval = random.uniform(min_seconds if max_seconds is not None else 0, max_seconds if max_seconds is not None else min_seconds)
    start_time = time.time()
    end_time = start_time + interval

    with tqdm(total=interval, desc=f"Sleeping for {format_duration(interval)}", unit=" sec", ncols=75, bar_format='{desc}: {bar} {remaining}') as pbar:
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            remaining_time = end_time - current_time
            if elapsed_time >= interval:
                break
            duration = min(1, interval - elapsed_time)  # Adjust sleep time to not exceed interval
            time.sleep(duration)
            pbar.update(duration)
            # Update remaining time display
            pbar.set_postfix_str(f"{format_duration(remaining_time)} remaining")


########################
### GHOST FUNCTIONS ###
########################



def generate_jwt_token():
    key_id, key_secret = GHOST_API_KEY.split(':')
    iat = int(date.now().timestamp())
    exp = iat + 5 * 60  # Token expiration time set to 5 minutes from now for consistency with the working script
    payload = {
        'iat': iat,
        'exp': exp,
        'aud': '/admin/'  # Adjusted to match the working script
    }
    token = jwt.encode(payload, bytes.fromhex(key_secret), algorithm='HS256', headers={'kid': key_id})
    return token.decode('utf-8') if isinstance(token, bytes) else token  # Ensure the token is decoded to UTF-8 string


def post_to_ghost(title, image_path, html_content, ghost_tags):
    jwt_token = generate_jwt_token()
    ghost_headers = {'Authorization': f'Ghost {jwt_token}'}

    # Upload the image to Ghost
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpg')}
        image_response = requests.post(f"{GHOST_API_URL}/images/upload/", headers=ghost_headers, files=files)
        image_response.raise_for_status()  # Ensure the request was successful
        image_url = image_response.json()['images'][0]['url']

    # Prepare the post content
    updated_html_content = f'<img src="{image_url}" alt="Image"/><hr/> {html_content}'
    mobiledoc = {
        "version": "0.3.1",
        "atoms": [],
        "cards": [["html", {"cardName": "html", "html": updated_html_content}]],
        "markups": [],
        "sections": [[10, 0]]
    }
    mobiledoc = json.dumps(mobiledoc)
    
    post_data = {
        'posts': [{
            'title': title,
            'mobiledoc': mobiledoc,
            'status': 'published',
            'tags': ghost_tags
        }]
    }

    # Create a new post
    post_response = requests.post(f"{GHOST_API_URL}/posts/", json=post_data, headers=ghost_headers)
    post_response.raise_for_status()
    post_url = post_response.json()['posts'][0]['url']

    return post_url
    


########################################################
@ig.post("/ig/flow")
async def ig_flow_endpoint(new_session: bool = False):
    current_unix_time = int(date.now().timestamp())    
    time_since_rollover = current_unix_time - rollover_time
    time_remaining = 30 - (time_since_rollover % 30)

    if time_remaining < 4:
        L.DEBUG("Too close to end of TOTP counter. Waiting.")
        sleepupto(5, 5)

        if not new_session and os.path.exists(IG_SESSION_PATH):
            cl.load_settings(IG_SESSION_PATH)
            L.DEBUG("Loaded past session.")

        elif new_session and cl.login(IG_USERNAME, IG_PASSWORD, verification_code=TOTP.now()):
            cl.dump_settings(IG_SESSION_PATH)
            L.DEBUG("Logged in and saved new session.")

        else:
            raise Exception(f"Failed to login as {IG_USERNAME}.")
