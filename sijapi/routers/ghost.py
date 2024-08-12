'''
WIP. Will be used to manage and update Ghost blogs.
'''
#routers/ghost.py

from fastapi import APIRouter
from datetime import date
import os
import requests
import json
import yaml
import jwt
from sijapi import GHOST_API_KEY, GHOST_API_URL
from sijapi.logs import get_logger
l = get_logger(__name__)

ghost = APIRouter()

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