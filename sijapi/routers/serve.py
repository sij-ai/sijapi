'''
Web server module. Used by other modules when serving static content is required, e.g. the sd image generation module. Also used to serve PUBLIC_KEY.
'''
import os
from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from pathlib import Path
from datetime import datetime
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi.utilities import bool_convert, sanitize_filename, assemble_journal_path, localize_dt
from sijapi import DATA_DIR, SD_IMAGE_DIR, PUBLIC_KEY, OBSIDIAN_VAULT_DIR

serve = APIRouter(tags=["public"])

@serve.get("/pgp")
async def get_pgp():
    return Response(PUBLIC_KEY, media_type="text/plain")

@serve.get("/img/{image_name}")
def serve_image(image_name: str):
    image_path = os.path.join(SD_IMAGE_DIR, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        return {"error": "Image not found"}
    

def construct_journal_path(date_str: str) -> Path:
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        journal_path = OBSIDIAN_VAULT_DIR / f'journal/{date_obj:%Y}/{date_obj:%Y-%m %B}/{date_obj:%Y-%m-%d %A}/{date_obj:%Y-%m-%d %A}.md'
        return journal_path
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

def is_valid_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

@serve.get("/notes/{file_path:path}")
async def get_file(file_path: str):
    try:
        date_time = localize_dt(file_path);
        absolute_path, local_path = assemble_journal_path(date_time, no_timestamp = True)
    except ValueError as e:
        DEBUG(f"Unable to parse {file_path} as a date, now trying to use it as a local path")
        absolute_path = OBSIDIAN_VAULT_DIR / file_path
        if not absolute_path.suffix:
            absolute_path = Path(absolute_path.with_suffix(".md"))

    if not absolute_path.is_file():
        WARN(f"{absolute_path} is not a valid file it seems.")
    elif absolute_path.suffix == '.md':
        try:
            with open(absolute_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return PlainTextResponse(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal Server Error")
    elif absolute_path.suffix in ['.png', '.jpg', '.jpeg']:
        return FileResponse(absolute_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
