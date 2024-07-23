'''
Web server module. Used by other modules when serving static content is required, e.g. the img image generation module. Also used to serve PUBLIC_KEY.
'''
import os
import io
import string
import json
import time
import base64
import asyncpg
import asyncio
import subprocess
import requests
import random
import paramiko
import aiohttp
import httpx
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from PyPDF2 import PdfReader
from fastapi import APIRouter, Form, HTTPException, Request, Response, BackgroundTasks, status, Path as PathParam
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sijapi import (
    L, API, DB, LOGS_DIR, TS_ID, CASETABLE_PATH, COURTLISTENER_DOCKETS_URL, COURTLISTENER_API_KEY,
    COURTLISTENER_BASE_URL, COURTLISTENER_DOCKETS_DIR, COURTLISTENER_SEARCH_DIR, ALERTS_DIR,
    MAC_UN, MAC_PW, MAC_ID, TS_TAILNET, IMG_DIR, PUBLIC_KEY, OBSIDIAN_VAULT_DIR
)
from sijapi.classes import WidgetUpdate
from sijapi.utilities import bool_convert, sanitize_filename, assemble_journal_path
from sijapi.routers import gis

logger = L.get_module_logger("serve")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.err(text)
def crit(text: str): logger.critical(text)

serve = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "sites")

@serve.get("/pgp")
async def get_pgp():
    return Response(PUBLIC_KEY, media_type="text/plain")

@serve.get("/img/{image_name}")
def serve_image(image_name: str):
    image_path = os.path.join(IMG_DIR, image_name)
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
async def get_file_endpoint(file_path: str):
    try:
        date_time = await gis.dt(file_path);
        absolute_path, local_path = assemble_journal_path(date_time, no_timestamp = True)
    except ValueError as e:
        debug(f"Unable to parse {file_path} as a date, now trying to use it as a local path")
        absolute_path = OBSIDIAN_VAULT_DIR / file_path
        if not absolute_path.suffix:
            absolute_path = Path(absolute_path.with_suffix(".md"))

    if not absolute_path.is_file():
        warn(f"{absolute_path} is not a valid file it seems.")
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


@serve.get("/health_check")
def hook_health():
    shellfish_health_check()

@serve.post("/update_widget")
def hook_widget_update(update: WidgetUpdate):
    shellfish_update_widget(update)

@serve.get("/alert")
async def hook_alert(request: Request):
    alert = request.query_params.get('alert')
    if not alert:
        raise HTTPException(status_code=400, detail='No alert provided.')
    
    return await notify(alert)

@serve.post("/alert/cd")
async def hook_changedetection(webhook_data: dict):
    body = webhook_data.get("body", {})
    message = body.get("message", "")
    
    if message and any(word in message.split() for word in ["SPI", "sierra", "pacific"]):
        filename = ALERTS_DIR / f"alert_{int(time.time())}.json"
        filename.write_text(json.dumps(webhook_data, indent=4))
        notify(message)

    return {"status": "received"}

async def notify(alert: str):
    fail = True
    try:
        if API.EXTENSIONS.shellfish == "on" or API.EXTENSIONS.shellfish == True:
            await notify_shellfish(alert)
            fail = False

        if API.EXTENSIONS.macnotify == "on" or API.EXTENSIONS.macnotify == True:
            if TS_ID == MAC_ID:
                await notify_local(alert)
                fail = False
            else:
                await notify_remote(f"{MAC_ID}.{TS_TAILNET}.net", alert, MAC_UN, MAC_PW)
                fail = False
    except:
        fail = True

    if fail == False:
        info(f"Delivered alert: {alert}")
        return {"message": alert}
    else:
        crit(f"Failed to deliver alert: {alert}")
        return {"message": f"Failed to deliver alert: {alert}"}

async def notify_local(message: str):
    await asyncio.to_thread(os.system, f'osascript -e \'display notification "{message}" with title "Notification Title"\'')


async def notify_remote(host: str, message: str, username: str = None, password: str = None, key_filename: str = None):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    connect_kwargs = {'hostname': host, 'username': username}
    if key_filename:
        connect_kwargs['key_filename'] = key_filename
    else:
        connect_kwargs['password'] = password

    await asyncio.to_thread(ssh.connect, **connect_kwargs)
    await asyncio.to_thread(ssh.exec_command, f'osascript -e \'display notification "{message}" with title "Notification Title"\'')
    ssh.close()


if API.EXTENSIONS.shellfish == "on" or API.EXTENSIONS.shellfish == True:
    async def notify_shellfish(alert: str):
        key = "d7e810e7601cd296a05776c169b4fe97a6a5ee1fd46abe38de54f415732b3f4b"
        user = "WuqPwm1VpGijF4U5AnIKzqNMVWGioANTRjJoonPm"
        iv = "ab5bbeb426015da7eedcee8bee3dffb7"
        
        plain = "Secure ShellFish Notify 2.0\n" + base64.b64encode(alert.encode()).decode() + "\n"

        openssl_command = [
            "openssl", "enc", "-aes-256-cbc", "-base64", "-K", key, "-iv", iv
        ]
        
        process = await asyncio.to_thread(subprocess.Popen, openssl_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await asyncio.to_thread(process.communicate, plain.encode())
        
        if process.returncode != 0:
            raise Exception(f"OpenSSL encryption failed: {stderr.decode()}")

        base64_encoded = stdout.decode().strip()

        url = f"https://secureshellfish.app/push/?user={user}&mutable"
        headers = {"Content-Type": "text/plain"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=base64_encoded) as response:
                if response.status != 200:
                    raise Exception(f"Failed to send notification: {response.status_code}")
                
    def shellfish_health_check():
        addresses = [
            "https://api.sij.ai/health",
            "http://100.64.64.20:4444/health",
            "http://100.64.64.30:4444/health",
            "http://100.64.64.11:4444/health",
            "http://100.64.64.15:4444/health"
        ]
        
        results = []
        up_count = 0
        for address in addresses:
            try:
                response = requests.get(address)
                if response.status_code == 200:
                    results.append(f"{address} is up")
                    up_count += 1
                else:
                    results.append(f"{address} returned status code {response.status_code}")
            except requests.exceptions.RequestException:
                results.append(f"{address} is down")
        
        # Generate a simple text-based graph
        graph = '|' * up_count + '.' * (len(addresses) - up_count)
        text_update = "\n".join(results)
        
        widget_command = ["widget", "--text", text_update, "--text", f"Graph: {graph}", "--icon", "network"]
        output = shellfish_run_widget_command(widget_command)
        return {"output": output, "graph": graph}


    def shellfish_update_widget(update: WidgetUpdate):
        widget_command = ["widget"]

        if update.text:
            widget_command.extend(["--text", update.text])
        if update.progress:
            widget_command.extend(["--progress", update.progress])
        if update.icon:
            widget_command.extend(["--icon", update.icon])
        if update.color:
            widget_command.extend(["--color", update.color])
        if update.url:
            widget_command.extend(["--url", update.url])
        if update.shortcut:
            widget_command.extend(["--shortcut", update.shortcut])
        if update.graph:
            widget_command.extend(["--text", update.graph])

        output = shellfish_run_widget_command(widget_command)
        return {"output": output}


    def shellfish_run_widget_command(args: List[str]):
        result = subprocess.run(args, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        return result.stdout


if API.EXTENSIONS.courtlistener == "on" or API.EXTENSIONS.courtlistener == True:
    with open(CASETABLE_PATH, 'r') as file:
        CASETABLE = json.load(file)

    @serve.post("/cl/search")
    async def hook_cl_search(request: Request, bg_tasks: BackgroundTasks):
        client_ip = request.client.host
        debug(f"Received request from IP: {client_ip}")
        data = await request.json()
        payload = data['payload']
        results = data['payload']['results']

        # Save the payload data
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        payload_file = LOGS_DIR / f"{timestamp}-{client_ip}_search.json"
        with open(payload_file, 'w') as file:
            json.dump(payload, file, indent=2)

        for result in results:
            bg_tasks.add_task(cl_search_process_result, result)
        return JSONResponse(content={"message": "Received"}, status_code=status.HTTP_200_OK)

    @serve.post("/cl/docket")
    async def hook_cl_docket(request: Request):
        client_ip = request.client.host
        debug(f"Received request from IP: {client_ip}")
        data = await request.json()
        await cl_docket(data, client_ip)

    async def cl_docket(data, client_ip, bg_tasks: BackgroundTasks):
        payload = data['payload']
        results = data['payload']['results']
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        payload_file = LOGS_DIR / f"{timestamp}-{client_ip}_docket.json"
        with open(payload_file, 'w') as file:
            json.dump(payload, file, indent=2)

        for result in results:
            bg_tasks.add_task(cl_docket_process, result)
        return JSONResponse(content={"message": "Received"}, status_code=status.HTTP_200_OK)

    async def cl_docket_process(result):
        async with httpx.AsyncClient() as session:
            await cl_docket_process_result(result, session)

    async def cl_docket_process_result(result, session):
        docket = str(result.get('docket'))
        case_code, case_shortname = cl_case_details(docket)
        date_filed = result.get('date_filed', 'No Date Filed')
        
        try:
            date_filed_formatted = datetime.strptime(date_filed, '%Y-%m-%d').strftime('%Y%m%d')
        except ValueError:
            date_filed_formatted = 'NoDateFiled'

        # Fetching court docket information from the API
        url = f"{COURTLISTENER_DOCKETS_URL}?id={docket}"
        headers = {'Authorization': f'Token {COURTLISTENER_API_KEY}'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:           
                if response.status == 200:
                    debug(f"Fetching CourtListener docket information for {docket}...")
                    data = await response.json()
                    court_docket = data['results'][0]['docket_number_core']
                    court_docket = f"{court_docket[:2]}-cv-{court_docket[2:]}"  # Formatting the docket number
                    case_name = data['results'][0]['case_name']
                    debug(f"Obtained from CourtListener: docket {court_docket}, case name {case_name}.")
                else:
                    debug("Failed to fetch data from CourtListener API.")
                    court_docket = 'NoCourtDocket'
                    case_name = 'NoCaseName'

        for document in result.get('recap_documents', []):
            filepath_ia = document.get('filepath_ia')
            filepath_local = document.get('filepath_local')

            if filepath_ia:
                file_url = filepath_ia
                debug(f"Found IA file at {file_url}.")
            elif filepath_local:
                file_url = f"{COURTLISTENER_BASE_URL}/{filepath_local}"
                debug(f"Found local file at {file_url}.")
            else:
                debug(f"No file URL found in filepath_ia or filepath_local for one of the documents.")
                continue

            document_number = document.get('document_number', 'NoDocumentNumber')
            description = document.get('description', 'NoDescription').replace(" ", "_").replace("/", "_")
            description = description[:50]  # Truncate description
            # case_shortname = case_name # TEMPORARY OVERRIDE
            file_name = f"{case_code}_{document_number}_{date_filed_formatted}_{description}.pdf"
            target_path = Path(COURTLISTENER_DOCKETS_DIR) / case_shortname / "Docket" / file_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            await cl_download_file(file_url, target_path, session)
            debug(f"Downloaded {file_name} to {target_path}")

    def cl_case_details(docket):
        case_info = CASETABLE.get(str(docket), {"code": "000", "shortname": "UNKNOWN"})
        case_code = case_info.get("code")
        short_name = case_info.get("shortname")
        return case_code, short_name

    async def cl_download_file(url: str, path: Path, session: aiohttp.ClientSession = None):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'
        }
        async with aiohttp.ClientSession() as session:
            debug(f"Attempting to download {url} to {path}.")
            try:
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    if response.status == 403:
                        err(f"Access denied (403 Forbidden) for URL: {url}. Skipping download.")
                        return
                    response.raise_for_status()

                    # Check if the response content type is a PDF
                    content_type = response.headers.get('Content-Type')
                    if content_type != 'application/pdf':
                        err(f"Invalid content type: {content_type}. Skipping download.")
                        return

                    # Create an in-memory buffer to store the downloaded content
                    buffer = io.BytesIO()
                    async for chunk in response.content.iter_chunked(1024):
                        buffer.write(chunk)

                    # Reset the buffer position to the beginning
                    buffer.seek(0)

                    # Validate the downloaded PDF content
                    try:
                        PdfReader(buffer)
                    except Exception as e:
                        err(f"Invalid PDF content: {str(e)}. Skipping download.")
                        return

                    # If the PDF is valid, write the content to the file on disk
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('wb') as file:
                        file.write(buffer.getvalue())

            except Exception as e:
                err(f"Error downloading file: {str(e)}")


    async def cl_search_process_result(result):
        async with httpx.AsyncClient() as session:
            download_url = result.get('download_url') 
            court_id = result.get('court_id')
            case_name_short = result.get('caseNameShort')
            case_name = result.get('caseName')
            debug(f"Received payload for case {case_name} ({court_id}) and download url {download_url}")

            court_folder = court_id

            if case_name_short:
                case_folder = case_name_short
            else:
                case_folder = case_name

            file_name = download_url.split('/')[-1]
            target_path = Path(COURTLISTENER_SEARCH_DIR) / court_folder / case_folder / file_name
            target_path.parent.mkdir(parents=True, exist_ok=True)

            await cl_download_file(download_url, target_path, session)
            debug(f"Downloaded {file_name} to {target_path}")


@serve.get("/s", response_class=HTMLResponse)
async def shortener_form(request: Request):
    return templates.TemplateResponse("shortener.html", {"request": request})


@serve.post("/s")
async def create_short_url(request: Request, long_url: str = Form(...), custom_code: Optional[str] = Form(None)):
    async with DB.get_connection() as conn:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS short_urls (
                short_code VARCHAR(3) PRIMARY KEY,
                long_url TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        if custom_code:
            if len(custom_code) != 3 or not custom_code.isalnum():
                return templates.TemplateResponse("shortener.html", {"request": request, "error": "Custom code must be 3 alphanumeric characters"})
            
            # Check if custom code already exists
            existing = await conn.fetchval('SELECT 1 FROM short_urls WHERE short_code = $1', custom_code)
            if existing:
                return templates.TemplateResponse("shortener.html", {"request": request, "error": "Custom code already in use"})
            
            short_code = custom_code
        else:
            # Generate a random 3-character alphanumeric string
            chars = string.ascii_letters + string.digits
            while True:
                short_code = ''.join(random.choice(chars) for _ in range(3))
                existing = await conn.fetchval('SELECT 1 FROM short_urls WHERE short_code = $1', short_code)
                if not existing:
                    break

        await conn.execute(
            'INSERT INTO short_urls (short_code, long_url) VALUES ($1, $2)',
            short_code, long_url
        )

    short_url = f"https://sij.ai/{short_code}"
    return templates.TemplateResponse("shortener.html", {"request": request, "short_url": short_url})


@serve.get("/s", response_class=HTMLResponse)
async def shortener_form(request: Request):
    return templates.TemplateResponse("shortener.html", {"request": request})

@serve.post("/s")
async def create_short_url(request: Request, long_url: str = Form(...), custom_code: Optional[str] = Form(None)):
    async with DB.get_connection() as conn:
        await create_short_urls_table(conn)

        if custom_code:
            if len(custom_code) != 3 or not custom_code.isalnum():
                return templates.TemplateResponse("shortener.html", {"request": request, "error": "Custom code must be 3 alphanumeric characters"})
            
            # Check if custom code already exists
            existing = await conn.fetchval('SELECT 1 FROM short_urls WHERE short_code = $1', custom_code)
            if existing:
                return templates.TemplateResponse("shortener.html", {"request": request, "error": "Custom code already in use"})
            
            short_code = custom_code
        else:
            # Generate a random 3-character alphanumeric string
            chars = string.ascii_letters + string.digits
            while True:
                short_code = ''.join(random.choice(chars) for _ in range(3))
                existing = await conn.fetchval('SELECT 1 FROM short_urls WHERE short_code = $1', short_code)
                if not existing:
                    break

        await conn.execute(
            'INSERT INTO short_urls (short_code, long_url) VALUES ($1, $2)',
            short_code, long_url
        )

    short_url = f"https://sij.ai/{short_code}"
    return templates.TemplateResponse("shortener.html", {"request": request, "short_url": short_url})

@serve.get("/{short_code}", response_class=RedirectResponse, status_code=301)
async def redirect_short_url(request: Request, short_code: str = PathParam(..., min_length=3, max_length=3)):
    if request.headers.get('host') != 'sij.ai':
        raise HTTPException(status_code=404, detail="Not Found")
    
    async with DB.get_connection() as conn:
        result = await conn.fetchrow(
            'SELECT long_url FROM short_urls WHERE short_code = $1',
            short_code
        )
    
    if result:
        return result['long_url']
    else:
        raise HTTPException(status_code=404, detail="Short URL not found")

async def create_short_urls_table(conn):
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS short_urls (
            short_code VARCHAR(3) PRIMARY KEY,
            long_url TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')