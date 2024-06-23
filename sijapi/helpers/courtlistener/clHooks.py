from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse
import httpx
import json
from pathlib import Path
import asyncio
from datetime import datetime
import os, io
from PyPDF2 import PdfReader
import aiohttp

hook = FastAPI()


# /Users/sij/Library/CloudStorage/OneDrive-WELC/Documents - WELC-Docket
SYNC_FOLDER = Path(__file__).resolve().parent.parent
HOME_FOLDER = Path.home()
DOCKETS_FOLDER = HOME_FOLDER / "Dockets"
SEARCH_FOLDER = HOME_FOLDER / "Watched Cases"
SCRIPTS_FOLDER = SYNC_FOLDER / ".scripts"
REQUESTS_FOLDER = HOME_FOLDER / "sync" / "requests"
COURTLISTENER_BASE_URL = "https://www.courtlistener.com"
COURTLISTENER_DOCKETS_URL = "https://www.courtlistener.com/api/rest/v3/dockets/"
COURTLISTENER_API_KEY = "efb5fe00f3c6c88d65a32541260945befdf53a7e"

with open(SCRIPTS_FOLDER / 'caseTable.json', 'r') as file:
    CASE_TABLE = json.load(file)

@hook.get("/health")
async def health():
    return {"status": "ok"}

@hook.post("/cl/docket")
async def respond(request: Request, background_tasks: BackgroundTasks):
    client_ip = request.client.host
    logging.info(f"Received request from IP: {client_ip}")
    data = await request.json()
    payload = data['payload']
    results = data['payload']['results']
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload_file = REQUESTS_FOLDER / f"{timestamp}-{client_ip}_docket.json"
    with open(payload_file, 'w') as file:
        json.dump(payload, file, indent=2)

    for result in results:
        background_tasks.add_task(process_docket, result)
    return JSONResponse(content={"message": "Received"}, status_code=status.HTTP_200_OK)

async def process_docket(result):
    async with httpx.AsyncClient() as session:
        await process_docket_result(result, session)


async def process_docket_result(result, session):
    docket = str(result.get('docket'))
    case_code, case_shortname = get_case_details(docket)
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
                logging.info(f"Fetching CourtListener docket information for {docket}...")
                data = await response.json()
                court_docket = data['results'][0]['docket_number_core']
                court_docket = f"{court_docket[:2]}-cv-{court_docket[2:]}"  # Formatting the docket number
                case_name = data['results'][0]['case_name']
                logging.info(f"Obtained from CourtListener: docket {court_docket}, case name {case_name}.")
            else:
                logging.info("Failed to fetch data from CourtListener API.")
                court_docket = 'NoCourtDocket'
                case_name = 'NoCaseName'

    for document in result.get('recap_documents', []):
        filepath_ia = document.get('filepath_ia')
        filepath_local = document.get('filepath_local')

        if filepath_ia:
            file_url = filepath_ia
            logging.info(f"Found IA file at {file_url}.")
        elif filepath_local:
            file_url = f"{COURTLISTENER_BASE_URL}/{filepath_local}"
            logging.info(f"Found local file at {file_url}.")
        else:
            logging.info(f"No file URL found in filepath_ia or filepath_local for one of the documents.")
            continue

        document_number = document.get('document_number', 'NoDocumentNumber')
        description = document.get('description', 'NoDescription').replace(" ", "_").replace("/", "_")
        description = description[:50]  # Truncate description
        # case_shortname = case_name # TEMPORARY OVERRIDE
        file_name = f"{case_code}_{document_number}_{date_filed_formatted}_{description}.pdf"
        target_path = Path(DOCKETS_FOLDER) / case_shortname / "Docket" / file_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await download_file(file_url, target_path, session)
        logging.info(f"Downloaded {file_name} to {target_path}")


def get_case_details(docket):
    case_info = CASE_TABLE.get(str(docket), {"code": "000", "shortname": "UNKNOWN"})
    case_code = case_info.get("code")
    short_name = case_info.get("shortname")
    return case_code, short_name



async def download_file(url: str, path: Path, session: aiohttp.ClientSession = None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'
    }
    async with aiohttp.ClientSession() as session:
        logging.info(f"Attempting to download {url} to {path}.")
        try:
            async with session.get(url, headers=headers, allow_redirects=True) as response:
                if response.status == 403:
                    logging.error(f"Access denied (403 Forbidden) for URL: {url}. Skipping download.")
                    return
                response.raise_for_status()

                # Check if the response content type is a PDF
                content_type = response.headers.get('Content-Type')
                if content_type != 'application/pdf':
                    logging.error(f"Invalid content type: {content_type}. Skipping download.")
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
                    logging.error(f"Invalid PDF content: {str(e)}. Skipping download.")
                    return

                # If the PDF is valid, write the content to the file on disk
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('wb') as file:
                    file.write(buffer.getvalue())

        except Exception as e:
            logging.error(f"Error downloading file: {str(e)}")

@hook.post("/cl/search")
async def respond_search(request: Request, background_tasks: BackgroundTasks):
    client_ip = request.client.host
    logging.info(f"Received request from IP: {client_ip}")
    data = await request.json()
    payload = data['payload']
    results = data['payload']['results']

    # Save the payload data
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload_file = REQUESTS_FOLDER / f"{timestamp}-{client_ip}_search.json"
    with open(payload_file, 'w') as file:
        json.dump(payload, file, indent=2)

    for result in results:
        background_tasks.add_task(process_search_result, result)
    return JSONResponse(content={"message": "Received"}, status_code=status.HTTP_200_OK)


async def process_search_result(result):
    async with httpx.AsyncClient() as session:
        download_url = result.get('download_url') 
        court_id = result.get('court_id')
        case_name_short = result.get('caseNameShort')
        case_name = result.get('caseName')
        logging.info(f"Received payload for case {case_name} ({court_id}) and download url {download_url}")

        court_folder = court_id

        if case_name_short:
            case_folder = case_name_short
        else:
            case_folder = case_name

        file_name = download_url.split('/')[-1]
        target_path = Path(SEARCH_FOLDER) / court_folder / case_folder / file_name
        target_path.parent.mkdir(parents=True, exist_ok=True)

        await download_file(download_url, target_path, session)
        logging.info(f"Downloaded {file_name} to {target_path}")