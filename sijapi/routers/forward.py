'''
Used for port-forwarding and reverse proxy configurations.
'''
#routers/forward.py

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
    L, API, Serve, LOGS_DIR, TS_ID, CASETABLE_PATH, COURTLISTENER_DOCKETS_URL, COURTLISTENER_API_KEY,
    COURTLISTENER_BASE_URL, COURTLISTENER_DOCKETS_DIR, COURTLISTENER_SEARCH_DIR, ALERTS_DIR,
    MAC_UN, MAC_PW, MAC_ID, TS_TAILNET, IMG_DIR, PUBLIC_KEY, OBSIDIAN_VAULT_DIR
)
from sijapi.classes import WidgetUpdate
from sijapi.utilities import bool_convert, sanitize_filename, assemble_journal_path
from sijapi.routers import gis

forward = APIRouter()

logger = L.get_module_logger("email")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

async def forward_traffic(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, destination: str):
    try:
        dest_host, dest_port = destination.split(':')
        dest_port = int(dest_port)
    except ValueError:
        warn(f"Invalid destination format: {destination}. Expected 'host:port'.")
        writer.close()
        await writer.wait_closed()
        return
    
    try:
        dest_reader, dest_writer = await asyncio.open_connection(dest_host, dest_port)
    except Exception as e:
        warn(f"Failed to connect to destination {destination}: {str(e)}")
        writer.close()
        await writer.wait_closed()
        return

    async def forward(src, dst):
        try:
            while True:
                data = await src.read(8192)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except Exception as e:
            warn(f"Error in forwarding: {str(e)}")
        finally:
            dst.close()
            await dst.wait_closed()

    await asyncio.gather(
        forward(reader, dest_writer),
        forward(dest_reader, writer)
    )

async def start_server(source: str, destination: str):
    if ':' in source:
        host, port = source.split(':')
        port = int(port)
    else:
        host = source
        port = 80

    server = await asyncio.start_server(
        lambda r, w: forward_traffic(r, w, destination),
        host,
        port
    )

    async with server:
        await server.serve_forever()


async def start_port_forwarding():
    if hasattr(Serve, 'forwarding_rules'):
        for rule in Serve.forwarding_rules:
            asyncio.create_task(start_server(rule.source, rule.destination))
    else:
        warn("No forwarding rules found in the configuration.")


@forward.get("/forward_status")
async def get_forward_status():
    if hasattr(Serve, 'forwarding_rules'):
        return {"status": "active", "rules": Serve.forwarding_rules}
    else:
        return {"status": "inactive", "message": "No forwarding rules configured"}


asyncio.create_task(start_port_forwarding())