'''
Used to archive sites visited with browser via the archivist.js UserScript.
'''
# routers/archivist.py

from fastapi import APIRouter, BackgroundTasks, UploadFile, Form, HTTPException, Query, Path as FastAPIPath
import os
import uuid
import asyncio
import shutil
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime as dt_datetime, timedelta
from typing import Optional, List, Tuple
import aiohttp
import aiofiles
import newspaper
import trafilatura
from adblockparser import AdblockRules
from urllib.parse import urlparse
import logging
from typing import Optional
from pathlib import Path
from newspaper import Article
from readability import Document
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime as dt_datetime
from better_profanity import profanity
from sijapi.logs import get_logger
from sijapi.utilities import html_to_markdown, sanitize_filename, assemble_journal_path, assemble_archive_path, contains_profanity, is_ad_or_tracker, initialize_adblock_rules, contains_blacklisted_word
from sijapi import Sys, Archivist, BLOCKLISTS_DIR, OBSIDIAN_VAULT_DIR, OBSIDIAN_RESOURCES_DIR
from sijapi.logs import get_logger
l = get_logger(__name__)

archivist = APIRouter()

adblock_rules = initialize_adblock_rules(BLOCKLISTS_DIR)

@archivist.post("/archive")
async def archive_post(
	url: Optional[str] = Form(None),
	source: Optional[str] = Form(None),
	title: Optional[str] = Form(None),
	encoding: str = Form('utf-8')
):
	if not url:
		l.warning(f"No URL provided to /archive endpoint.")
		raise HTTPException(status_code=400, detail="URL is required")
		
	if is_ad_or_tracker(url, adblock_rules):
		l.debug(f"Skipping likely ad or tracker URL: {url}")
		raise HTTPException(status_code=400, detail="URL is likely an ad or tracker")
	
	markdown_filename = await process_archive(url, title, encoding, source)
	return {"message": "Clip saved successfully", "markdown_filename": markdown_filename}

async def process_archive(
	url: str,
	title: Optional[str] = None,
	encoding: str = 'utf-8',
	source: Optional[str] = None,
) -> Optional[Path]:
	
	# Check URL against blacklist
	if contains_blacklisted_word(url, Archivist.blacklist):
		l.info(f"Not archiving {url} due to blacklisted word in URL")
		return None
	
	timestamp = dt_datetime.now().strftime('%b %d, %Y at %H:%M')
	readable_title = title if title else f"{url} - {timestamp}"
	
	content = await html_to_markdown(url, source)
	if content is None:
		raise HTTPException(status_code=400, detail="Failed to convert content to markdown")
	
	# Check content for profanity
	if contains_profanity(content, threshold=0.01, custom_words=Archivist.blacklist):
		l.info(f"Not archiving {url} due to profanity in content")
		return None
	
	try:
		markdown_path, relative_path = assemble_archive_path(filename=readable_title, extension=".md")
	except Exception as e:
		l.warning(f"Failed to assemble archive path for {url}: {str(e)}")
		return None
	
	markdown_content = f"---\n"
	markdown_content += f"title: \"{readable_title}\"\n"
	markdown_content += f"added: {timestamp}\n"
	markdown_content += f"url: \"{url}\"\n"
	markdown_content += f"date: \"{dt_datetime.now().strftime('%Y-%m-%d')}\"\n"
	markdown_content += f"---\n\n"
	markdown_content += f"# {readable_title}\n\n"
	markdown_content += f"Clipped from [{url}]({url}) on {timestamp}\n\n"
	markdown_content += content
	
	try:
		markdown_path.parent.mkdir(parents=True, exist_ok=True)
		with open(markdown_path, 'w', encoding=encoding) as md_file:
			md_file.write(markdown_content)
		l.debug(f"Successfully saved to {markdown_path}")
		return markdown_path
	except Exception as e:
		l.warning(f"Failed to write markdown file: {str(e)}")
		return None
