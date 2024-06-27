'''
Manages an Obsidian vault, in particular daily notes, using information and functionality drawn from the other routers, primarily calendar, email, ig, llm, rag, sd, serve, time, tts, and weather.
'''
from fastapi import APIRouter, BackgroundTasks, File, UploadFile, Form, HTTPException, Response, Query, Path as FastAPIPath
from fastapi.responses import JSONResponse, PlainTextResponse
from io import BytesIO
from pydantic import BaseModel
import os, re
import uuid
import traceback
import requests
import mimetypes
import shutil
from typing import Optional, Union, Dict, List, Tuple
from urllib.parse import urlparse
from urllib3.util.retry import Retry
from newspaper import Article
import trafilatura
from requests.adapters import HTTPAdapter
import re
import os
from datetime import timedelta, datetime, time as dt_time, date as dt_date
from fastapi import HTTPException, status
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from sijapi import L, OBSIDIAN_VAULT_DIR, OBSIDIAN_RESOURCES_DIR, BASE_URL, OBSIDIAN_BANNER_SCENE, DEFAULT_11L_VOICE, DEFAULT_VOICE, TZ
from sijapi.routers import tts, llm, time, sd, locate, weather, asr, calendar
from sijapi.routers.locate import Location
from sijapi.utilities import assemble_journal_path, convert_to_12_hour_format, sanitize_filename, convert_degrees_to_cardinal, HOURLY_COLUMNS_MAPPING


note = APIRouter()


@note.get("/note/bulk/{dt_start}/{dt_end}")
async def build_daily_note_range_endpoint(dt_start: str, dt_end: str):
    start_date = datetime.strptime(dt_start, "%Y-%m-%d")
    end_date = datetime.strptime(dt_end, "%Y-%m-%d")
    
    results = []
    current_date = start_date
    while current_date <= end_date:
        formatted_date = await locate.localize_datetime(current_date)
        result = await build_daily_note(formatted_date)
        results.append(result)
        current_date += timedelta(days=1)
    
    return {"urls": results}

async def build_daily_note(date_time: datetime):
    '''
Obsidian helper. Takes a datetime and creates a new daily note. Note: it uses the sijapi configuration file to place the daily note and does NOT presently interface with Obsidian's daily note or periodic notes extensions. It is your responsibility to ensure they match.
    '''
    absolute_path, _ = assemble_journal_path(date_time)

    formatted_day = date_time.strftime("%A %B %d, %Y")  # Monday May 27, 2024 formatting
    day_before = (date_time - timedelta(days=1)).strftime("%Y-%m-%d %A")  # 2024-05-26 Sunday formatting
    day_after = (date_time + timedelta(days=1)).strftime("%Y-%m-%d %A")  # 2024-05-28 Tuesday formatting
    header = f"# [[{day_before}|← ]] {formatted_day} [[{day_after}| →]]\n\n"
    
    places = await locate.fetch_locations(date_time)
    location = await locate.reverse_geocode(places[0].latitude, places[0].longitude)
    
    timeslips = await build_daily_timeslips(date_time)

    fm_day = date_time.strftime("%Y-%m-%d")

    _, weather_path = assemble_journal_path(date_time, filename="Weather", extension=".md", no_timestamp = True)
    weather_note = await update_dn_weather(date_time)
    weather_embed = f"![[{weather_path}]]\n"

    events = await update_daily_note_events(date_time)
    _, event_path = assemble_journal_path(date_time, filename="Events", extension=".md", no_timestamp = True)
    event_embed = f"![[{event_path}]]"

    _, task_path = assemble_journal_path(date_time, filename="Tasks", extension=".md", no_timestamp = True)
    task_embed = f"![[{task_path}]]"

    _, note_path = assemble_journal_path(date_time, filename="Notes", extension=".md", no_timestamp = True)
    note_embed = f"![[{note_path}]]"

    _, banner_path = assemble_journal_path(date_time, filename="Banner", extension=".jpg", no_timestamp = True)
   
    body = f"""---
date: "{fm_day}"
banner: "![[{banner_path}]]"
tags:
 - daily-note
created: "{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
---
    
{header}
{weather_embed}

## Events
{event_embed}
 
## Tasks
{task_embed}

## Notes
{note_embed}

## Timeslips
{timeslips}
"""

    with open(absolute_path, 'wb') as f:
        f.write(body.encode())

    banner = await generate_banner(formatted_day, location, weather_note)

    return absolute_path
    

async def build_daily_timeslips(date):
    '''

    '''
    absolute_path, relative_path = assemble_journal_path(date, filename = "Timeslips", extension=".md", no_timestamp = True)
    content = await time.process_timing_markdown(date, date)
    # document_content = await document.read()
    with open(absolute_path, 'wb') as f:
        f.write(content.encode())
    
    return f"![[{relative_path}]]"


### CLIPPER ###
@note.post("/clip")
async def clip_post(
    background_tasks: BackgroundTasks,
    file: UploadFile = None,
    url: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    tts: str = Form('summary'),
    voice: str = Form(DEFAULT_VOICE),
    encoding: str = Form('utf-8')
):
    markdown_filename = await process_article(background_tasks, url, title, encoding, source, tts, voice)
    return {"message": "Clip saved successfully", "markdown_filename": markdown_filename}

@note.post("/archive")
async def archive_post(
    background_tasks: BackgroundTasks,
    file: UploadFile = None,
    url: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    encoding: str = Form('utf-8')
):
    markdown_filename = await process_archive(background_tasks, url, title, encoding, source)
    return {"message": "Clip saved successfully", "markdown_filename": markdown_filename}

@note.get("/clip")
async def clip_get(
    background_tasks: BackgroundTasks,
    url: str,
    title: Optional[str] = Query(None),
    encoding: str = Query('utf-8'),
    tts: str = Query('summary'),
    voice: str = Query(DEFAULT_VOICE)
):
    markdown_filename = await process_article(background_tasks, url, title, encoding, tts=tts, voice=voice)
    return {"message": "Clip saved successfully", "markdown_filename": markdown_filename}

@note.post("/note/add")
async def note_add_endpoint(file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), source: Optional[str] = Form(None)):
    if not file and not text:
        raise HTTPException(status_code=400, detail="Either text or a file must be provided")
    else:
        result = await process_for_daily_note(file, text, source)
        L.INFO(f"Result on /note/add: {result}")
        return JSONResponse(result, status_code=204)

async def process_for_daily_note(file: Optional[UploadFile] = File(None), text: Optional[str] = None, source: Optional[str] = None):
    now = datetime.now()

    transcription_entry = ""
    file_entry = ""
    if file:
        file_content = await file.read()
        audio_io = BytesIO(file_content)
        file_type, _ = mimetypes.guess_type(file.filename) 

        if 'audio' in file_type:
            subdir = "Audio"
        elif 'image' in file_type:
            subdir = "Images"
        else:
            subdir = "Documents"

        absolute_path, relative_path = assemble_journal_path(now, subdir=subdir, filename=file.filename)
        with open(absolute_path, 'wb') as f:
            f.write(file_content)

        if 'audio' in file_type:
            transcription = await asr.transcribe_audio(file_path=absolute_path, params=asr.TranscribeParams(model="small-en", language="en", threads=6))
            file_entry = f"![[{relative_path}]]"

        elif 'image' in file_type:
            file_entry = f"![[{relative_path}]]"
        
        else:
            file_entry = f"[Source]({relative_path})"
    

    text_entry = text if text else ""
    L.INFO(f"transcription: {transcription}\nfile_entry: {file_entry}\ntext_entry: {text_entry}")
    return await add_to_daily_note(transcription, file_entry, text_entry, now)


async def add_to_daily_note(transcription: str = None, file_link: str = None, additional_text: str = None, date_time: datetime = None):
    date_time = date_time or datetime.now()
    note_path, _ = assemble_journal_path(date_time, filename='Notes', extension=".md", no_timestamp = True)
    time_str = date_time.strftime("%H:%M")
    
    entry_lines = []
    if additional_text and additional_text.strip():
        entry_lines.append(f"\t* {additional_text.strip()}") 
    if transcription and transcription.strip():
        entry_lines.append(f"\t* {transcription.strip()}") 
    if file_link and file_link.strip():
        entry_lines.append(f"\t\t {file_link.strip()}")

    entry = f"\n* **{time_str}**\n" + "\n".join(entry_lines)

    # Write the entry to the end of the file
    if note_path.exists():
        with open(note_path, 'a', encoding='utf-8') as note_file:
            note_file.write(entry)
    else: 
        date_str = date_time.strftime("%Y-%m-%d")
        frontmatter = f"""---
date: {date_str}
tags:
 - notes
---

"""
        content = frontmatter + entry
        # If the file doesn't exist, create it and start with "Notes"
        with open(note_path, 'w', encoding='utf-8') as note_file:
            note_file.write(content)

    return entry

async def handle_text(title:str, summary:str, extracted_text:str, date_time: datetime = None):
    date_time = date_time if date_time else datetime.now()
    absolute_path, relative_path = assemble_journal_path(date_time, filename=title, extension=".md", no_timestamp = True)
    with open(absolute_path, "w") as file:
        file.write(f"# {title}\n\n## Summary\n{summary}\n\n## Transcript\n{extracted_text}")
        
    # add_to_daily_note(f"**Uploaded [[{title}]]**: *{summary}*", absolute_path)

    return True


async def process_document(
    background_tasks: BackgroundTasks,
    document: File,
    title: Optional[str] = None,
    tts_mode: str = "summary",
    voice: str = DEFAULT_VOICE
):
    timestamp = datetime.now().strftime('%b %d, %Y at %H:%M')

    # Save the document to OBSIDIAN_RESOURCES_DIR
    document_content = await document.read()
    file_path = Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR / document.filename
    with open(file_path, 'wb') as f:
        f.write(document_content)

    parsed_content = await llm.extract_text(file_path)  # Ensure extract_text is awaited

    llm_title, summary = await llm.title_and_summary(parsed_content)
    try:
        readable_title = sanitize_filename(title if title else document.filename)

        if tts_mode == "full" or tts_mode == "content" or tts_mode == "body":
            tts_text = parsed_content
        elif tts_mode == "summary" or tts_mode == "excerpt":
            tts_text = summary
        else:
            tts_text = None

        frontmatter = f"""---
title: {readable_title}
added: {timestamp}
---
"""
        body = f"# {readable_title}\n\n"

        if tts_text:
            try:
                datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
                audio_filename = f"{datetime_str} {readable_title}"
                audio_path = await tts.generate_speech(
                    background_tasks=background_tasks,
                    text=tts_text,
                    voice=voice,
                    model="eleven_turbo_v2",
                    podcast=True,
                    title=audio_filename,
                    output_dir=Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR
                )
                audio_ext = Path(audio_path).suffix
                obsidian_link = f"![[{OBSIDIAN_RESOURCES_DIR}/{audio_filename}{audio_ext}]]"
                body += f"{obsidian_link}\n\n"
            except Exception as e:
                L.ERR(f"Failed in the TTS portion of clipping: {e}")

        body += f"> [!summary]+\n"
        body += f"> {summary}\n\n"
        body += parsed_content
        markdown_content = frontmatter + body

        markdown_filename = f"{readable_title}.md"
        encoding = 'utf-8'

        with open(markdown_filename, 'w', encoding=encoding) as md_file:
            md_file.write(markdown_content)

        L.INFO(f"Successfully saved to {markdown_filename}")

        return markdown_filename

    except Exception as e:
        L.ERR(f"Failed to clip: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_article(
    background_tasks: BackgroundTasks,
    url: str,
    title: Optional[str] = None,
    encoding: str = 'utf-8',
    source: Optional[str] = None,
    tts_mode: str = "summary", 
    voice: str = DEFAULT_11L_VOICE
):

    timestamp = datetime.now().strftime('%b %d, %Y at %H:%M')

    parsed_content = await parse_article(url, source)
    if parsed_content is None:
        return {"error": "Failed to retrieve content"}

    readable_title = sanitize_filename(title or parsed_content.get("title") or timestamp)
    markdown_filename, relative_path = assemble_journal_path(datetime.now(), subdir="Articles", filename=readable_title, extension=".md")

    try:
        summary = await llm.summarize_text(parsed_content["content"], "Summarize the provided text. Respond with the summary and nothing else. Do not otherwise acknowledge the request. Just provide the requested summary.")
        summary = summary.replace('\n', ' ')  # Remove line breaks

        if tts_mode == "full" or tts_mode == "content":
            tts_text = parsed_content["content"]
        elif tts_mode == "summary" or tts_mode == "excerpt":
            tts_text = summary
        else:
            tts_text = None

        banner_markdown = ''
        try:
            banner_url = parsed_content.get('image', '')
            if banner_url != '':
                banner_image = download_file(banner_url, Path(OBSIDIAN_VAULT_DIR / OBSIDIAN_RESOURCES_DIR))
                if banner_image:
                    banner_markdown = f"![[{OBSIDIAN_RESOURCES_DIR}/{banner_image}]]"
                
        except Exception as e:
            L.ERR(f"No image found in article")

        authors = ', '.join('[[{}]]'.format(author) for author in parsed_content.get('authors', ['Unknown']))

        frontmatter = f"""---
title: {readable_title}
authors: {', '.join('[[{}]]'.format(author) for author in parsed_content.get('authors', ['Unknown']))}
published: {parsed_content.get('date_published', 'Unknown')}
added: {timestamp}
excerpt: {parsed_content.get('excerpt', '')}
banner: "{banner_markdown}"
tags:

"""
        frontmatter += '\n'.join(f" - {tag}" for tag in parsed_content.get('tags', []))
        frontmatter += '\n---\n'

        body = f"# {readable_title}\n\n"

        if tts_text:
            datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
            audio_filename = f"{datetime_str} {readable_title}"
            try:
                audio_path = await tts.generate_speech(background_tasks=background_tasks, text=tts_text, voice=voice, model="eleven_turbo_v2", podcast=True, title=audio_filename,
                output_dir=Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR)
                audio_ext = Path(audio_path).suffix
                obsidian_link = f"![[{OBSIDIAN_RESOURCES_DIR}/{audio_filename}{audio_ext}]]"
                body += f"{obsidian_link}\n\n"
            except Exception as e:
                L.ERR(f"Failed to generate TTS for np3k. {e}")

        try:
            body += f"by {authors} in [{parsed_content.get('domain', urlparse(url).netloc.replace('www.', ''))}]({url}).\n\n"
            body += f"> [!summary]+\n"
            body += f"> {summary}\n\n"
            body += parsed_content["content"]
            markdown_content = frontmatter + body

        except Exception as e:
            L.ERR(f"Failed to combine elements of article markdown.")

        try:
            with open(markdown_filename, 'w', encoding=encoding) as md_file:
                md_file.write(markdown_content)

            L.INFO(f"Successfully saved to {markdown_filename}")
            add_to_daily_note
            return markdown_filename
        
        except Exception as e:
            L.ERR(f"Failed to write markdown file")
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        L.ERR(f"Failed to clip {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def parse_article(url: str, source: Optional[str] = None):
    source = source if source else trafilatura.fetch_url(url)
    traf = trafilatura.extract_metadata(filecontent=source, default_url=url)

    # Pass the HTML content to newspaper3k:
    np3k = Article(url)
    np3k.set_html(source)
    np3k.parse()

    L.INFO(f"Parsed {np3k.title}")
    

    title = np3k.title or traf.title
    authors = np3k.authors or traf.author
    authors = authors if isinstance(authors, List) else [authors]
    date = np3k.publish_date or traf.date
    try:
        date = await locate.localize_datetime(date)
    except:
        L.DEBUG(f"Failed to localize {date}")
        date = await locate.localize_datetime(datetime.now())
    excerpt = np3k.meta_description or traf.description
    content = trafilatura.extract(source, output_format="markdown", include_comments=False) or np3k.text
    image = np3k.top_image or traf.image
    domain = traf.sitename or urlparse(url).netloc.replace('www.', '').title()
    tags = np3k.meta_keywords or traf.categories or traf.tags
    tags = tags if isinstance(tags, List) else [tags]

    return {
        'title': title.replace("  ", " "),
        'authors': authors,
        'date': date.strftime("%b %d, %Y at %H:%M"),
        'excerpt': excerpt,
        'content': content,
        'image': image,
        'url': url,
        'domain': domain,
        'tags': np3k.meta_keywords
    }



async def process_archive(
    background_tasks: BackgroundTasks,
    url: str,
    title: Optional[str] = None,
    encoding: str = 'utf-8',
    source: Optional[str] = None,
):

    timestamp = datetime.now().strftime('%b %d, %Y at %H:%M')

    parsed_content = await parse_article(url, source)
    if parsed_content is None:
        return {"error": "Failed to retrieve content"}
    content = parsed_content["content"]

    readable_title = sanitize_filename(title if title else parsed_content.get("title", "Untitled"))
    if not readable_title:
        readable_title = timestamp

    markdown_path = OBSIDIAN_VAULT_DIR / "archive"

    try:
        frontmatter = f"""---
title: {readable_title}
author: {parsed_content.get('author', 'Unknown')}
published: {parsed_content.get('date_published', 'Unknown')}
added: {timestamp}
excerpt: {parsed_content.get('excerpt', '')}
---
"""
        body = f"# {readable_title}\n\n"

        try:
            authors = parsed_content.get('author', '')
            authors_in_brackets = [f"[[{author.strip()}]]" for author in authors.split(",")]
            authors_string = ", ".join(authors_in_brackets)

            body += f"by {authors_string} in [{parsed_content.get('domain', urlparse(url).netloc.replace('www.', ''))}]({parsed_content.get('url', url)}).\n\n"
            body += content
            markdown_content = frontmatter + body
        except Exception as e:
            L.ERR(f"Failed to combine elements of article markdown.")

        try:
            with open(markdown_path, 'w', encoding=encoding) as md_file:
                md_file.write(markdown_content)

            L.INFO(f"Successfully saved to {markdown_path}")
            add_to_daily_note
            return markdown_path
        
        except Exception as e:
            L.ERR(f"Failed to write markdown file")
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        L.ERR(f"Failed to clip {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def download_file(url, folder):
    os.makedirs(folder, exist_ok=True)
    filename = str(uuid.uuid4()) + os.path.splitext(urlparse(url).path)[-1]
    filepath = os.path.join(folder, filename)
    
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            if 'image' in response.headers.get('Content-Type', ''):
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            else:
                L.ERR(f"Failed to download image: {url}, invalid content type: {response.headers.get('Content-Type')}")
                return None
        else:
            L.ERR(f"Failed to download image: {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        L.ERR(f"Failed to download image: {url}, error: {str(e)}")
        return None
    return filename

def copy_file(local_path, folder):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.basename(local_path)
    destination_path = os.path.join(folder, filename)
    shutil.copy(local_path, destination_path)
    return filename



async def save_file(file: UploadFile, folder: Path) -> Path:
    file_path = folder / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    return file_path



    
### FRONTMATTER, BANNER

@note.put("/note/update_frontmatter")
async def update_frontmatter_endpoint(date: str, key: str, value: str):
    date_time = datetime.strptime(date, "%Y-%m-%d")
    result = await update_frontmatter(date_time, key, value)
    return result
    
async def update_frontmatter(date_time: datetime, key: str, value: str):
    # Parse the date and format paths
    file_path, relative_path = assemble_journal_path(date_time)

    # Check if the file exists
    if not file_path.exists():
        L.CRIT(f"Markdown file not found at {file_path}")
        raise HTTPException(status_code=404, detail="Markdown file not found.")

    # Read the file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Extract the frontmatter
    try:
        start_index = lines.index("---\n") + 1
        end_index = lines[start_index:].index("---\n") + start_index
        frontmatter = lines[start_index:end_index]
    except ValueError:
        raise HTTPException(status_code=500, detail="Frontmatter not found.")

    # Remove the existing key if present
    pattern = re.compile(f"^{key}:.*", re.IGNORECASE)
    frontmatter = [line for line in frontmatter if not pattern.match(line)]

    # Process value as a CSV string into a list
    values = value.split(',')

    # Determine insertion format
    if len(values) == 1:
        # Single value, add as a simple key-value
        new_entry = f"{key}: {values[0]}\n"
    else:
        # Multiple values, format as a list under the key
        new_entry = f"{key}:\n" + "\n".join([f" - {val}" for val in values]) + "\n"

    # Insert the new key-value(s)
    frontmatter.append(new_entry)

    # Reassemble the file
    content = lines[:start_index] + frontmatter + ["---\n"] + lines[end_index + 1:]

    # Write changes back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(content)

    return {"message": "Frontmatter updated successfully."}

@note.post("/note/banner")
async def banner_endpoint(dt: str, location: str = None, mood: str = None, other_context: str = None):
    '''
        Endpoint (POST) that generates a new banner image for the Obsidian daily note for a specified date, taking into account optional additional information, then updates the frontmatter if necessary.
    '''
    L.DEBUG(f"banner_endpoint requested with date: {dt} ({type(dt)})")
    date_time = await locate.localize_datetime(dt)
    L.DEBUG(f"date_time after localization: {date_time} ({type(date_time)})")
    jpg_path = await generate_banner(date_time, location, mood=mood, other_context=other_context)
    return jpg_path


async def get_note(date_time: datetime):
    date_time = await locate.localize_datetime(date_time);
    absolute_path, local_path = assemble_journal_path(date_time, filename = "Notes", extension = ".md", no_timestamp = True)

    if absolute_path.is_file():
        with open(absolute_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content if content else None

async def sentiment_analysis(date_time: datetime):
    most_recent_note = await get_note(date_time)
    most_recent_note = most_recent_note or await get_note(date_time - timedelta(days=1))
    if most_recent_note:
        sys_msg = "You are a sentiment analysis AI bot. Your task is to analyze text and give a one-word description of the mood it contains, such as 'optimistic', 'pensive', 'nostalgic', 'livid', et cetera."
        prompt = f"Provide sentiment analysis of the following notes: {most_recent_note}"
        multishot_prompt = ["Provide sentiment analysis of the following notes: I am sad today my girlfriend broke up with me", "lonely", "Provide sentiment analysis of the following notes: Work has been so busy lately it is like there are not enough hours in the day", "hectic", prompt]
        analysis = await llm.query_ollama_multishot(multishot_prompt, sys_msg, max_tokens = 10)
        return analysis
    else:
        return ""

async def generate_banner(dt, location: Location = None, forecast: str = None, mood: str = None, other_context: str = None):
    # L.DEBUG(f"Location: {location}, forecast: {forecast}, mood: {mood}, other_context: {other_context}")
    date_time = await locate.localize_datetime(dt)
    L.DEBUG(f"generate_banner called with date_time: {date_time}")
    destination_path, local_path = assemble_journal_path(date_time, filename="Banner", extension=".jpg", no_timestamp = True)
    L.DEBUG(f"destination path generated: {destination_path}")
    
    if not location:
        locations = await locate.fetch_locations(date_time)
        if locations:
            location = locations[0]
    
    display_name = "Location: "
    if location:
        lat, lon = location.latitude, location.longitude
        override_location = await locate.find_override_locations(lat, lon)
        display_name += f"{override_location}, " if override_location else ""
        if location.display_name:
            display_name += f"{location.display_name}"

        else:
            display_name += f"{location.road}, " if location.road else ""
            display_name += f"the {location.neighbourhood} neighbourhood of " if location.neighbourhood else ""
            display_name += f"the {location.suburb} suburb of " if location.suburb else ""
            display_name += f"the {location.quarter} quarter, " if location.quarter else ""
            display_name += f"{location.city}, " if location.city else ""
            display_name += f"{location.state} " if location.state else ""
            display_name += f"{location.country} " if location.country else ""

        if display_name == "Location: ":
            geocoded_location = await locate.reverse_geocode(lat, lon)
            if geocoded_location.display_name or geocoded_location.city or geocoded_location.country:
                return await generate_banner(dt, geocoded_location, forecast, mood, other_context)
            else:
                L.WARN(f"Failed to get a useable location for purposes of generating a banner, but we'll generate one anyway.")
    
    if not forecast:
        forecast = "The weather forecast is: " + await update_dn_weather(date_time)

    sentiment = await sentiment_analysis(date_time)
    mood = sentiment if not mood else mood
    mood = f"Mood: {mood}" if mood else ""

    if mood and sentiment: mood = f"Mood: {mood}, {sentiment}"
    elif mood and not sentiment: mood = f"Mood: {mood}"
    elif sentiment and not mood: mood = f"Mood: {sentiment}"
    else: mood = ""

    events = await calendar.get_events(date_time, date_time)
    formatted_events = []
    for event in events:
        event_str = event.get('name')
        if event.get('location'):
            event_str += f" at {event.get('location')}"
        formatted_events.append(event_str)

    additional_info = ', '.join(formatted_events) if formatted_events else ''

    other_context = f"{other_context}, {additional_info}" if other_context else additional_info
    other_context = f"Additional information: {other_context}" if other_context else "" 

    prompt = "Generate an aesthetically appealing banner image for a daily note that helps to visualize the following scene information: "
    prompt += "\n".join([display_name, forecast, mood, other_context])
    L.DEBUG(f"Prompt: {prompt}")
    # sd.workflow(prompt: str, scene: str = None, size: str = None, style: str = "photorealistic", earlyurl: bool = False, destination_path: str = None):
    final_path = await sd.workflow(prompt, scene=OBSIDIAN_BANNER_SCENE, size="1080x512", style="romantic", destination_path=destination_path)
    if not str(local_path) in str(final_path):
        L.INFO(f"Apparent mismatch between local path, {local_path}, and final_path, {final_path}")

    jpg_embed = f"\"![[{local_path}]]\""
    await update_frontmatter(date_time, "banner", jpg_embed)

    return local_path


@note.get("/note/weather", response_class=JSONResponse)
async def note_weather_get(
    date: str = Query(default="0", description="Enter a date in YYYY-MM-DD format, otherwise it will default to today."),
    latlon: str = Query(default="45,-125"),
    refresh: bool = Query(default=False, description="Set to true to refresh the weather data")
):

    try:
        date_time = datetime.now() if date == "0" else await locate.localize_datetime(date)
        L.DEBUG(f"date: {date} .. date_time: {date_time}")
        content = await update_dn_weather(date_time) #, lat, lon)
        return JSONResponse(content={"forecast": content}, status_code=200)
    
    except HTTPException as e:
        return JSONResponse(content={"detail": str(e.detail)}, status_code=e.status_code)

    except Exception as e:
        L.ERR(f"Error in note_weather_get: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
                    

@note.post("/update/note/{date}")
async def post_update_daily_weather_and_calendar_and_timeslips(date: str) -> PlainTextResponse:
    date_time = await locate.localize_datetime(date)
    await update_dn_weather(date_time)
    await update_daily_note_events(date_time)
    await build_daily_timeslips(date_time)
    return f"[Refresh]({BASE_URL}/update/note/{date_time.strftime('%Y-%m-%d')}"

async def update_dn_weather(date_time: datetime):
    try:
        L.DEBUG(f"Updating weather for {date_time}")

        places = await locate.fetch_locations(date_time)
        place = places[0]
        lat = place.latitude
        lon = place.longitude

        city = await locate.find_override_locations(lat, lon)
        if city:
            L.INFO(f"Using override location: {city}")

        else:
            if place.city and place.city != "":
                city = place.city
                L.INFO(f"City in data: {city}")

            else:
                loc = await locate.reverse_geocode(lat, lon)
                L.DEBUG(f"loc: {loc}")
                city = loc.name
                city = city if city else loc.city
                city = city if city else loc.house_number + ' ' + loc.road
                
                L.INFO(f"City geocoded: {city}")

        # Assemble journal path
        absolute_path, relative_path = assemble_journal_path(date_time, filename="Weather", extension=".md", no_timestamp = True)
        L.DEBUG(f"Journal path: absolute_path={absolute_path}, relative_path={relative_path}")

        try:
            L.DEBUG(f"passing date_time {date_time}, {lat}/{lon} into fetch_and_store")
            day = await weather.get_weather(date_time, lat, lon)
            L.DEBUG(f"day information obtained from get_weather: {day}")
            if day:
                DailyWeather = day.get('DailyWeather')
                HourlyWeather = day.get('HourlyWeather')
                if DailyWeather:
                    L.DEBUG(f"Day: {DailyWeather}")
                    icon = DailyWeather.get('icon')
                    L.DEBUG(f"Icon: {icon}")
                    
                    weather_icon, admonition = get_icon_and_admonition(icon) if icon else (":LiSunMoon:", "ad-weather")
                    
                    temp = DailyWeather.get('feelslike')

                    if DailyWeather.get('tempmax', 0) > 85:
                        tempicon = ":RiTempHotLine:"
                    elif DailyWeather.get('tempmin', 65) < 32:
                        tempicon = ":LiThermometerSnowflake:"
                    else:
                        tempicon = ":LiThermometerSun:"
                    wind_direction = convert_degrees_to_cardinal(DailyWeather.get("winddir"))
                    wind_str = f":LiWind: {DailyWeather.get('windspeed')}mph {wind_direction}"
                    gust = DailyWeather.get('windgust', 0)

                    if gust and gust > DailyWeather.get('windspeed') * 1.2:
                        wind_str += f", gusts to {DailyWeather.get('windgust')}mph"

                    uvindex = DailyWeather.get('uvindex', 0)
                    uvwarn = f" - :LiRadiation: Caution! UVI today is {uvindex}! :LiRadiation:\n" if uvindex > 8 else ""

                    sunrise = DailyWeather.get('sunrise')
                    sunset = DailyWeather.get('sunset')
                    srise_str = sunrise.time().strftime("%H:%M")
                    sset_str = sunset.time().strftime("%H:%M")
                    

                    date_str = date_time.strftime("%Y-%m-%d")
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    detailed_forecast = (
                        f"---\n"
                        f"date: {date_str}\n"
                        f"latitude: {lat}"
                        f"longitude: {lon}"
                        f"tags:\n"
                        f" - weather\n"
                        f"updated: {now}\n"
                        f"---\n"
                        f"```{admonition}\n"
                        f"title: **{city}:** {temp}˚ F \n"
                        f" - {tempicon} {DailyWeather.get('tempmax')}˚ ↾⇃ {DailyWeather.get('tempmin')}˚ \n"
                        f" - {wind_str} \n"
                        f" - :LiSunrise: {srise_str} :LiOrbit: {sset_str} :LiSunset: \n"
                        f"{uvwarn} \n"
                        f"```\n\n"
                        f"```{admonition}\n"
                        f"title: {DailyWeather.get('description')} \n"
                    )
                    narrative = f"{city} on {date_str}: high of {DailyWeather.get('tempmax')}, low of {DailyWeather.get('tempmin')}. {DailyWeather.get('description')}"

                    if HourlyWeather:
                        times, condition_symbols, temps, winds = [], [], [], []

                        for hour in HourlyWeather:
                            if hour.get('datetime').strftime("%H:%M:%S") in HOURLY_COLUMNS_MAPPING.values():

                                times.append(format_hourly_time(hour)) 

                                condition_symbols.append(format_hourly_icon(hour, sunrise, sunset))

                                temps.append(format_hourly_temperature(hour))

                                winds.append(format_hourly_wind(hour))
                        
                        detailed_forecast += assemble_hourly_data_table(times, condition_symbols, temps, winds)
                        detailed_forecast += f"```\n\n"
                    
                    L.DEBUG(f"Detailed forecast: {detailed_forecast}.")

                    with open(absolute_path, 'w', encoding='utf-8') as note_file:
                        note_file.write(detailed_forecast)

                    L.DEBUG(f"Operation complete.")

                    return narrative
                else:
                    L.ERR(f"Failed to get DailyWeather from day: {day}")
            else:
                L.ERR(f"Failed to get day")
                raise HTTPException(status_code=500, detail="Failed to retrieve weather data")
            
        except HTTPException as e:
            L.ERR(f"HTTP error: {e}")
            L.ERR(traceback.format_exc())
            raise e
        
        except Exception as e:
            L.ERR(f"Error: {e}")
            L.ERR(traceback.format_exc())
            raise HTTPException(status_code=999, detail=f"Error: {e}")


    except ValueError as ve:
        L.ERR(f"Value error in update_dn_weather: {str(ve)}")
        L.ERR(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Value error: {str(ve)}")
    
    except Exception as e:
        L.ERR(f"Error in update_dn_weather: {str(e)}")
        L.ERR(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in update_dn_weather: {str(e)}")

def format_hourly_time(hour):
    try:
        hour_12 = convert_to_12_hour_format(hour.get("datetime"))
        return hour_12
    except Exception as e:
        L.ERR(f"Error in format_hourly_time: {str(e)}")
        L.ERR(traceback.format_exc())
        return ""

    
def format_hourly_icon(hour, sunrise, sunset):
    try:
        icon_str = hour.get('icon', '')
        icon, _ = get_icon_and_admonition(icon_str)
            
        precip = hour.get('precip', float(0.0))
        precip_prob = hour.get('precipprob', float(0.0))
        L.DEBUG(f"precip: {precip}, prob: {precip_prob}")
        
        sp_str = None

        if (precip > 0.05 and precip_prob > 25):
            precip_type = hour.get('preciptype', [''])
            sp_str = f"{str(precip)}mm" 

        if abs(hour.get('datetime') - sunrise) < timedelta(minutes=60):
            icon = ":LiSunrise:"
        elif abs(hour.get('datetime') - sunset) < timedelta(minutes=60):
            icon = ":LiSunset:"
        elif "thunder" in hour.get('icon'):
            icon += ":LiZap:"
        elif hour.get('uvindex') > 8:
            icon = ":LiRadiation:"
            sp_str = f"UV: {hour.get('uvindex', '')}"
        
        formatted = f"{icon}" if icon else ""
        formatted += f" {sp_str}" if sp_str else " "

        return formatted
    
    except Exception as e:
        L.ERR(f"Error in format_hourly_special: {str(e)}")
        L.ERR(traceback.format_exc())
        return ""

def format_hourly_temperature(hour):
    try:
        temp_str = f"{hour.get('temp', '')}˚ F"
        return temp_str
    except Exception as e:
        L.ERR(f"Error in format_hourly_temperature: {str(e)}")
        L.ERR(traceback.format_exc())
        return ""
    
def format_hourly_wind(hour):
    try:
        windspeed = hour.get('windspeed', '')
        winddir = convert_degrees_to_cardinal(float(hour.get('winddir', ''))) if windspeed else ""
        wind_str = f"{str(windspeed)}:LiWind: {winddir}"
        return wind_str
    except Exception as e:
        L.ERR(f"Error in format_hourly_wind: {str(e)}")
        L.ERR(traceback.format_exc())
        return ""

def assemble_hourly_data_table(times, condition_symbols, temps, winds):
    table_rows = [times, condition_symbols, temps, winds]
    table = "| " + " | ".join(times) + " |\n"
    table += "| " + " | ".join([':----:' for _ in times]) + " |\n"
    for row in table_rows[1:]:
        table += "| " + " | ".join(row) + " |\n"
    return table + "\n\n"


def get_icon_and_admonition(icon_str) -> Tuple:
    L.DEBUG(f"Received request for emoji {icon_str}")
    if icon_str.startswith(":") and icon_str.endswith(":"):
        return icon_str
    
    icon_str = icon_str.lower()

    if icon_str == "clear-day":
        icon = ":LiSun:"
        ad = "ad-sun"
    elif icon_str == "clear-night":
        icon = ":LiMoon:"
        ad = "ad-sun"
    elif icon_str == "partly-cloudy-day":
        icon = ":LiCloudSun:"
        ad = "ad-partly"
    elif icon_str == "partly-cloudy-night":
        icon = ":LiCloudMoon:"
        ad = "ad-partly"
    elif icon_str == "cloudy":
        icon = ":LiCloud:"
        ad = "ad-cloud"
    elif icon_str == "rain":
        icon = ":LiCloudRain:"
        ad = "ad-rain"
    elif icon_str == "snow":
        icon = ":LiSnowflake:"
        ad = "ad-snow"
    elif icon_str == "snow-showers-day":
        icon = ":LiCloudSnow:"
        ad = "ad-snow"
    elif icon_str == "snow-showers-night":
        icon = ":LiCloudSnow:"
        ad = "ad-snow"
    elif icon_str == "showers-day":
        icon = ":LiCloudSunRain:"
        ad = "ad-rain"
    elif icon_str == "showers-night":
        icon = ":LiCloudMoonRain:"
        ad = "ad-rain"
    elif icon_str == "fog":
        icon = ":LiCloudFog:"
        ad = "ad-fog"
    elif icon_str == "wind":
        icon = ":LiWind:"
        ad = "ad-wind"
    elif icon_str == "thunder-rain":
        icon = ":LiCloudLightning:"
        ad = "ad-thunder"
    elif icon_str == "thunder-showers-day":
        icon = ":LiCloudLightning:"
        ad = "ad-thunder"
    elif icon_str == "thunder-showers-night":
        icon = ":LiCloudLightning:"
        ad = "ad-thunder"
    else:
        icon = ":LiHelpCircle:"
        ad = "ad-weather" 
    
    return icon, ad

def get_weather_emoji(weather_condition):
    condition = weather_condition.lower()
    if 'clear' in condition or 'sunny' in condition:
        return "☀️"
    elif 'cloud' in condition or 'overcast' in condition:
        return "☁️"
    elif 'rain' in condition:
        return "🌧️"
    elif 'snow' in condition:
        return "❄️"
    elif 'thunder' in condition or 'storm' in condition:
        return "⛈️"
    elif 'fog' in condition or 'mist' in condition:
        return "🌫️"
    elif 'wind' in condition:
        return "💨"
    elif 'hail' in condition:
        return "🌨️"
    elif 'sleet' in condition:
        return "🌧️"
    elif 'partly' in condition:
        return "⛅"
    else:
        return "🌡️"  # Default emoji for unclassified weather
    

### CALENDAR ###


async def format_events_as_markdown(event_data: Dict[str, Union[str, List[Dict[str, str]]]]) -> str:
    def remove_characters(s: str) -> str:
        s = s.replace('---', '\n')
        s = s.strip('\n')
        s = re.sub(r'^_+|_+$', '', s)
        return s
    
    date_str = event_data["date"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    events_markdown = []

    event_data["events"] = sorted(event_data["events"], key=lambda event: (not event['all_day'], datetime.strptime(event['start'], "%H:%M")), reverse=False)

    total_events = len(event_data["events"])
    event_markdown = f"```ad-events"
    for event in event_data["events"]:
        L.DEBUG(f"event busy status: {event['busy']}; all day status: {event['all_day']}")
        if not event['name'].startswith('TC '):
            url = f"hook://ical/eventID={event['uid']}calendarID=17"
            if event['url']:
                url = event['url']

            elif event['location'] and event['location'].startswith(('http', 'www.')):
                url = event['location']
                event['url'] = url
                event['location'] = ''

            event_name = event['name'][:80]
            markdown_name = f"[{event_name}]({url})"

            if (event['all_day']) or (event['start'] == event['end'] == "00:00") or (datetime.combine(dt_date.min, datetime.strptime(event['end'], "%H:%M").time()) - datetime.combine(dt_date.min, datetime.strptime(event['start'], "%H:%M").time()) >= timedelta(hours=23, minutes=59)):
                event_markdown += f"\n - [ ] **{markdown_name}** (All day)"

            else:
                event_markdown += f"\n - [ ] **{event['start']}—{event['end']}** {markdown_name}"
                
            if event['attendees']:
                attendee_list = []
                for att in event['attendees']:
                    attendee_list.append(f'[{att["name"]}](mailto:{att["email"]})')
                attendees_markdown = ', '.join(attendee_list)
                event_markdown += f"\n     * **Attendees:** {attendees_markdown}"

            if event['location'] and not event['location'].startswith(('http', 'www.')):
                location = event['location']
                location = remove_characters(location)
                location = remove_characters(location)
                event_markdown += f"\n     * **Location:** {location}"

            if event['description']:
                description = event['description']
           #     # This was intended to clean up the descriptions of Zoom and Teams events but is presently broken; should be an easy fix.
           #     if 'Zoom Meeting' in description:
           #         description_parts = description.split('---')
           #         if len(description_parts) > 2:
           #             description = description_parts[1].strip()
           #     if 'Microsoft Teams' in description:
           #         description_parts = description.split('---')
           #         if len(description_parts) > 2:
           #             event_data['description'] = description_parts[1].strip()
           #     description = remove_characters(description)
           #     description = remove_characters(description)
                if len(description) > 150:
                    description = await llm.summarize_text(description, length_override=150)

                event_markdown += f"\n     * {description}"
            event_markdown += f"\n "
   
    event_markdown += "\n```\n"
    events_markdown.append(event_markdown)
    
    header = (
        "---\n"
        f"date: {date_str}\n"
        "tags:\n"
        " - events\n"
        f"updated: {now}\n"
        "---\n"
    )
    
    detailed_events = (
        f"{header}"
        f"{''.join(events_markdown)}"
    )
    return detailed_events

@note.get("/note/events", response_class=PlainTextResponse)
async def note_events_endpoint(date: str = Query(None)):
        
    date_time = await locate.localize_datetime(date) if date else datetime.now(TZ)
    response = await update_daily_note_events(date_time)
    return PlainTextResponse(content=response, status_code=200)

async def update_daily_note_events(date_time: datetime):
    L.DEBUG(f"Looking up events on date: {date_time.strftime('%Y-%m-%d')}")
    try:    
        events = await calendar.get_events(date_time, date_time)
        L.DEBUG(f"Raw events: {events}")
        event_data = {
            "date": date_time.strftime('%Y-%m-%d'),
            "events": events
        }
        events_markdown = await format_events_as_markdown(event_data)
        L.DEBUG(f"Markdown events: {events_markdown}")
        absolute_path, _ = assemble_journal_path(date_time, filename="Events", extension=".md", no_timestamp = True)
        L.DEBUG(f"Writing events to file: {absolute_path}")

        with open(absolute_path, 'w', encoding='utf-8') as note_file:
            note_file.write(events_markdown)

        return events_markdown

    except Exception as e:
        L.ERR(f"Error processing events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

