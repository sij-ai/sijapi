# routers/news.py
import os
import uuid
import asyncio
import shutil
import requests
import mimetypes
from io import BytesIO
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from datetime import datetime as dt_datetime, timedelta
from typing import Optional

import aiohttp
import aiofiles
import newspaper
import trafilatura
from readability import Document
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, Form, HTTPException, Response, Query, Path as FastAPIPath
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pathlib import Path
from sijapi import API, L, Dir, News, OBSIDIAN_VAULT_DIR, OBSIDIAN_RESOURCES_DIR, OBSIDIAN_BANNER_SCENE, DEFAULT_11L_VOICE, DEFAULT_VOICE, GEO
from sijapi.utilities import sanitize_filename, assemble_journal_path, assemble_archive_path
from sijapi.routers import llm, tts, asr, loc

from newspaper import Article

news = APIRouter()
logger = L.get_module_logger("news")

async def download_and_save_article(article, site_name, earliest_date, bg_tasks: BackgroundTasks, tts_mode: str = "summary", voice: str = DEFAULT_11L_VOICE):
    try:
        url = article.url
        source = trafilatura.fetch_url(url)
        
        if source is None:
            # Fallback to newspaper3k if trafilatura fails
            article.download()
            article.parse()
            traf = None
        else:
            traf = trafilatura.extract_metadata(filecontent=source, default_url=url)
            article.download()
            article.parse()

        # Update article properties, preferring trafilatura data when available
        article.title = traf.title if traf and traf.title else article.title or url
        article.authors = traf.author if traf and traf.author else article.authors or []
        article.publish_date = traf.date if traf and traf.date else article.publish_date
        try:
            article.publish_date = await loc.dt(article.publish_date, "UTC")
        except:
            logger.debug(f"Failed to localize {article.publish_date}")
            article.publish_date = await loc.dt(dt_datetime.now(), "UTC")
        article.meta_description = traf.description if traf and traf.description else article.meta_description
        article.text = trafilatura.extract(source, output_format="markdown", include_comments=False) if source else article.text
        article.top_image = traf.image if traf and traf.image else article.top_image
        article.source_url = traf.sitename if traf and traf.sitename else urlparse(url).netloc.replace('www.', '').title()
        article.meta_keywords = traf.categories or traf.tags if traf else article.meta_keywords or []
        article.meta_keywords = article.meta_keywords if isinstance(article.meta_keywords, list) else [article.meta_keywords]

        if not is_article_within_date_range(article, earliest_date):
            return False


        timestamp = dt_datetime.now().strftime('%b %d, %Y at %H:%M')
        readable_title = sanitize_filename(article.title or timestamp)
        markdown_filename, relative_path = assemble_journal_path(dt_datetime.now(), subdir="Articles", filename=readable_title, extension=".md")

        summary = await llm.summarize_text(article.text, "Summarize the provided text. Respond with the summary and nothing else. Do not otherwise acknowledge the request. Just provide the requested summary.")
        summary = summary.replace('\n', ' ')  # Remove line breaks

        if tts_mode == "full" or tts_mode == "content":
            tts_text = article.text
        elif tts_mode == "summary" or tts_mode == "excerpt":
            tts_text = summary
        else:
            tts_text = None

        banner_markdown = ''
        try:
            banner_url = article.top_image
            if banner_url:
                banner_image = download_file(banner_url, Path(OBSIDIAN_VAULT_DIR / OBSIDIAN_RESOURCES_DIR / f"{dt_datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"))
                if banner_image:
                    banner_markdown = f"![[{OBSIDIAN_RESOURCES_DIR}/{banner_image}]]"
        except Exception as e:
            logger.error(f"No image found in article")

        
        authors = ', '.join(['[[{}]]'.format(author.strip()) for author in article.authors if author.strip()])
        if not authors:
            authors = '[[Unknown Author]]'

        frontmatter = f"""---
title: {readable_title}
authors: {authors}
published: {article.publish_date}
added: {timestamp}
banner: "{banner_markdown}"
tags:
"""
        frontmatter += '\n'.join(f" - {tag}" for tag in article.meta_keywords)
        frontmatter += '\n---\n'

        body = f"# {readable_title}\n\n"
        if tts_text:
            audio_filename = f"{article.publish_date.strftime('%Y-%m-%d')} {readable_title}"
            try:
                audio_path = await tts.generate_speech(
                    bg_tasks=bg_tasks, 
                    text=tts_text, 
                    voice=voice, 
                    model="xtts2", 
                    podcast=True, 
                    title=audio_filename,
                    output_dir=Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR
                )
                if isinstance(audio_path, Path):
                    audio_ext = audio_path.suffix
                    obsidian_link = f"![[{audio_path.name}]]"
                    body += f"{obsidian_link}\n\n"
                else:
                    logger.warning(f"Unexpected audio_path type: {type(audio_path)}. Value: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to generate TTS for {audio_filename}. Error: {str(e)}")
                logger.error(f"TTS error details - voice: {voice}, model: eleven_turbo_v2, podcast: True")
                logger.error(f"Output directory: {Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR}")

        body += f"by {authors} in {article.source_url}\n\n"
        body += f"> [!summary]+\n"
        body += f"> {summary}\n\n"
        body += article.text

        markdown_content = frontmatter + body

        with open(markdown_filename, 'w') as md_file:
            md_file.write(markdown_content)

        logger.info(f"Successfully saved to {markdown_filename}")
        add_to_daily_note(relative_path)
        print(f"Saved article: {relative_path}")
        return True


    except Exception as e:
        logger.error(f"Error processing article from {article.url}: {str(e)}")
        return False

# You'll need to update your is_article_within_date_range function:
def is_article_within_date_range(article, earliest_date):
    return article.publish_date is not None and article.publish_date.date() >= earliest_date

async def process_news_site(site, bg_tasks: BackgroundTasks):
    print(f"Downloading articles from {site.name}...")
    
    earliest_date = dt_datetime.now().date() - timedelta(days=site.days_back)
    
    try:
        news_source = newspaper.build(site.url, memoize_articles=False)
        
        tasks = []
        for article in news_source.articles[:site.max_articles]:
            task = asyncio.create_task(download_and_save_article(
                article, 
                site.name, 
                earliest_date, 
                bg_tasks, 
                tts_mode=site.tts if hasattr(site, 'tts') else "off",
                voice=site.voice if hasattr(site, 'voice') else DEFAULT_11L_VOICE
            ))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        articles_downloaded = sum(results)
        
        print(f"Downloaded {articles_downloaded} articles from {site.name}")
    except Exception as e:
        print(f"Error processing {site.name}: {str(e)}")

# Update your news_refresh_endpoint function:
@news.get("/news/refresh")
async def news_refresh_endpoint(bg_tasks: BackgroundTasks):
    tasks = [process_news_site(site, bg_tasks) for site in News.sites]
    await asyncio.gather(*tasks)
    return "OK"


async def generate_path(article, site_name):
    publish_date = await loc.dt(article.publish_date, 'UTC') if article.publish_date else await loc.dt(dt_datetime.now(), 'UTC')
    title_slug = "".join(c if c.isalnum() else "_" for c in article.title)
    filename = f"{site_name} - {title_slug[:50]}.md"
    absolute_path, relative_path = assemble_journal_path(publish_date, 'Articles', filename, extension='.md', no_timestamp=True)
    return absolute_path, relative_path

async def save_article_to_file(content, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(output_path, 'w', encoding='utf-8') as file:
        await file.write(content)



### CLIPPER ###
@news.post("/clip")
async def clip_post(
    bg_tasks: BackgroundTasks,
    url: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    tts: str = Form('summary'),
    voice: str = Form(DEFAULT_VOICE),
    encoding: str = Form('utf-8')
):
    markdown_filename = await process_article(bg_tasks, url, title, encoding, source, tts, voice)
    return {"message": "Clip saved successfully", "markdown_filename": markdown_filename}

@news.post("/archive")
async def archive_post(
    url: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    encoding: str = Form('utf-8')
):
    markdown_filename = await process_archive(url, title, encoding, source)
    return {"message": "Clip saved successfully", "markdown_filename": markdown_filename}

@news.get("/clip")
async def clip_get(
    bg_tasks: BackgroundTasks,
    url: str,
    tts: str = Query('summary'),
    voice: str = Query(DEFAULT_VOICE)
):
    parsed_content = await parse_article(url)
    markdown_filename = await process_article2(bg_tasks, parsed_content, tts, voice)
    return {"message": "Clip saved successfully", "markdown_filename": markdown_filename}

@news.post("/note/add")
async def note_add_endpoint(file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), source: Optional[str] = Form(None), bg_tasks: BackgroundTasks = None):
    logger.debug(f"Received request on /note/add...")
    if not file and not text:
        logger.warning(f"... without any file or text!")
        raise HTTPException(status_code=400, detail="Either text or a file must be provided")
    else:
        result = await process_for_daily_note(file, text, source, bg_tasks)
        logger.info(f"Result on /note/add: {result}")
        return JSONResponse(result, status_code=204)

async def process_for_daily_note(file: Optional[UploadFile] = File(None), text: Optional[str] = None, source: Optional[str] = None, bg_tasks: BackgroundTasks = None):
    now = dt_datetime.now()
    transcription_entry = ""
    file_entry = ""
    if file:
        logger.debug("File received...")
        file_content = await file.read()
        audio_io = BytesIO(file_content)
        
        # Improve error handling for file type guessing
        guessed_type = mimetypes.guess_type(file.filename)
        file_type = guessed_type[0] if guessed_type[0] else "application/octet-stream"
        
        logger.debug(f"Processing as {file_type}...")
        
        # Extract the main type (e.g., 'audio', 'image', 'video')
        main_type = file_type.split('/')[0]
        subdir = main_type.title() if main_type else "Documents"
        
        absolute_path, relative_path = assemble_journal_path(now, subdir=subdir, filename=file.filename)
        logger.debug(f"Destination path: {absolute_path}")
        
        with open(absolute_path, 'wb') as f:
            f.write(file_content)
        logger.debug(f"Processing {f.name}...")
        
        if main_type == 'audio':
            transcription = await asr.transcribe_audio(file_path=absolute_path, params=asr.TranscribeParams(model="small-en", language="en", threads=6))
            file_entry = f"![[{relative_path}]]"
        elif main_type == 'image':
            file_entry = f"![[{relative_path}]]"
        else:
            file_entry = f"[Source]({relative_path})"
    
    text_entry = text if text else ""
    logger.debug(f"transcription: {transcription_entry}\nfile_entry: {file_entry}\ntext_entry: {text_entry}")
    return await add_to_daily_note(transcription_entry, file_entry, text_entry, now)

async def add_to_daily_note(transcription: str = None, file_link: str = None, additional_text: str = None, date_time: dt_datetime = None):
    date_time = date_time or dt_datetime.now()
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



async def process_document(
    bg_tasks: BackgroundTasks,
    document: File,
    title: Optional[str] = None,
    tts_mode: str = "summary",
    voice: str = DEFAULT_VOICE
):
    timestamp = dt_datetime.now().strftime('%b %d, %Y at %H:%M')

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
                datetime_str = dt_datetime.now().strftime("%Y%m%d%H%M%S")
                audio_filename = f"{datetime_str} {readable_title}"
                audio_path = await tts.generate_speech(
                    bg_tasks=bg_tasks,
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
                logger.error(f"Failed in the TTS portion of clipping: {e}")

        body += f"> [!summary]+\n"
        body += f"> {summary}\n\n"
        body += parsed_content
        markdown_content = frontmatter + body

        markdown_filename = f"{readable_title}.md"
        encoding = 'utf-8'

        with open(markdown_filename, 'w', encoding=encoding) as md_file:
            md_file.write(markdown_content)

        logger.info(f"Successfully saved to {markdown_filename}")

        return markdown_filename

    except Exception as e:
        logger.error(f"Failed to clip: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



async def process_article2(
    bg_tasks: BackgroundTasks,
    parsed_content: Article,
    tts_mode: str = "summary", 
    voice: str = DEFAULT_11L_VOICE
):
    timestamp = dt_datetime.now().strftime('%b %d, %Y at %H:%M')

    readable_title = sanitize_filename(parsed_content.title or timestamp)
    markdown_filename, relative_path = assemble_journal_path(dt_datetime.now(), subdir="Articles", filename=readable_title, extension=".md")

    try:
        summary = await llm.summarize_text(parsed_content.clean_doc, "Summarize the provided text. Respond with the summary and nothing else. Do not otherwise acknowledge the request. Just provide the requested summary.")
        summary = summary.replace('\n', ' ')  # Remove line breaks

        if tts_mode == "full" or tts_mode == "content":
            tts_text = parsed_content.clean_doc
        elif tts_mode == "summary" or tts_mode == "excerpt":
            tts_text = summary
        else:
            tts_text = None

        banner_markdown = ''
        try:
            banner_url = parsed_content.top_image
            if banner_url != '':
                banner_image = download_file(banner_url, Path(OBSIDIAN_VAULT_DIR / OBSIDIAN_RESOURCES_DIR))
                if banner_image:
                    banner_markdown = f"![[{OBSIDIAN_RESOURCES_DIR}/{banner_image}]]"
                
        except Exception as e:
            logger.error(f"No image found in article")

        authors = ', '.join('[[{}]]'.format(author) for author in parsed_content.authors)
        published_date = parsed_content.publish_date
        frontmatter = f"""---
title: {readable_title}
authors: {authors}
published: {published_date}
added: {timestamp}
banner: "{banner_markdown}"
tags:

"""
        frontmatter += '\n'.join(f" - {tag}" for tag in parsed_content.tags)
        frontmatter += '\n---\n'

        body = f"# {readable_title}\n\n"
        if tts_text:
            audio_filename = f"{published_date} {readable_title}"
            try:
                audio_path = await tts.generate_speech(bg_tasks=bg_tasks, text=tts_text, voice=voice, model="eleven_turbo_v2", podcast=True, title=audio_filename,
                output_dir=Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR)
                audio_ext = Path(audio_path).suffix
                obsidian_link = f"![[{OBSIDIAN_RESOURCES_DIR}/{audio_filename}{audio_ext}]]"
                body += f"{obsidian_link}\n\n"
            except Exception as e:
                logger.error(f"Failed to generate TTS for np3k. {e}")

        try:
            body += f"by {authors} in {parsed_content.canonical_link}" # update with method for getting the newspaper name
            body += f"> [!summary]+\n"
            body += f"> {summary}\n\n"
            body += parsed_content["content"]
            markdown_content = frontmatter + body

        except Exception as e:
            logger.error(f"Failed to combine elements of article markdown.")

        try:
            with open(markdown_filename, 'w') as md_file:
                md_file.write(markdown_content)

            logger.info(f"Successfully saved to {markdown_filename}")
            add_to_daily_note
            return markdown_filename
        
        except Exception as e:
            logger.error(f"Failed to write markdown file")
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"Failed to clip: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_article(
    bg_tasks: BackgroundTasks,
    url: str,
    title: Optional[str] = None,
    encoding: str = 'utf-8',
    source: Optional[str] = None,
    tts_mode: str = "summary", 
    voice: str = DEFAULT_11L_VOICE
):

    timestamp = dt_datetime.now().strftime('%b %d, %Y at %H:%M')

    parsed_content = await parse_article(url, source)
    if parsed_content is None:
        return {"error": "Failed to retrieve content"}

    readable_title = sanitize_filename(title or parsed_content.get("title") or timestamp)
    markdown_filename, relative_path = assemble_journal_path(dt_datetime.now(), subdir="Articles", filename=readable_title, extension=".md")

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
            logger.error(f"No image found in article")

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
            datetime_str = dt_datetime.now().strftime("%Y%m%d%H%M%S")
            audio_filename = f"{datetime_str} {readable_title}"
            try:
                audio_path = await tts.generate_speech(bg_tasks=bg_tasks, text=tts_text, voice=voice, model="eleven_turbo_v2", podcast=True, title=audio_filename,
                output_dir=Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR)
                audio_ext = Path(audio_path).suffix
                obsidian_link = f"![[{OBSIDIAN_RESOURCES_DIR}/{audio_filename}{audio_ext}]]"
                body += f"{obsidian_link}\n\n"
            except Exception as e:
                logger.error(f"Failed to generate TTS for np3k. {e}")

        try:
            body += f"by {authors} in [{parsed_content.get('domain', urlparse(url).netloc.replace('www.', ''))}]({url}).\n\n"
            body += f"> [!summary]+\n"
            body += f"> {summary}\n\n"
            body += parsed_content["content"]
            markdown_content = frontmatter + body

        except Exception as e:
            logger.error(f"Failed to combine elements of article markdown.")

        try:
            with open(markdown_filename, 'w', encoding=encoding) as md_file:
                md_file.write(markdown_content)

            logger.info(f"Successfully saved to {markdown_filename}")
            add_to_daily_note
            return markdown_filename
        
        except Exception as e:
            logger.error(f"Failed to write markdown file")
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"Failed to clip {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



async def parse_article(url: str, source: Optional[str] = None) -> Article:
    source = source if source else trafilatura.fetch_url(url)
    traf = trafilatura.extract_metadata(filecontent=source, default_url=url)

    # Create and parse the newspaper3k Article
    article = Article(url)
    article.set_html(source)
    article.parse()

    logger.info(f"Parsed {article.title}")

    # Update or set properties based on trafilatura and additional processing
    article.title = article.title or traf.title or url
    article.authors = article.authors or (traf.author if isinstance(traf.author, list) else [traf.author])
    
    article.publish_date = article.publish_date or traf.date
    try:
        article.publish_date = await loc.dt(article.publish_date, "UTC")
    except:
        logger.debug(f"Failed to localize {article.publish_date}")
        article.publish_date = await loc.dt(dt_datetime.now(), "UTC")

    article.meta_description = article.meta_description or traf.description
    article.text = trafilatura.extract(source, output_format="markdown", include_comments=False) or article.text
    article.top_image = article.top_image or traf.image
    article.source_url = traf.sitename or urlparse(url).netloc.replace('www.', '').title()
    article.meta_keywords = article.meta_keywords or traf.categories or traf.tags
    article.meta_keywords = article.meta_keywords if isinstance(article.meta_keywords, list) else [article.meta_keywords]

    # Set additional data in the additional_data dictionary
    article.additional_data = {
        'excerpt': article.meta_description,
        'domain': article.source_url,
        'tags': article.meta_keywords,
        'content': article.text  # Store the markdown content here
    }

    return article



async def html_to_markdown(url: str = None, source: str = None) -> Optional[str]:
    if source:
        html_content = source
    elif url:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html_content = await response.text()
    else:
        logger.error(f"Unable to convert nothing to markdown.")
        return None

    # Use readability to extract the main content
    doc = Document(html_content)
    cleaned_html = doc.summary()

    # Parse the cleaned HTML with BeautifulSoup for any additional processing
    soup = BeautifulSoup(cleaned_html, 'html.parser')

    # Remove any remaining unwanted elements
    for element in soup(['script', 'style']):
        element.decompose()

    # Convert to markdown
    markdown_content = md(str(soup), heading_style="ATX")

    return markdown_content


async def process_archive(
    url: str,
    title: Optional[str] = None,
    encoding: str = 'utf-8',
    source: Optional[str] = None,
) -> Path:
    timestamp = dt_datetime.now().strftime('%b %d, %Y at %H:%M')
    readable_title = title if title else f"{url} - {timestamp}"
    
    content = await html_to_markdown(url, source)
    if content is None:
        raise HTTPException(status_code=400, detail="Failed to convert content to markdown")
    
    markdown_path, relative_path = assemble_archive_path(readable_title, ".md")
    
    markdown_content = f"---\n"
    markdown_content += f"title: {readable_title}\n"
    markdown_content += f"added: {timestamp}\n"
    markdown_content += f"url: {url}"
    markdown_content += f"date: {dt_datetime.now().strftime('%Y-%m-%d')}"
    markdown_content += f"---\n\n"
    markdown_content += f"# {readable_title}\n\n"
    markdown_content += f"Clipped from [{url}]({url}) on {timestamp}"
    markdown_content += content
    
    try:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        with open(markdown_path, 'w', encoding=encoding) as md_file:
            md_file.write(markdown_content)
        logger.debug(f"Successfully saved to {markdown_path}")
        return markdown_path
    except Exception as e:
        logger.warning(f"Failed to write markdown file: {str(e)}")
        return None

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
                logger.error(f"Failed to download image: {url}, invalid content type: {response.headers.get('Content-Type')}")
                return None
        else:
            logger.error(f"Failed to download image: {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Failed to download image: {url}, error: {str(e)}")
        return None
    return filename

def copy_file(local_path, folder):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.basename(local_path)
    destination_path = os.path.join(folder, filename)
    shutil.copy(local_path, destination_path)
    return filename


async def save_file(file: UploadFile, folder: Path) -> Path:
    file_path = folder / f"{dt_datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    return file_path
