# routers/news.py
import os
import uuid
import asyncio
import shutil
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime as dt_datetime, timedelta
from typing import Optional
import aiohttp
import aiofiles
import newspaper
import trafilatura
from newspaper import Article
from readability import Document
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import APIRouter, BackgroundTasks, UploadFile, Form, HTTPException, Query, Path as FastAPIPath
from pathlib import Path
from sijapi import L, News, OBSIDIAN_VAULT_DIR, OBSIDIAN_RESOURCES_DIR, DEFAULT_11L_VOICE, DEFAULT_VOICE
from sijapi.utilities import sanitize_filename, assemble_journal_path, assemble_archive_path
from sijapi.routers import gis, llm, tts, note


news = APIRouter()
logger = L.get_module_logger("news")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

async def process_and_save_article(
    bg_tasks: BackgroundTasks,
    url: str,
    title: Optional[str] = None,
    tts_mode: str = "summary",
    voice: str = DEFAULT_VOICE,
    site_name: Optional[str] = None
) -> str:
    try:
        # Fetch and parse article
        article = await fetch_and_parse_article(url)
        
        # Generate title and file paths
        title = sanitize_filename(title or article.title or f"Untitled - {dt_datetime.now().strftime('%Y-%m-%d')}")
        markdown_filename, relative_path = assemble_journal_path(dt_datetime.now(), subdir="Articles", filename=title, extension=".md")

        # Generate summary
        summary = await generate_summary(article.text)

        # Handle TTS
        audio_link = await handle_tts(bg_tasks, article, title, tts_mode, voice, summary)

        # Generate markdown content
        markdown_content = generate_markdown_content(article, title, summary, audio_link, site_name)

        # Save markdown file
        await save_markdown_file(markdown_filename, markdown_content)

        # Add to daily note
        await note.add_to_daily_note(relative_path)

        return f"Successfully saved: {relative_path}"

    except Exception as e:
        err(f"Failed to process article {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def fetch_and_parse_article(url: str) -> Article:
    source = trafilatura.fetch_url(url)
    traf = trafilatura.extract_metadata(filecontent=source, default_url=url)

    article = Article(url)
    article.set_html(source)
    article.parse()

    # Update article properties with trafilatura data
    article.title = article.title or traf.title or url
    article.authors = article.authors or (traf.author if isinstance(traf.author, list) else [traf.author])
    article.publish_date = await gis.dt(article.publish_date or traf.date or dt_datetime.now(), "UTC")
    article.text = trafilatura.extract(source, output_format="markdown", include_comments=False) or article.text
    article.top_image = article.top_image or traf.image
    article.source_url = traf.sitename or urlparse(url).netloc.replace('www.', '').title()
    article.meta_keywords = list(set(article.meta_keywords or traf.categories or traf.tags or []))

    return article

def is_article_within_date_range(article: Article, days_back: int) -> bool:
    earliest_date = dt_datetime.now().date() - timedelta(days=days_back)
    return article.publish_date.date() >= earliest_date

async def generate_summary(text: str) -> str:
    summary = await llm.summarize_text(text, "Summarize the provided text. Respond with the summary and nothing else.")
    return summary.replace('\n', ' ')

async def handle_tts(bg_tasks: BackgroundTasks, article: Article, title: str, tts_mode: str, voice: str, summary: str) -> Optional[str]:
    if tts_mode in ["full", "content"]:
        tts_text = article.text
    elif tts_mode in ["summary", "excerpt"]:
        tts_text = summary
    else:
        return None

    audio_filename = f"{article.publish_date.strftime('%Y-%m-%d')} {title}"
    try:
        audio_path = await tts.generate_speech(
            bg_tasks=bg_tasks,
            text=tts_text,
            voice=voice,
            model="xtts",
            podcast=True,
            title=audio_filename,
            output_dir=Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR
        )
        return f"![[{Path(audio_path).name}]]"
    except HTTPException as e:
        err(f"Failed to generate TTS: {str(e)}")
        return None


def generate_markdown_content(article: Article, title: str, summary: str, audio_link: Optional[str], site_name: Optional[str] = None) -> str:
    frontmatter = f"""---
title: {title}
authors: {', '.join(f'[[{author}]]' for author in article.authors)}
published: {article.publish_date}
added: {dt_datetime.now().strftime('%b %d, %Y at %H:%M')}
banner: "{get_banner_markdown(article.top_image)}"
tags:
{chr(10).join(f' - {tag}' for tag in article.meta_keywords)}
"""
    if site_name:
        frontmatter += f"site: {site_name}\n"
    frontmatter += "---\n\n"

    body = f"# {title}\n\n"
    if audio_link:
        body += f"{audio_link}\n\n"
    body += f"by {', '.join(article.authors)} in [{article.source_url}]({article.url})\n\n"
    body += f"> [!summary]+\n> {summary}\n\n"
    body += article.text

    return frontmatter + body


def get_banner_markdown(image_url: str) -> str:
    if not image_url:
        return ''
    try:
        banner_image = download_file(image_url, Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR)
        return f"![[{OBSIDIAN_RESOURCES_DIR}/{banner_image}]]" if banner_image else ''
    except Exception as e:
        err(f"Failed to download banner image: {str(e)}")
        return ''

async def save_markdown_file(filename: str, content: str):
    async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
        await f.write(content)


async def download_and_save_article(article, site_name, earliest_date, bg_tasks: BackgroundTasks, tts_mode: str = "off", voice: str = DEFAULT_11L_VOICE):
    try:
        url = article.url
        parsed_article = await fetch_and_parse_article(url)
        
        if not is_article_within_date_range(parsed_article, earliest_date):
            return False

        return await process_and_save_article(bg_tasks, url, None, tts_mode, voice, site_name=site_name)

    except Exception as e:
        err(f"Error processing article from {article.url}: {str(e)}")
        return False

async def process_news_site(site, bg_tasks: BackgroundTasks):
    info(f"Downloading articles from {site.name}...")
    
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
        
        info(f"Downloaded {articles_downloaded} articles from {site.name}")
    except Exception as e:
        err(f"Error processing {site.name}: {str(e)}")


@news.get("/news/refresh")
async def news_refresh_endpoint(bg_tasks: BackgroundTasks):
    tasks = [process_news_site(site, bg_tasks) for site in News.sites]
    await asyncio.gather(*tasks)
    return "OK"


async def generate_path(article, site_name):
    publish_date = await gis.dt(article.publish_date, 'UTC') if article.publish_date else await gis.dt(dt_datetime.now(), 'UTC')
    title_slug = "".join(c if c.isalnum() else "_" for c in article.title)
    filename = f"{site_name} - {title_slug[:50]}.md"
    absolute_path, relative_path = assemble_journal_path(publish_date, 'Articles', filename, extension='.md', no_timestamp=True)
    return absolute_path, relative_path


async def save_article_to_file(content, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(output_path, 'w', encoding='utf-8') as file:
        await file.write(content)


@news.post("/clip")
async def clip_post(
    bg_tasks: BackgroundTasks,
    url: str = Form(...),
    title: Optional[str] = Form(None),
    tts: str = Form('summary'),
    voice: str = Form(DEFAULT_VOICE),
):
    result = await process_and_save_article(bg_tasks, url, title, tts, voice)
    return {"message": "Clip saved successfully", "result": result}

@news.get("/clip")
async def clip_get(
    bg_tasks: BackgroundTasks,
    url: str,
    tts: str = Query('summary'),
    voice: str = Query(DEFAULT_VOICE)
):
    result = await process_and_save_article(bg_tasks, url, None, tts, voice)
    return {"message": "Clip saved successfully", "result": result}



@news.post("/archive")
async def archive_post(
    url: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    encoding: str = Form('utf-8')
):
    markdown_filename = await process_archive(url, title, encoding, source)
    return {"message": "Clip saved successfully", "markdown_filename": markdown_filename}


async def parse_article(url: str, source: Optional[str] = None) -> Article:
    source = source if source else trafilatura.fetch_url(url)
    traf = trafilatura.extract_metadata(filecontent=source, default_url=url)

    # Create and parse the newspaper3k Article
    article = Article(url)
    article.set_html(source)
    article.parse()

    info(f"Parsed {article.title}")

    # Update or set properties based on trafilatura and additional processing
    article.title = article.title or traf.title or url
    article.authors = article.authors or (traf.author if isinstance(traf.author, list) else [traf.author])
    
    article.publish_date = article.publish_date or traf.date
    try:
        article.publish_date = await gis.dt(article.publish_date, "UTC")
    except:
        debug(f"Failed to localize {article.publish_date}")
        article.publish_date = await gis.dt(dt_datetime.now(), "UTC")

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
        err(f"Unable to convert nothing to markdown.")
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
        debug(f"Successfully saved to {markdown_path}")
        return markdown_path
    except Exception as e:
        warn(f"Failed to write markdown file: {str(e)}")
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
                err(f"Failed to download image: {url}, invalid content type: {response.headers.get('Content-Type')}")
                return None
        else:
            err(f"Failed to download image: {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        err(f"Failed to download image: {url}, error: {str(e)}")
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


deprecated = '''
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
            err(f"No image found in article")

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
                err(f"Failed to generate TTS for np3k. {e}")

        try:
            body += f"by {authors} in {parsed_content.canonical_link}" # update with method for getting the newspaper name
            body += f"> [!summary]+\n"
            body += f"> {summary}\n\n"
            body += parsed_content["content"]
            markdown_content = frontmatter + body

        except Exception as e:
            err(f"Failed to combine elements of article markdown.")

        try:
            with open(markdown_filename, 'w') as md_file:
                md_file.write(markdown_content)

            info(f"Successfully saved to {markdown_filename}")
            await note.add_to_daily_note(relative_path)
            return markdown_filename
        
        except Exception as e:
            err(f"Failed to write markdown file")
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        err(f"Failed to clip: {str(e)}")
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
            err(f"No image found in article")

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
                err(f"Failed to generate TTS for np3k. {e}")

        try:
            body += f"by {authors} in [{parsed_content.get('domain', urlparse(url).netloc.replace('www.', ''))}]({url}).\n\n"
            body += f"> [!summary]+\n"
            body += f"> {summary}\n\n"
            body += parsed_content["content"]
            markdown_content = frontmatter + body

        except Exception as e:
            err(f"Failed to combine elements of article markdown.")

        try:
            with open(markdown_filename, 'w', encoding=encoding) as md_file:
                md_file.write(markdown_content)

            info(f"Successfully saved to {markdown_filename}")
            await note.add_to_daily_note(relative_path)
            return markdown_filename
        
        except Exception as e:
            err(f"Failed to write markdown file")
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        err(f"Failed to clip {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
'''