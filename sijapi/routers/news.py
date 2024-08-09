'''
Used to scrape, process, summarize, markdownify, and speechify news articles.
'''
# routers/news.py

import os
import asyncio
from bs4 import BeautifulSoup
from datetime import datetime as dt_datetime, timedelta
from typing import Optional, List, Tuple
import aiofiles
import trafilatura
import newspaper
from newspaper import Article
import math
from urllib.parse import urlparse
from markdownify import markdownify as md
from better_profanity import profanity
from fastapi import APIRouter, BackgroundTasks, UploadFile, Form, HTTPException, Query, Path as FastAPIPath
from pathlib import Path
from sijapi import L, News, Archivist, OBSIDIAN_VAULT_DIR, OBSIDIAN_RESOURCES_DIR
from sijapi.utilities import html_to_markdown, download_file, sanitize_filename, assemble_journal_path, assemble_archive_path, contains_profanity, is_ad_or_tracker
from sijapi.routers import gis, llm, tts, note

news = APIRouter()
logger = L.get_module_logger("news")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)


@news.post("/clip")
async def clip_post(
    bg_tasks: BackgroundTasks,
    url: str = Form(...),
    title: Optional[str] = Form(None),
    tts: str = Form('summary'),
    voice: str = Form(None),
):
    result = await process_and_save_article(bg_tasks, url, title, tts, voice)
    return {"message": "Clip saved successfully", "result": result}

@news.get("/clip")
async def clip_get(
    bg_tasks: BackgroundTasks,
    url: str,
    voice: str = Query(None)
):
    result = await process_and_save_article(bg_tasks, url, None, tts, voice)
    return {"message": "Clip saved successfully", "result": result}

@news.get("/news/refresh")
async def news_refresh_endpoint(bg_tasks: BackgroundTasks):
    tasks = [process_news_site(site, bg_tasks) for site in News.sites]
    await asyncio.gather(*tasks)
    return "OK"

def is_article_within_date_range(article: Article, days_back: int) -> bool:
    earliest_date = dt_datetime.now().date() - timedelta(days=days_back)
    return article.publish_date.date() >= earliest_date

async def generate_summary(text: str) -> str:
    summary = await llm.summarize_text(text, "Summarize the provided text. Respond with the summary and nothing else.")
    return summary.replace('\n', ' ')

async def handle_tts(bg_tasks: BackgroundTasks, article: Article, title: str, tts_mode: str, voice: str, summary: str, model: str = "eleven_turbo_v2") -> Optional[str]:
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
            model=model,
            podcast=True,
            title=audio_filename,
            output_dir=Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_RESOURCES_DIR
        )
        return f"![[{Path(audio_path).name}]]"

    except HTTPException as e:
        err(f"Failed to generate TTS: {str(e)}")
        return None



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
                voice=site.voice if hasattr(site, 'voice') else Tts.elevenlabs.default
            ))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        articles_downloaded = sum(results)
        
        info(f"Downloaded {articles_downloaded} articles from {site.name}")
    except Exception as e:
        err(f"Error processing {site.name}: {str(e)}")


async def download_and_save_article(article, site_name, earliest_date, bg_tasks: BackgroundTasks, tts_mode: str = "off", voice: str = Tts.elevenlabs.default):
    try:
        url = article.url
        parsed_article = await fetch_and_parse_article(url)
        
        if not is_article_within_date_range(parsed_article, earliest_date):
            return False

        return await process_and_save_article(bg_tasks, url, None, tts_mode, voice, site_name=site_name)

    except Exception as e:
        err(f"Error processing article from {article.url}: {str(e)}")
        return False


async def process_and_save_article(
    bg_tasks: BackgroundTasks,
    url: str,
    title: Optional[str] = None,
    tts_mode: str = "summary",
    voice: str = Tts.elevenlabs.default,
    site_name: Optional[str] = None
) -> str:
    
    try:
        # Fetch and parse article
        article = await fetch_and_parse_article(url)

        try:        
            # Generate title and file paths
            title = sanitize_filename(title or article.title or f"Untitled - {dt_datetime.now().strftime('%Y-%m-%d')}")
            markdown_filename, relative_path = assemble_journal_path(dt_datetime.now(), subdir="Articles", filename=title, extension=".md")

            # Generate summary
            summary = await generate_summary(article.text)

            try:
                # Handle TTS
                audio_link = await handle_tts(bg_tasks, article, title, tts_mode, voice, summary)

                try:
                    # Generate markdown content
                    markdown_content = generate_markdown_content(article, title, summary, audio_link, site_name)

                    # Save markdown file
                    await save_markdown_file(markdown_filename, markdown_content)

                    return f"Successfully saved: {relative_path}"
                
                except Exception as e:
                    err(f"Failed to handle final markdown content preparation and/or saving to daily note; {e}")
            
            except Exception as e:
                err(f"Failed to handle TTS: {e}")
        
        except Exception as e:
            err(f"Failed to generate title, file paths, and summary: {e}")
        
    except Exception as e:
        err(f"Failed to fetch and parse article {url}: {str(e)}")
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


def generate_markdown_content(article, title: str, summary: str, audio_link: Optional[str], site_name: Optional[str] = None) -> str:
    def format_date(date):
        return date.strftime("%Y-%m-%d") if date else "Unknown Date"

    def estimate_reading_time(text, words_per_minute=200):
        word_count = len(text.split())
        return math.ceil(word_count / words_per_minute)

    def format_tags(tags):
        return '\n'.join(f' - {tag}' for tag in (tags or []))

    def get_banner_markdown(image_url):
        return image_url if image_url else ""

    # Prepare metadata
    publish_date = format_date(article.publish_date)
    added_date = dt_datetime.now().strftime("%b %d, %Y at %H:%M")
    reading_time = estimate_reading_time(article.text)

    frontmatter = f"""---
title: {title}
authors: {', '.join(f'[[{author}]]' for author in article.authors)}
published: {publish_date}
added: {added_date}
banner: "{get_banner_markdown(article.top_image)}"
url: {article.url}
reading_minutes: {reading_time}
tags:
{format_tags(article.meta_keywords)}"""

    if site_name:
        frontmatter += f"\nsite: {site_name}"
    frontmatter += "\n---\n\n"

    body = f"# {title}\n\n"
    if article.top_image:
        body += f"![{title}]({article.top_image})\n\n"
    if audio_link:
        body += f"{audio_link}\n\n"
    body += f"by {', '.join(article.authors)} in [{article.source_url}]({article.url})\n\n"
    body += f"> [!summary]+\n> {summary}\n\n"
    body += article.text

    return frontmatter + body