'''
Manages an Obsidian vault, in particular daily notes, using information and functionality drawn from the other routers, primarily calendar, email, ig, llm, rag, img, serve, time, tts, and weather.
'''
# routers/note.py

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, Form, HTTPException, Response, Query, Path as FastAPIPath
from fastapi.responses import JSONResponse, PlainTextResponse
import os, re
import traceback
from typing import Optional, Union, Dict, List, Tuple
import re
import os
from io import BytesIO
import mimetypes
from datetime import timedelta, datetime as dt_datetime, time as dt_time, date as dt_date
from dateutil.parser import parse as dateutil_parse
from fastapi import HTTPException, status
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from sijapi import Sys, OBSIDIAN_VAULT_DIR, OBSIDIAN_RESOURCES_DIR, OBSIDIAN_BANNER_SCENE, GEO
from sijapi.routers import asr, cal, gis, img, llm, serve, timing, tts, weather
from sijapi.utilities import assemble_journal_path, convert_to_12_hour_format, sanitize_filename, convert_degrees_to_cardinal, check_file_name, HOURLY_COLUMNS_MAPPING
from sijapi.classes import Location
from sijapi.logs import get_logger

l = get_logger(__name__)

note = APIRouter()

@note.post("/note/add")
async def note_add_endpoint(file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), source: Optional[str] = Form(None), bg_tasks: BackgroundTasks = None):
    l.debug(f"Received request on /note/add...")
    if not file and not text:
        l.warning(f"... without any file or text!")
        raise HTTPException(status_code=400, detail="Either text or a file must be provided")
    else:
        result = await process_for_daily_note(file, text, source, bg_tasks)
        l.info(f"Result on /note/add: {result}")
        return JSONResponse({"message": "Note added successfully", "entry": result}, status_code=201)


async def process_for_daily_note(file: Optional[UploadFile] = File(None), text: Optional[str] = None, source: Optional[str] = None, bg_tasks: BackgroundTasks = None):
    now = dt_datetime.now()
    transcription_entry = ""
    file_entry = ""
    if file:
        l.debug("File received...")
        file_content = await file.read()
        audio_io = BytesIO(file_content)

        # Improve error handling for file type guessing
        guessed_type = mimetypes.guess_type(file.filename)
        file_type = guessed_type[0] if guessed_type[0] else "application/octet-stream"

        l.debug(f"Processing as {file_type}...")

        # Extract the main type (e.g., 'audio', 'image', 'video')
        main_type = file_type.split('/')[0]
        subdir = main_type.title() if main_type else "Documents"

        absolute_path, relative_path = assemble_journal_path(now, subdir=subdir, filename=file.filename)
        l.debug(f"Destination path: {absolute_path}")

        with open(absolute_path, 'wb') as f:
            f.write(file_content)
        l.debug(f"Processing {f.name}...")

        if main_type == 'audio':
            transcription = await asr.transcribe_audio(file_path=absolute_path, params=asr.TranscribeParams(model="small-en", language="en", threads=6))
            file_entry = f"![[{relative_path}]]"
        elif main_type == 'image':
            file_entry = f"![[{relative_path}]]"
        else:
            file_entry = f"[Source]({relative_path})"

    text_entry = text if text else ""
    l.debug(f"transcription: {transcription_entry}\nfile_entry: {file_entry}\ntext_entry: {text_entry}")
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

    return {"timestamp": time_str, "content": entry.strip()}


async def process_document(
    bg_tasks: BackgroundTasks,
    document: File,
    title: Optional[str] = None,
    tts_mode: str = "summary",
    voice: str = None,
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
                l.error(f"Failed in the TTS portion of clipping: {e}")

        body += f"> [!summary]+\n"
        body += f"> {summary}\n\n"
        body += parsed_content
        markdown_content = frontmatter + body

        markdown_filename = f"{readable_title}.md"
        encoding = 'utf-8'

        with open(markdown_filename, 'w', encoding=encoding) as md_file:
            md_file.write(markdown_content)

        l.info(f"Successfully saved to {markdown_filename}")

        return markdown_filename

    except Exception as e:
        l.error(f"Failed to clip: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def list_and_correct_impermissible_files(root_dir, rename: bool = False):
    """List and correct all files with impermissible names."""
    impermissible_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if check_file_name(filename):
                file_path = Path(dirpath) / filename
                impermissible_files.append(file_path)
                l.debug(f"Impermissible file found: {file_path}")

                # Sanitize the file name
                new_filename = sanitize_filename(filename)
                new_file_path = Path(dirpath) / new_filename

                # Ensure the new file name does not already exist
                if new_file_path.exists():
                    counter = 1
                    base_name, ext = os.path.splitext(new_filename)
                    while new_file_path.exists():
                        new_filename = f"{base_name}_{counter}{ext}"
                        new_file_path = Path(dirpath) / new_filename
                        counter += 1

                # Rename the file
                if rename:
                    os.rename(file_path, new_file_path)
                    l.debug(f"Renamed: {file_path} -> {new_file_path}")

    return impermissible_files

journal = OBSIDIAN_VAULT_DIR / "journal"
list_and_correct_impermissible_files(journal, rename=True)


@note.get("/note/bulk/{dt_start}/{dt_end}")
async def build_daily_note_range_endpoint(dt_start: str, dt_end: str):
    start_date = dt_datetime.strptime(dt_start, "%Y-%m-%d")
    end_date = dt_datetime.strptime(dt_end, "%Y-%m-%d")

    results = []
    current_date = start_date
    while current_date <= end_date:
        formatted_date = await gis.dt(current_date)
        result = await build_daily_note(formatted_date)
        results.append(result)
        current_date += timedelta(days=1)

    return {"urls": results}


@note.get("/note/create")
async def build_daily_note_getpoint():
    try:
        loc = await gis.get_last_location()
        if not loc:
            raise ValueError("Unable to retrieve last location")

        tz = await GEO.tz_current(loc)
        if not tz:
            raise ValueError(f"Unable to determine timezone for location: {loc}")

        date_time = dt_datetime.now(tz)
        path = await build_daily_note(date_time, loc.latitude, loc.longitude)
        path_str = str(path)

        l.info(f"Successfully created daily note at {path_str}")
        return JSONResponse(content={"path": path_str}, status_code=200)

    except ValueError as ve:
        error_msg = f"Value Error in build_daily_note_getpoint: {str(ve)}"
        l.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        error_msg = f"Unexpected error in build_daily_note_getpoint: {str(e)}"
        l.error(error_msg)
        l.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@note.post("/note/create")
async def build_daily_note_endpoint(
    date_str: Optional[str] = Form(dt_datetime.now().strftime("%Y-%m-%d")),
    location: Optional[str] = Form(None)
):
    lat, lon = None, None
    try:
        if not date_str:
            date_str = dt_datetime.now().strftime("%Y-%m-%d")
        if location:
            lat, lon = map(float, location.split(','))
            tz = await GEO.tz_at(lat, lon)
            date_time = dateutil_parse(date_str).replace(tzinfo=tz)
        else:
            raise ValueError("Location is not provided or invalid.")
    except (ValueError, AttributeError, TypeError) as e:
        l.warning(f"Falling back to localized datetime due to error: {e}")
        try:
            date_time = await gis.dt(date_str)
            places = await gis.fetch_locations(date_time)
            lat, lon = places[0].latitude, places[0].longitude
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)

    path = await build_daily_note(date_time, lat, lon)

    path_str = str(path)  # Convert PosixPath to string

    return JSONResponse(content={"path": path_str}, status_code=200)


async def build_daily_note(date_time: dt_datetime, lat: float = None, lon: float = None):
    '''
Obsidian helper. Takes a datetime and creates a new daily note. Note: it uses the sijapi configuration file to place the daily note and does NOT presently interface with Obsidian's daily note or periodic notes extensions. It is your responsibility to ensure they match.
    '''
    absolute_path, _ = assemble_journal_path(date_time)
    l.debug(f"Using {date_time.strftime('%Y-%m-%d %H:%M:%S')} as our datetime in build_daily_note.")
    formatted_day = date_time.strftime("%A %B %d, %Y")  # Monday May 27, 2024 formatting
    day_before = (date_time - timedelta(days=1)).strftime("%Y-%m-%d %A")  # 2024-05-26 Sunday formatting
    day_after = (date_time + timedelta(days=1)).strftime("%Y-%m-%d %A")  # 2024-05-28 Tuesday formatting
    header = f"# [[{day_before}|← ]] {formatted_day} [[{day_after}| →]]\n\n"

    if not lat or not lon:
        places = await gis.fetch_locations(date_time)
        lat, lon = places[0].latitude, places[0].longitude

    location = await GEO.code((lat, lon))

    timeslips = await build_daily_timeslips(date_time)

    fm_day = date_time.strftime("%Y-%m-%d")

    _, weather_path = assemble_journal_path(date_time, filename="Weather", extension=".md", no_timestamp = True)
    weather_note = await update_dn_weather(date_time, force_refresh=True)
    weather_embed = f"![[{weather_path}]]\n"

    events = await update_daily_note_events(date_time)
    _, event_path = assemble_journal_path(date_time, filename="Events", extension=".md", no_timestamp = True)
    event_embed = f"![[{event_path}]]"

    _, task_path = assemble_journal_path(date_time, filename="Tasks", extension=".md", no_timestamp = True)
    task_embed = f"![[{task_path}]]"

    _, note_path = assemble_journal_path(date_time, filename="Notes", extension=".md", no_timestamp = True)
    note_embed = f"![[{note_path}]]"

    absolute_map_path, map_path = assemble_journal_path(date_time, filename="Map", extension=".png", no_timestamp = True)
    map = await gis.generate_and_save_heatmap(date_time, output_path=absolute_map_path)
    map_embed = f"![[{map_path}]]"

    _, banner_path = assemble_journal_path(date_time, filename="Banner", extension=".jpg", no_timestamp = True)

    body = f"""---
date: "{fm_day}"
banner: "![[{banner_path}]]"
tags:
 - daily-note
created: "{dt_datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
---

{header}
{weather_embed}
{map_path}

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

    if Sys.EXTENSIONS.comfyui:
        banner = await generate_banner(formatted_day, location, weather_note)

    return absolute_path


async def build_daily_timeslips(date):
    absolute_path, relative_path = assemble_journal_path(date, filename = "Timeslips", extension=".md", no_timestamp = True)
    content = await timing.process_timing_markdown(date, date)
    # document_content = await document.read()
    with open(absolute_path, 'wb') as f:
        f.write(content.encode())

    return f"![[{relative_path}]]"


@note.put("/note/update_frontmatter")
async def update_frontmatter_endpoint(date: str, key: str, value: str):
    date_time = dt_datetime.strptime(date, "%Y-%m-%d")
    result = await update_frontmatter(date_time, key, value)
    return result


async def update_frontmatter(date_time: dt_datetime, key: str, value: str):
    file_path, relative_path = assemble_journal_path(date_time)
    if not file_path.exists():
        l.critical(f"Markdown file not found at {file_path}")
        raise HTTPException(status_code=404, detail="Markdown file not found.")

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    try:
        start_index = lines.index("---\n") + 1
        end_index = lines[start_index:].index("---\n") + start_index
        frontmatter = lines[start_index:end_index]
    except ValueError:
        raise HTTPException(status_code=500, detail="Frontmatter not found.")

    pattern = re.compile(f"^{key}:.*", re.IGNORECASE)
    frontmatter = [line for line in frontmatter if not pattern.match(line)]
    values = value.split(',')
    if len(values) == 1:
        new_entry = f"{key}: {values[0]}\n"
    else:
        new_entry = f"{key}:\n" + "\n".join([f" - {val}" for val in values]) + "\n"

    frontmatter.append(new_entry)
    content = lines[:start_index] + frontmatter + ["---\n"] + lines[end_index + 1:]
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(content)

    return {"message": "Frontmatter updated successfully."}


if Sys.EXTENSIONS.comfyui:
    @note.post("/note/banner")
    async def banner_endpoint(dt: str, location: str = None, forecast: str = None, mood: str = None, other_context: str = None):
        '''
            Endpoint (POST) that generates a new banner image for the Obsidian daily note for a specified date, taking into account optional additional information, then updates the frontmatter if necessary.
        '''
        l.debug(f"banner_endpoint requested with date: {dt} ({type(dt)})")
        date_time = await gis.dt(dt)
        l.debug(f"date_time after localization: {date_time} ({type(date_time)})")
        context = await generate_banner_context(dt, location, forecast, mood, other_context)
        jpg_path = await generate_banner(date_time, location, mood=mood, other_context=other_context)
        return jpg_path


async def generate_banner(dt, location: Location = None, forecast: str = None, mood: str = None, other_context: str = None):
    if Sys.EXTENSIONS.comfyui:
        date_time = await gis.dt(dt)
        destination_path, local_path = assemble_journal_path(date_time, filename="Banner", extension=".jpg", no_timestamp = True)
        if not location or not isinstance(location, Location):
            locations = await gis.fetch_locations(date_time)
            if locations:
                location = locations[0]
        if not forecast:
            forecast = await update_dn_weather(date_time, False, location.latitude, location.longitude)
    
        prompt = await generate_banner_context(date_time, location, forecast, mood, other_context)
        l.debug(f"Prompt: {prompt}")
        final_path = await img.workflow(prompt, scene=OBSIDIAN_BANNER_SCENE, destination_path=destination_path)
        if not str(local_path) in str(final_path):
            l.info(f"Apparent mismatch between local path, {local_path}, and final_path, {final_path}")
        jpg_embed = f"\"![[{local_path}]]\""
        await update_frontmatter(date_time, "banner", jpg_embed)
        return local_path
    else:
        l.warn(f"generate_banner called, but comfyui extension is disabled")


async def generate_banner_context(date_time, location: Location, forecast: str, mood: str, other_context: str):
    if Sys.EXTENSIONS.comfyui:
        display_name = "Location: "
        if location and isinstance(location, Location):
            lat, lon = location.latitude, location.longitude
            override_location = GEO.find_override_location(lat, lon)
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
                geocoded_location = await GEO.code((lat, lon))
                if geocoded_location.display_name or geocoded_location.city or geocoded_location.country:
                    return await generate_banner_context(date_time, geocoded_location, forecast, mood, other_context)
                else:
                    l.warning(f"Failed to get a useable location for purposes of generating a banner, but we'll generate one anyway.")
        elif location and isinstance(location, str):
            display_name = f"Location: {location}\n"
        else:
            display_name = ""
    
        if not forecast:
            forecast = "The weather forecast is: " + await update_dn_weather(date_time)
    
        sentiment = await sentiment_analysis(date_time)
        mood = sentiment if not mood else mood
        mood = f"Mood: {mood}" if mood else ""
        if mood and sentiment: mood = f"Mood: {mood}, {sentiment}"
        elif mood and not sentiment: mood = f"Mood: {mood}"
        elif sentiment and not mood: mood = f"Mood: {sentiment}"
        else: mood = ""
    
        events = await cal.get_events(date_time, date_time)
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
    
        return prompt
    else:
        l.warn(f"generate_banner_context called, but comfyui extension disabled")


async def get_note(date_time: dt_datetime):
    date_time = await gis.dt(date_time);
    absolute_path, local_path = assemble_journal_path(date_time, filename = "Notes", extension = ".md", no_timestamp = True)

    if absolute_path.is_file():
        with open(absolute_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content if content else None


async def sentiment_analysis(date_time: dt_datetime):
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


@note.get("/note/weather", response_class=JSONResponse)
async def note_weather_get(
    date: str = Query(default="0", description="Enter a date in YYYY-MM-DD format, otherwise it will default to today."),
    latlon: str = Query(default="45,-125"),
    refresh: str = Query(default="False", description="Set to True to force refresh the weather data")
):
    force_refresh_weather = refresh == "True"
    try:
        date_time = dt_datetime.now() if date == "0" else await gis.dt(date)
        l.debug(f"Using {date_time.strftime('%Y-%m-%d %H:%M:%S')} as our dt_datetime in note_weather_get.")
        l.debug(f"date: {date} .. date_time: {date_time}")
        content = await update_dn_weather(date_time, force_refresh_weather) #, lat, lon)
        return JSONResponse(content={"forecast": content}, status_code=200)

    except HTTPException as e:
        return JSONResponse(content={"detail": str(e.detail)}, status_code=e.status_code)

    except Exception as e:
        l.error(f"Error in note_weather_get: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@note.post("/update/note/{date}")
async def post_update_daily_weather_and_calendar_and_timeslips(date: str, refresh: str="True") -> PlainTextResponse:
    date_time = await gis.dt(date)
    l.debug(f"Using {date_time.strftime('%Y-%m-%d %H:%M:%S')} as our dt_datetime in post_update_daily_weather_and_calendar_and_timeslips.")
    force_refresh_weather = refresh == "True" 
    await update_dn_weather(date_time, force_refresh_weather)
    await update_daily_note_events(date_time)
    await build_daily_timeslips(date_time)
    return f"[Refresh]({Sys.URL}/update/note/{date_time.strftime('%Y-%m-%d')}"



async def update_dn_weather(date_time: dt_datetime, force_refresh: bool = False, lat: float = None, lon: float = None):
    l.debug(f"Using {date_time.strftime('%Y-%m-%d %H:%M:%S')} as our datetime in update_dn_weather.")
    try:
        if lat and lon:
            place = await GEO.code((lat, lon))
        else:
            l.debug(f"Updating weather for {date_time}")
            places = await gis.fetch_locations(date_time)
            place = places[0]
            lat = place.latitude
            lon = place.longitude
    
        # Get local timezone for the coordinates
        local_tz = await GEO.timezone(lat, lon)
        if not local_tz:
            raise ValueError("Could not determine timezone for coordinates")
        
        l.debug(f"lat: {lat}, lon: {lon}, place: {place}, timezone: {local_tz}")
        
        city = GEO.find_override_location(lat, lon)
        if city:
            l.info(f"Using override location: {city}")
        else:
            if place.city and place.city != "":
                city = place.city
                l.info(f"City in data: {city}")
            else:
                location = await GEO.code((lat, lon))
                l.debug(f"location: {location}")
                city = location.name
                city = city if city else location.city
                city = city if city else location.house_number + ' ' + location.road
                l.debug(f"City geocoded: {city}")
    
        absolute_path, relative_path = assemble_journal_path(date_time, filename="Weather", extension=".md", no_timestamp=True)
        l.debug(f"Journal path: absolute_path={absolute_path}, relative_path={relative_path}")
    
        try:
            l.debug(f"passing date_time {date_time.strftime('%Y-%m-%d %H:%M:%S')}, {lat}/{lon} into get_weather")
            day = await weather.get_weather(date_time, lat, lon, force_refresh)
            l.debug(f"day information obtained from get_weather: {day}")
            if day:
                DailyWeather = day.get('DailyWeather')
                HourlyWeather = day.get('HourlyWeather')
                if DailyWeather:
                    icon = DailyWeather.get('icon')
                    l.debug(f"Icon: {icon}")
    
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
                    uvwarn = f" - :LiRadiation: Caution! UVI today is {uvindex}! :LiRadiation:\n" if (uvindex and uvindex > 8) else ""
    
                    # Convert sunrise/sunset times to local timezone
                    sunrise = DailyWeather.get('sunrise').astimezone(local_tz)
                    sunset = DailyWeather.get('sunset').astimezone(local_tz)
                    srise_str = sunrise.strftime("%H:%M")
                    sset_str = sunset.strftime("%H:%M")
    
                    date_str = date_time.astimezone(local_tz).strftime("%Y-%m-%d")
                    now = dt_datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")
    
                    detailed_forecast = (
                        f"---\n"
                        f"date: {date_str}\n"
                        f"latitude: {lat}\n"
                        f"longitude: {lon}\n"
                        f"timezone: {local_tz}\n"
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
                            # Convert hourly datetime to local timezone before checking/formatting
                            local_hour = hour.get('datetime').astimezone(local_tz)
                            if local_hour.strftime("%H:%M:%S") in HOURLY_COLUMNS_MAPPING.values():
                                # Pass localized datetime to formatting functions
                                times.append(format_hourly_time(local_hour))
                                condition_symbols.append(format_hourly_icon(hour, sunrise, sunset))
                                temps.append(format_hourly_temperature(hour))
                                winds.append(format_hourly_wind(hour))
    
                        detailed_forecast += assemble_hourly_data_table(times, condition_symbols, temps, winds)
                        detailed_forecast += f"```\n\n"
    
                    l.debug(f"Detailed forecast: {detailed_forecast}.")
    
                    with open(absolute_path, 'w', encoding='utf-8') as note_file:
                        note_file.write(detailed_forecast)
    
                    l.debug(f"Operation complete.")
                    return narrative
                else:
                    l.error(f"Failed to get DailyWeather from day: {day}")
            else:
                l.error(f"Failed to get day")
                raise HTTPException(status_code=500, detail="Failed to retrieve weather data")
    
        except HTTPException as e:
            l.error(f"HTTP error: {e}")
            l.error(traceback.format_exc())
            raise e
    
        except Exception as e:
            l.error(f"Error: {e}")
            l.error(traceback.format_exc())
            raise HTTPException(status_code=999, detail=f"Error: {e}")
    
    except ValueError as ve:
        l.error(f"Value error in update_dn_weather: {str(ve)}")
        l.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Value error: {str(ve)}")
    
    except Exception as e:
        l.error(f"Error in update_dn_weather: {str(e)}")
        l.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in update_dn_weather: {str(e)}")


def format_hourly_time(hour_datetime):
    try:
        # Assumes hour_datetime is already in local timezone from the calling function
        hour_12 = convert_to_12_hour_format(hour_datetime)
        return hour_12
    except Exception as e:
        l.error(f"Error in format_hourly_time: {str(e)}")
        l.error(traceback.format_exc())
        return ""

def format_hourly_icon(hour, sunrise, sunset):
    try:
        icon_str = hour.get('icon', '')
        icon, _ = get_icon_and_admonition(icon_str)

        precip = hour.get('precip', float(0.0))
        precip_prob = hour.get('precipprob', float(0.0))
        l.debug(f"precip: {precip}, prob: {precip_prob}")

        sp_str = None

        if (precip > 0.05 and precip_prob > 25):
            precip_type = hour.get('preciptype', [''])
            sp_str = f"{str(precip)}mm"

        # Use the already-converted datetime from the calling function
        hour_datetime = hour.get('datetime')  # This should already be in local timezone
        
        # Sunrise and sunset should already be in local timezone from calling function
        if abs(hour_datetime - sunrise) < timedelta(minutes=60):
            icon = ":LiSunrise:"
        elif abs(hour_datetime - sunset) < timedelta(minutes=60):
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
        l.error(f"Error in format_hourly_special: {str(e)}")
        l.error(traceback.format_exc())
        return ""


def format_hourly_temperature(hour):
    try:
        temp_str = f"{hour.get('temp', '')}˚ F"
        return temp_str
    except Exception as e:
        l.error(f"Error in format_hourly_temperature: {str(e)}")
        l.error(traceback.format_exc())
        return ""


def format_hourly_wind(hour):
    try:
        windspeed = hour.get('windspeed', '')
        winddir = convert_degrees_to_cardinal(float(hour.get('winddir', ''))) if windspeed else ""
        wind_str = f"{str(windspeed)}:LiWind: {winddir}"
        return wind_str
    except Exception as e:
        l.error(f"Error in format_hourly_wind: {str(e)}")
        l.error(traceback.format_exc())
        return ""

def assemble_hourly_data_table(times, condition_symbols, temps, winds):
    table_rows = [times, condition_symbols, temps, winds]
    table = "| " + " | ".join(times) + " |\n"
    table += "| " + " | ".join([':----:' for _ in times]) + " |\n"
    for row in table_rows[1:]:
        table += "| " + " | ".join(row) + " |\n"
    return table + "\n\n"


def get_icon_and_admonition(icon_str) -> Tuple:
    l.debug(f"Received request for emoji {icon_str}")
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
    if 'clear' in condition or 'sunny' in condition: return "☀️"
    elif 'cloud' in condition or 'overcast' in condition: return "☁️"
    elif 'rain' in condition: return "🌧️"
    elif 'snow' in condition: return "❄️"
    elif 'thunder' in condition or 'storm' in condition: return "⛈️"
    elif 'fog' in condition or 'mist' in condition: return "🌫️"
    elif 'wind' in condition: return "💨"
    elif 'hail' in condition: return "🌨️"
    elif 'sleet' in condition: return "🌧️"
    elif 'partly' in condition: return "⛅"
    else: return "🌡️"  # Default emoji for unclassified weather


async def format_events_as_markdown(event_data: Dict[str, Union[str, List[Dict[str, str]]]]) -> str:
    def remove_characters(s: str) -> str:
        s = s.replace('---', '\n')
        s = s.strip('\n')
        s = re.sub(r'^_+|_+$', '', s)
        return s

    date_str = event_data["date"]
    now = dt_datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    events_markdown = []

    event_data["events"] = sorted(event_data["events"], key=lambda event: (not event['all_day'], dt_datetime.strptime(event['start'], "%H:%M")), reverse=False)

    total_events = len(event_data["events"])
    event_markdown = f"```ad-events"
    for event in event_data["events"]:
        l.debug(f"event busy status: {event['busy']}; all day status: {event['all_day']}")
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

            if (event['all_day']) or (event['start'] == event['end'] == "00:00") or (dt_datetime.combine(dt_date.min, dt_datetime.strptime(event['end'], "%H:%M").time()) - dt_datetime.combine(dt_date.min, dt_datetime.strptime(event['start'], "%H:%M").time()) >= timedelta(hours=23, minutes=59)):
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

    date_time = await gis.dt(date) if date else await gis.dt(dt_datetime.now())
    response = await update_daily_note_events(date_time)
    return PlainTextResponse(content=response, status_code=200)


async def update_daily_note_events(date_time: dt_datetime):
    l.debug(f"Looking up events on date: {date_time.strftime('%Y-%m-%d')}")
    try:
        events = await cal.get_events(date_time, date_time)
        l.debug(f"Raw events: {events}")
        event_data = {
            "date": date_time.strftime('%Y-%m-%d'),
            "events": events
        }
        events_markdown = await format_events_as_markdown(event_data)
        l.debug(f"Markdown events: {events_markdown}")
        absolute_path, _ = assemble_journal_path(date_time, filename="Events", extension=".md", no_timestamp = True)
        l.debug(f"Writing events to file: {absolute_path}")

        with open(absolute_path, 'w', encoding='utf-8') as note_file:
            note_file.write(events_markdown)

        return events_markdown

    except Exception as e:
        l.error(f"Error processing events: {e}")
        raise HTTPException(status_code=500, detail=str(e))
