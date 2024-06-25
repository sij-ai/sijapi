import tempfile
import os
import json
import requests
import csv
import subprocess
import asyncio
import httpx
import io
import re
import pytz
import httpx
import sqlite3
import math
from httpx import Timeout
from fastapi import APIRouter, UploadFile, File, Response, Header, Query, Depends, FastAPI, Request, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from sijapi.utilities import localize_datetime
from decimal import Decimal, ROUND_UP
from typing import Optional, List, Dict, Union, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from traceback import format_exc
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi import HOME_DIR, TIMING_API_KEY, TIMING_API_URL

### INITIALIZATIONS ###
time = APIRouter(tags=["private"])


########################
#### INITIALIZATION ####
########################

script_directory = os.path.dirname(os.path.abspath(__file__))

# Configuration constants
pacific = pytz.timezone('America/Los_Angeles')

emoji_pattern = re.compile(r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+ ')
timeout = Timeout(connect=30, read=600, write=120, pool=5)

# Define your models
class TimingRequest(BaseModel):
    start_date: str = Field(..., pattern=r"\d{4}-\d{2}-\d{2}")
    end_date: Optional[str] = Field(None, pattern=r"\d{4}-\d{2}-\d{2}")
    output_format: Optional[str] = 'json'


####################
#### TIMING API ####
####################
@time.post("/time/post")
async def post_time_entry_to_timing(entry: Dict):
    url = 'https://web.timingapp.com/api/v1/time-entries'
    headers = {
        'Authorization': f'Bearer {TIMING_API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Time-Zone': 'America/Los_Angeles'
    }
    DEBUG(f"Received entry: {entry}")
    response = None  # Initialize response
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=entry)
            response.raise_for_status()  # This will only raise for 4xx and 5xx responses
    except httpx.HTTPStatusError as exc:
        DEBUG(f"HTTPStatusError caught: Status code: {exc.response.status_code}, Detail: {exc.response.text}")
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc.response.text))
    except Exception as exc:
        DEBUG(f"General exception caught: {exc}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

    if response:
        return response.json()
    else:
        # Handle the case where the response was not set due to an error.
        raise HTTPException(status_code=500, detail="Failed to make the external API request")

def project_sort_key(project):
    # Remove any leading emoji characters for sorting
    return emoji_pattern.sub('', project)


def prepare_date_range_for_query(start_date, end_date=None):
    # Adjust the start date to include the day before
    start_date_adjusted = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    # If end_date is not provided, use the original start_date as the end_date
    end_date = end_date if end_date else start_date
    # Format the end_date
    end_date_formatted = f"{end_date}T23:59:59"
    return f"{start_date_adjusted}T00:00:00", end_date_formatted


def truncate_project_title(title):
    return title.split(' - ')[0] if ' - ' in title else title


async def fetch_and_prepare_timing_data(start: datetime, end: Optional[datetime] = None) -> List[Dict]:
    # start_date = localize_datetime(start)
    # end_date = localize_datetime(end) if end else None
    # Adjust the start date to include the day before and format the end date
    start_date_adjusted = (start - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    end_date_formatted = f"{datetime.strftime(end, '%Y-%m-%d')}T23:59:59" if end else f"{datetime.strftime(start, '%Y-%m-%d')}T23:59:59"

    # Fetch timing data from the API using TIMING_API_KEY
    url = f"{TIMING_API_URL}/time-entries?start_date_min={start_date_adjusted}&start_date_max={end_date_formatted}&include_project_data=1"
    headers = {
        'Authorization': f'Bearer {TIMING_API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Time-Zone': 'America/Los_Angeles'
    }

    processed_timing_data = []
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    if response.status_code != 200:
        response.raise_for_status()

    raw_timing_data = response.json().get('data', [])

    for entry in raw_timing_data:
        entry_start_utc = datetime.strptime(entry['start_date'], '%Y-%m-%dT%H:%M:%S.%f%z')
        entry_end_utc = datetime.strptime(entry['end_date'], '%Y-%m-%dT%H:%M:%S.%f%z')

        entry_start_pacific = entry_start_utc.astimezone(pacific)
        entry_end_pacific = entry_end_utc.astimezone(pacific)

        while entry_start_pacific.date() < entry_end_pacific.date():
            midnight = pacific.localize(datetime.combine(entry_start_pacific.date() + timedelta(days=1), datetime.min.time()))
            duration_to_midnight = (midnight - entry_start_pacific).total_seconds()

            if entry_start_pacific.date() >= start.date():
                processed_entry = create_time_entry(entry, entry_start_pacific, midnight, duration_to_midnight)
                processed_timing_data.append(processed_entry)

            entry_start_pacific = midnight

        if entry_start_pacific.date() >= start.date():
            duration_remaining = (entry_end_pacific - entry_start_pacific).total_seconds()
            processed_entry = create_time_entry(entry, entry_start_pacific, entry_end_pacific, duration_remaining)
            processed_timing_data.append(processed_entry)

    return processed_timing_data


def format_duration(duration):
    duration_in_hours = Decimal(duration) / Decimal(3600)
    rounded_duration = duration_in_hours.quantize(Decimal('0.1'), rounding=ROUND_UP)
    return str(rounded_duration)


def create_time_entry(original_entry, start_time, end_time, duration_seconds):
    """Formats a time entry, preserving key details and adding necessary elements."""
    
    # Format start and end times in the appropriate timezone
    start_time_aware = start_time.astimezone(pacific)
    end_time_aware = end_time.astimezone(pacific)

    # Check if project is None and handle accordingly
    if original_entry.get('project'):
        project_title = original_entry['project'].get('title', 'No Project')
        project_color = original_entry['project'].get('color', '#FFFFFF')  # Default color
    else:
        project_title = 'No Project'
        project_color = '#FFFFFF'  # Default color

    # Construct the processed entry
    processed_entry = {
        'start_time': start_time_aware.strftime('%Y-%m-%dT%H:%M:%S.%f%z'),
        'end_time': end_time_aware.strftime('%Y-%m-%dT%H:%M:%S.%f%z'),
        'start_date': start_time_aware.strftime('%Y-%m-%d'), 
        'end_date': end_time_aware.strftime('%Y-%m-%d'),
        'duration': format_duration(duration_seconds),
        'notes': original_entry.get('notes', ''),
        'title': original_entry.get('title', 'Untitled'),
        'is_running': original_entry.get('is_running', False),
        'project': {
            'title': project_title,
            'color': project_color,
            # Include other project fields as needed
        },
        # Additional original fields as required
    }
    return processed_entry


# TIMELINE
@time.get("/time/line")
async def get_timing_timeline(
    request: Request,
    start_date: str = Query(..., regex=r"\d{4}-\d{2}-\d{2}"),
    end_date: Optional[str] = Query(None, regex=r"\d{4}-\d{2}-\d{2}")
):

    # Retain these for processing timeline data with the correct timezone
    queried_start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=pacific).date()
    queried_end_date = (datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pacific).date() 
                        if end_date else queried_start_date)

    # Fetch and process timing data
    timing_data = await fetch_and_prepare_timing_data(start_date, end_date)

    # Process timeline data
    timeline_formatted_data = process_timeline(timing_data, queried_start_date, queried_end_date)

    return Response(content=timeline_formatted_data, media_type="text/markdown")


def process_timeline(timing_data, queried_start_date, queried_end_date):
    timeline_output = []
    entries_by_date = defaultdict(list)

    for entry in timing_data:
        # Convert start and end times to datetime objects and localize to Pacific timezone
        start_datetime = datetime.strptime(entry['start_time'], '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(pacific)
        end_datetime = datetime.strptime(entry['end_time'], '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(pacific)

        project_title = truncate_project_title(entry['project']['title']) if entry.get('project') else 'No Project'
        task_title = entry['title'] if entry.get('title') else 'Untitled'

        # Check if the entry's date falls within the queried date range
        if queried_start_date <= start_datetime.date() <= queried_end_date:
            duration_seconds = (end_datetime - start_datetime).total_seconds()
            duration_hours = format_duration(duration_seconds)

            entries_by_date[start_datetime.date()].append(
                (start_datetime.strftime('%H:%M:%S'), project_title, task_title, duration_hours)
            )
    
    # Sorting and outputting the timeline
    for date, entries in sorted(entries_by_date.items()):
        sorted_entries = sorted(entries, key=lambda x: x[0])
        day_total_duration = sum(Decimal(entry[3]) for entry in sorted_entries)

        if queried_start_date != queried_end_date:
            timeline_output.append(f"## {date.strftime('%Y-%m-%d')} {date.strftime('%A')} [{day_total_duration}]\n")
        for start_time, project, task, duration in sorted_entries:
            timeline_output.append(f" - {start_time} – {project} - {task} [{duration}]")

    return "\n".join(timeline_output)


# CSV
@time.get("/time/csv")
async def get_timing_csv(
    request: Request,
    start_date: str = Query(..., regex=r"\d{4}-\d{2}-\d{2}"),
    end_date: Optional[str] = Query(None, regex=r"\d{4}-\d{2}-\d{2}")
):

    # Fetch and process timing data
    timing_data = await fetch_and_prepare_timing_data(start_date, end_date)

    # Retain these for processing CSV data with the correct timezone
    queried_start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=pacific).date()
    queried_end_date = (datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pacific).date() 
                        if end_date else queried_start_date)

    # Process CSV data
    csv_data = process_csv(timing_data, queried_start_date, queried_end_date)
    if not csv_data or csv_data.strip() == "":
        return Response(content="No CSV data available for the specified date range.", media_type="text/plain")
    return Response(content=csv_data, media_type="text/csv")

def process_csv(timing_data, queried_start_date, queried_end_date):
    project_task_data = defaultdict(lambda: defaultdict(list))

    for entry in timing_data:
        # Convert start and end times to datetime objects and localize to Pacific timezone
        start_datetime = datetime.strptime(entry['start_time'], '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(pacific)
        end_datetime = datetime.strptime(entry['end_time'], '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(pacific)

        # Ensure the entry's date falls within the queried date range
        if queried_start_date <= start_datetime.date() <= queried_end_date:
            duration_seconds = (end_datetime - start_datetime).total_seconds()
            duration_hours = format_duration(duration_seconds)  # Convert duration to hours
            project_title = truncate_project_title(entry['project']['title']) if 'title' in entry['project'] else 'No Project'

            project_task_data[start_datetime.date()][project_title].append(
                (entry['title'] if entry.get('title') else 'Untitled', duration_hours)
            )

    output = io.StringIO()
    writer = csv.writer(output, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Date', 'Project', 'Task', 'Notes', 'Duration'])

    for date, project_tasks in sorted(project_task_data.items()):
        day_total_duration = Decimal(0)
        formatted_date = date.strftime('%Y-%m-%d %a')
        for project, tasks in sorted(project_tasks.items(), key=lambda item: project_sort_key(item[0])):
            task_summary = defaultdict(Decimal)
            for task, duration in tasks:
                task_summary[task] += Decimal(duration)
            project_duration = sum(task_summary.values()).quantize(Decimal('0.1'))
            day_total_duration += project_duration
            tasks_formatted = "; ".join([f"{task.replace(';', ',')} [{str(task_summary[task].quantize(Decimal('0.1')))}]" for task in task_summary])
            writer.writerow([formatted_date, project, tasks_formatted, '', str(project_duration.quantize(Decimal('0.1')))])
        writer.writerow([formatted_date, 'Day Total', '', '', str(day_total_duration.quantize(Decimal('0.1')))])
        writer.writerow(['', '', '', '', ''])

    return output.getvalue()

# MARKDOWN
@time.get("/time/markdown3")
async def get_timing_markdown3(
    request: Request,
    start_date: str = Query(..., regex=r"\d{4}-\d{2}-\d{2}"),
    end_date: Optional[str] = Query(None, regex=r"\d{4}-\d{2}-\d{2}")
):

    # Fetch and process timing data
    start = localize_datetime(start_date)
    end = localize_datetime(end_date) if end_date else None
    timing_data = await fetch_and_prepare_timing_data(start, end)

    # Retain these for processing Markdown data with the correct timezone
    queried_start_date = start.replace(tzinfo=pacific).date()
    queried_end_date = end.replace(tzinfo=pacific).date() if end else queried_start_date

    # Process Markdown data
    markdown_formatted_data = process_timing_markdown3(timing_data, queried_start_date, queried_end_date)
    return Response(content=markdown_formatted_data, media_type="text/markdown")

def process_timing_markdown3(timing_data, queried_start_date, queried_end_date):
    markdown_output = []
    project_task_data = defaultdict(lambda: defaultdict(list))

    for entry in timing_data:
        # Convert start and end times to datetime objects and localize to Pacific timezone
        start_datetime = datetime.strptime(entry['start_time'], '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(pacific)
        end_datetime = datetime.strptime(entry['end_time'], '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(pacific)

        # Check if the entry's date falls within the queried date range
        if queried_start_date <= start_datetime.date() <= queried_end_date:
            duration_seconds = (end_datetime - start_datetime).total_seconds()
            duration_hours = format_duration(duration_seconds)
            project_title = truncate_project_title(entry['project']['title']) if 'title' in entry['project'] else 'No Project'

            project_task_data[start_datetime.date()][project_title].append(
                (entry['title'] if entry.get('title') else 'Untitled', duration_hours)
            )

    for date, projects in sorted(project_task_data.items()):
        day_total_duration = Decimal(0)
        tasks_output = []

        for project, tasks in sorted(projects.items(), key=lambda item: project_sort_key(item[0])):
            task_summary = defaultdict(Decimal)
            for task, duration in tasks:
                task_summary[task] += Decimal(duration)

            project_duration = sum(task_summary.values()).quantize(Decimal('0.1'))
            day_total_duration += project_duration
            tasks_formatted = "; ".join([f"{task.replace(';', ',')} [{duration}]" for task, duration in task_summary.items()])
            tasks_output.append(f"- {project} - {tasks_formatted} - *{project_duration}*.")

        if queried_start_date != queried_end_date:
            markdown_output.append(f"## {date.strftime('%Y-%m-%d %A')} [{day_total_duration}]\n")
        
        markdown_output.extend(tasks_output)
        markdown_output.append("")

    return "\n".join(markdown_output)


@time.get("/time/markdown")
async def get_timing_markdown(
    request: Request,
    start: str = Query(..., regex=r"\d{4}-\d{2}-\d{2}"),
    end: Optional[str] = Query(None, regex=r"\d{4}-\d{2}-\d{2}")
):
    start_date = localize_datetime(start)
    end_date = localize_datetime(end)
    markdown_formatted_data = await process_timing_markdown(start_date, end_date)

    return Response(content=markdown_formatted_data, media_type="text/markdown")

    #return JSONResponse(content={"markdown": markdown_formatted_data}, media_type="text/markdown")


async def process_timing_markdown(start_date: datetime, end_date: datetime): # timing_data, queried_start_date, queried_end_date)
    timing_data = await fetch_and_prepare_timing_data(start_date, end_date)

    queried_start_date = start_date.replace(tzinfo=pacific).date()
    queried_end_date = (end_date.replace(tzinfo=pacific).date() if end_date else queried_start_date)

    markdown_output = []
    project_task_data = defaultdict(lambda: defaultdict(list))
    # pacific = pytz.timezone('US/Pacific')

    for entry in timing_data:
        # Convert start and end times to datetime objects and localize to Pacific timezone
        start_datetime = datetime.strptime(entry['start_time'], '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(pacific)
        end_datetime = datetime.strptime(entry['end_time'], '%Y-%m-%dT%H:%M:%S.%f%z').astimezone(pacific)

        # Check if the entry's date falls within the queried date range
        if queried_start_date <= start_datetime.date() <= queried_end_date:
            duration_seconds = (end_datetime - start_datetime).total_seconds()
            duration_hours = format_duration(duration_seconds)
            project_title = truncate_project_title(entry['project']['title']) if 'title' in entry['project'] else 'No Project'

            project_task_data[start_datetime.date()][project_title].append(
                (entry['title'] if entry.get('title') else 'Untitled', duration_hours)
            )

    for date, projects in sorted(project_task_data.items()):
        day_total_duration = Decimal(0)
        tasks_output = []

        for project, tasks in sorted(projects.items(), key=lambda item: project_sort_key(item[0])):
            task_summary = defaultdict(Decimal)
            for task, duration in tasks:
                task_summary[task] += Decimal(duration)

            project_duration = sum(task_summary.values()).quantize(Decimal('0.1'))
            day_total_duration += project_duration
            tasks_formatted = "; ".join([f"{task.replace(';', ',')} [{duration}]" for task, duration in task_summary.items()])
            tasks_output.append(f"|{project}|{tasks_formatted}|{project_duration}|")

        if queried_start_date != queried_end_date:
            markdown_output.append(f"## {date.strftime('%Y-%m-%d %A')} [{day_total_duration}]\n")
        tableheader = """|Project|Task(s)|Duration|
|-------|-------|-------:|"""
        markdown_output.append(tableheader)
        markdown_output.extend(tasks_output)
        markdown_output.append(f"|TOTAL| |{day_total_duration}|\n")
        markdown_output.append("")

    return "\n".join(markdown_output)


#JSON
@time.get("/time/json")
async def get_timing_json(
    request: Request,
    start_date: str = Query(..., regex=r"\d{4}-\d{2}-\d{2}"),
    end_date: Optional[str] = Query(None, regex=r"\d{4}-\d{2}-\d{2}")
):

    # Fetch and process timing data
    start = localize_datetime(start_date)
    end = localize_datetime(end_date)
    timing_data = await fetch_and_prepare_timing_data(start, end)

    # Convert processed data to the required JSON structure
    json_data = process_json(timing_data)
    return JSONResponse(content=json_data)

def process_json(timing_data):
    structured_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for entry in timing_data:
        date_key = entry['start_date']  # Already in 'YYYY-MM-DD' format
        project_title = entry['project']['title'] if 'title' in entry['project'] else 'No Project'
        task_title = entry['title']

        structured_data[date_key][project_title][task_title].append(entry)

    return dict(structured_data)


# ROCKETMATTER CSV PARSING

def load_project_names(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)


def parse_input(fields, project_name_mappings, start_times_by_date):
    project_code = fields[3].strip()
    project_name = project_name_mappings.get(project_code, project_code)
    task_descriptions = fields[4].strip()
    billing_date_str = fields[6].strip()
    total_hours = float(fields[9].strip())

    billing_date = datetime.strptime(billing_date_str, "%m/%d/%Y").date()

    # If no start time is recorded for this billing_date, default to 8 AM
    if billing_date not in start_times_by_date:
        start_time = pacific.localize(datetime.combine(billing_date, datetime.min.time()).replace(hour=8))
    else:
        start_time = start_times_by_date[billing_date]

    # Normalize the task descriptions by converting line breaks and variations of task separators (],), (),)\s to standard form [,]
    task_descriptions = re.sub(r'(\)|\])(\s+|$)(?=\[|\(|[A-Za-z])', '],', task_descriptions)
    task_descriptions = re.sub(r'(\r?\n|\r)', ',', task_descriptions)

    # Regex pattern to match task descriptions along with their respective durations.
    task_pattern = re.compile(r'(.*?)[\[\(](\d+\.\d+)[\]\)]\s*,?')

    tasks_with_durations = task_pattern.findall(task_descriptions)

    tasks = []
    total_calc_hours = 0

    # Process tasks with explicit durations
    for task in tasks_with_durations:
        task_name, duration_hours = task[0].strip(' ,;'), float(task[1])
        task_name = task_name if task_name else "Undefined Task"
        tasks.append((task_name, duration_hours))
        total_calc_hours += duration_hours

    # If there are hours not accounted for, consider them for a task without a specific duration
    remainder = total_hours - total_calc_hours
    if remainder > 0:
        # Include non-specific task or "Undefined Task"
        non_duration_task = re.sub(task_pattern, '', task_descriptions).strip(' ,;')
        if not non_duration_task:  
            non_duration_task = "Undefined Task"
        tasks.append((non_duration_task, remainder))

    # If no specific task durations are found in the description, treat the entire description as one task
    if not tasks_with_durations:
        task_name = task_descriptions if task_descriptions else "Undefined Task"
        tasks.append((task_name, total_hours))

    json_entries = []
    for task_name, duration_hours in tasks:
        duration = timedelta(hours=duration_hours)
        end_time = start_time + duration
        entry = {
            "project": project_name,
            "Task": task_name,
            "Start_time": start_time.strftime("%Y-%m-%d %H:%M:%S-07:00"),
            "End_time": end_time.strftime("%Y-%m-%d %H:%M:%S-07:00")
        }
        json_entries.append(entry)
        start_time = end_time

    # Update the start time for the billing_date in the dictionary
    start_times_by_date[billing_date] = start_time

    return json_entries


async def post_time_entry_to_timing(entry):
    url = f"{TIMING_API_URL}/time-entries"  # The URL for posting time entries
    headers = {
        "Authorization": f"Bearer {TIMING_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        'X-Time-Zone': 'America/Los_Angeles'  # Set the timezone for the API request
    }
    data = {
        "start_date": entry["Start_time"],  # Format these according to the API's requirements
        "end_date": entry["End_time"],
        "project": entry["project"],
        "title": entry["Task"],
        "notes": "Automatically generated based on Rocketmatter reports.",
        "replace_existing": False
    }
    response = await httpx.post(url, headers=headers, json=data)
    return response.status_code, response.json()


@time.get("/time/flagemoji/{country_code}")
def flag_emoji(country_code: str):
    offset = 127397
    flag = ''.join(chr(ord(char) + offset) for char in country_code.upper())
    return {"emoji": flag}


@time.head("/time/")
async def read_root():
    return {}


@time.get("/time/")
async def root():
    return {"message": "Ring, ring, ring, ring, ring, ring, ring. \n\n. Banana phone."}

