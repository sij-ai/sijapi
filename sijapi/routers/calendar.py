'''
Calendar module using macOS Calendars and/or Microsoft 365 via its Graph API.
Depends on: 
  LOGGER, ICAL_TOGGLE, ICALENDARS, MS365_TOGGLE, MS365_CLIENT_ID, MS365_SECRET, MS365_AUTHORITY_URL, MS365_SCOPE, MS365_REDIRECT_PATH, MS365_TOKEN_PATH
'''
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
import httpx
import json
import os
import time
from dateutil.parser import isoparse as parse_iso
import threading
from typing import Dict, List, Any
from datetime import datetime, timedelta
from Foundation import NSDate, NSRunLoop
import EventKit as EK
from sijapi import ICAL_TOGGLE, ICALENDARS, MS365_TOGGLE, MS365_CLIENT_ID, MS365_SECRET, MS365_AUTHORITY_URL, MS365_SCOPE, MS365_REDIRECT_PATH, MS365_TOKEN_PATH
from sijapi.utilities import localize_datetime, localize_datetime
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL

calendar = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
timeout = httpx.Timeout(12)

if MS365_TOGGLE is True:
    CRITICAL(f"Visit https://api.sij.ai/o365/login to obtain your Microsoft 365 authentication token.")

    @calendar.get("/o365/login")
    async def login():
        DEBUG(f"Received request to /o365/login")
        DEBUG(f"SCOPE: {MS365_SCOPE}")
        if not MS365_SCOPE:
            ERR("No scopes defined for authorization.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No scopes defined for authorization."
            )
        authorization_url = f"{MS365_AUTHORITY_URL}/oauth2/v2.0/authorize?client_id={MS365_CLIENT_ID}&response_type=code&redirect_uri={MS365_REDIRECT_PATH}&scope={'+'.join(MS365_SCOPE)}"
        INFO(f"Redirecting to authorization URL: {authorization_url}")
        return RedirectResponse(authorization_url)

    @calendar.get("/o365/oauth_redirect")
    async def oauth_redirect(code: str = None, error: str = None):
        INFO(f"Received request to /o365/oauth_redirect")
        if error:
            ERR(f"OAuth2 Error: {error}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth2 Error"
            )
        INFO(f"Requesting token with authorization code: {code}")
        token_url = f"{MS365_AUTHORITY_URL}/oauth2/v2.0/token"
        data = {
            "client_id": MS365_CLIENT_ID,
            "client_secret": MS365_SECRET,
            "code": code,
            "redirect_uri": MS365_REDIRECT_PATH,
            "grant_type": "authorization_code"
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(token_url, data=data)
        DEBUG(f"Token endpoint response status code: {response.status_code}")
        INFO(f"Token endpoint response text: {response.text}")
        result = response.json()
        if 'access_token' in result:
            await save_token(result)
            INFO("Access token obtained successfully")
            return {"message": "Access token stored successfully"}
        else:
            CRITICAL(f"Failed to obtain access token. Response: {result}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to obtain access token"
            )

    @calendar.get("/o365/me")
    async def read_items():
        INFO(f"Received request to /o365/me")
        token = await load_token()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Access token not found",
            )
        graph_url = "https://graph.microsoft.com/v1.0/me"
        headers = {"Authorization": f"Bearer {token['access_token']}"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(graph_url, headers=headers)
        if response.status_code == 200:
            user = response.json()
            INFO(f"User retrieved: {user}")
            return user
        else:
            ERR("Invalid or expired token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
    async def save_token(token):
        DEBUG(f"Saving token: {token}")
        try:
            token["expires_at"] = int(time.time()) + token["expires_in"]
            with open(MS365_TOKEN_PATH, "w") as file:
                json.dump(token, file)
                DEBUG(f"Saved token to {MS365_TOKEN_PATH}")
        except Exception as e:
            ERR(f"Failed to save token: {e}")

    async def load_token():
        if os.path.exists(MS365_TOKEN_PATH):
            try:
                with open(MS365_TOKEN_PATH, "r") as file:
                    token = json.load(file)
            except FileNotFoundError:
                ERR("Token file not found.")
                return None
            except json.JSONDecodeError:
                ERR("Failed to decode token JSON")
                return None
            
            if token:
                token["expires_at"] = int(time.time()) + token["expires_in"]
                DEBUG(f"Loaded token: {token}")  # Add this line to log the loaded token
                return token
            else:
                DEBUG("No token found.")
                return None
        else:
            ERR(f"No file found at {MS365_TOKEN_PATH}")
            return None


    async def is_token_expired(token):
        if "expires_at" not in token:
            return True  # Treat missing expiration time as expired token
        expiry_time = datetime.fromtimestamp(token["expires_at"])
        return expiry_time <= datetime.now()

    async def is_token_expired2(token):
        graph_url = "https://graph.microsoft.com/v1.0/me"
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(graph_url, headers=headers)
        return response.status_code == 401

    async def get_new_token_with_refresh_token(refresh_token):
        token_url = f"{MS365_AUTHORITY_URL}/oauth2/v2.0/token"
        data = {
            "client_id": MS365_CLIENT_ID,
            "client_secret": MS365_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "scope": " ".join(MS365_SCOPE),
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(token_url, data=data)
        result = response.json()
        if "access_token" in result:
            INFO("Access token refreshed successfully")
            return result
        else:
            ERR("Failed to refresh access token")
            return None


    async def refresh_token():
        token = await load_token()
        if not token:
            ERR("No token found in storage")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No token found",
            )

        if 'refresh_token' not in token:
            ERR("Refresh token not found in the loaded token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not found",
            )

        refresh_token = token['refresh_token']
        DEBUG("Found refresh token, attempting to refresh access token")

        new_token = await get_new_token_with_refresh_token(refresh_token)
        
        if new_token:
            await save_token(new_token)
            INFO("Token refreshed and saved successfully")
        else:
            ERR("Failed to refresh token")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to refresh token",
            )


def get_calendar_ids() -> Dict[str, str]:
    event_store = EK.EKEventStore.alloc().init()
    all_calendars = event_store.calendarsForEntityType_(0)  # 0 corresponds to EKEntityTypeEvent

    calendar_identifiers = {
        calendar.title() : calendar.calendarIdentifier() for calendar in all_calendars
    }
    INFO(f"{calendar_identifiers}")
    return calendar_identifiers

# Helper to convert datetime to NSDate
def datetime_to_nsdate(dt: datetime) -> NSDate:
    return NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())


@calendar.get("/events")
async def get_events_endpoint(start_date: str, end_date: str):
    start_dt = localize_datetime(start_date)
    end_dt = localize_datetime(end_date)
    datetime.strptime(start_date, "%Y-%m-%d") or datetime.now()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") or datetime.now()
    response = await get_events(start_dt, end_dt)
    return JSONResponse(content=response, status_code=200)

async def get_events(start_dt: datetime, end_dt: datetime) -> List:
    combined_events = []
    if MS365_TOGGLE:
        ms_events = await get_ms365_events(start_dt, end_dt)
        combined_events.extend(ms_events)  # Use extend instead of append
    
    if ICAL_TOGGLE:
        calendar_ids = ICALENDARS  
        macos_events = get_macos_calendar_events(start_dt, end_dt, calendar_ids)
        combined_events.extend(macos_events)  # Use extend instead of append
    
    parsed_events = await parse_calendar_for_day(start_dt, end_dt, combined_events)
    return parsed_events


def get_macos_calendar_events(start_date: datetime, end_date: datetime, calendar_ids: List[str] = None) -> List[Dict]:
    event_store = EK.EKEventStore.alloc().init()

    # Request access to EventKit
    def request_access() -> bool:
        access_granted = []

        def completion_handler(granted, error):
            if error is not None:
                ERR(f"Error: {error}")
            access_granted.append(granted)
            # Notify the main thread that the completion handler has executed
            with access_granted_condition:
                access_granted_condition.notify()

        access_granted_condition = threading.Condition()
        with access_granted_condition:
            event_store.requestAccessToEntityType_completion_(0, completion_handler)  # 0 corresponds to EKEntityTypeEvent
            # Wait for the completion handler to be called
            access_granted_condition.wait(timeout=10)
            # Verify that the handler was called and access_granted is not empty
            if access_granted:
                return access_granted[0]
            else:
                ERR("Request access timed out or failed")
                return False

    if not request_access():
        ERR("Access to calendar data was not granted")
        return []

    ns_start_date = datetime_to_nsdate(start_date)
    ns_end_date = datetime_to_nsdate(end_date)

    # Retrieve all calendars
    all_calendars = event_store.calendarsForEntityType_(0)  # 0 corresponds to EKEntityTypeEvent
    if calendar_ids:
        selected_calendars = [cal for cal in all_calendars if cal.calendarIdentifier() in calendar_ids]
    else:
        selected_calendars = all_calendars

    # Filtering events by selected calendars
    predicate = event_store.predicateForEventsWithStartDate_endDate_calendars_(ns_start_date, ns_end_date, selected_calendars)
    events = event_store.eventsMatchingPredicate_(predicate)

    event_list = []
    for event in events:
        # Check if event.attendees() returns None
        if event.attendees():
            attendees = [{'name': att.name(), 'email': att.emailAddress()} for att in event.attendees() if att.emailAddress()]
        else:
            attendees = []

        # Format the start and end dates properly
        start_date_str = event.startDate().descriptionWithLocale_(None)
        end_date_str = event.endDate().descriptionWithLocale_(None)

        event_data = {
            "subject": event.title(),
            "id": event.eventIdentifier(),
            "start": start_date_str,
            "end": end_date_str,
            "bodyPreview": event.notes() if event.notes() else '',
            "attendees": attendees,
            "location": event.location() if event.location() else '',
            "onlineMeetingUrl": '',  # Defaulting to empty as macOS EventKit does not provide this
            "showAs": 'busy',  # Default to 'busy'
            "isAllDay": event.isAllDay()
        }

        event_list.append(event_data)

    return event_list

async def get_ms365_events(start_date: datetime, end_date: datetime):
    token = await load_token()
    if token:
        if await is_token_expired(token):
            await refresh_token()
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token not found",
        )
    # this looks like it might need updating to use tz-aware datetimes converted to UTC...
    graph_url = f"https://graph.microsoft.com/v1.0/me/events?$filter=start/dateTime ge '{start_date}T00:00:00' and end/dateTime le '{end_date}T23:59:59'"
    headers = {
        "Authorization": f"Bearer {token['access_token']}",
        "Prefer": 'outlook.timezone="Pacific Standard Time"',
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(graph_url, headers=headers)

    if response.status_code != 200:
        ERR("Failed to retrieve events from Microsoft 365")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve events",
        )

    ms_events = response.json().get("value", [])
    return ms_events


async def parse_calendar_for_day(range_start: datetime, range_end: datetime, events: List[Dict[str, Any]]):
    range_start = localize_datetime(range_start)
    range_end = localize_datetime(range_end)
    event_list = []

    for event in events:
        INFO(f"Event: {event}")
        start_str = event.get('start')
        end_str = event.get('end')

        if isinstance(start_str, dict):
            start_str = start_str.get('dateTime')
        else:
            INFO(f"Start date string not a dict")

        if isinstance(end_str, dict):
            end_str = end_str.get('dateTime')
        else:
            INFO(f"End date string not a dict")

        try:
            start_date = localize_datetime(start_str) if start_str else None
        except (ValueError, TypeError) as e:
            ERR(f"Invalid start date format: {start_str}, error: {e}")
            continue

        try:
            end_date = localize_datetime(end_str) if end_str else None
        except (ValueError, TypeError) as e:
            ERR(f"Invalid end date format: {end_str}, error: {e}")
            continue

        DEBUG(f"Comparing {start_date} with range {range_start} to {range_end}")

        if start_date:
            # Ensure start_date is timezone-aware
            start_date = localize_datetime(start_date)
            
            # If end_date is not provided, assume it's the same as start_date
            if not end_date:
                end_date = start_date
            else:
                end_date = localize_datetime(end_date)
            
            # Check if the event overlaps with the given range
            if (start_date < range_end) and (end_date > range_start):
                attendees = [{'name': att['name'], 'email': att['email']} for att in event.get('attendees', []) if 'name' in att and 'email' in att]
                location = event.get('location', '')
                if isinstance(location, dict):
                    location = location.get('displayName', '')
                
                event_data = {
                    "name": event.get('subject', ''),
                    "uid": event.get('id', ''),
                    "start": start_date.strftime('%H:%M'),
                    "end": end_date.strftime('%H:%M') if end_date else '',
                    "description": event.get('bodyPreview', ''),
                    "attendees": attendees,
                    "location": location,
                    "url": event.get('onlineMeetingUrl', ''),
                    "busystatus": event.get('showAs', ''),
                    "busy": event.get('showAs', '') in ['busy', 'tentative'],
                    "all_day": event.get('isAllDay', False)
                }
                INFO(f"Event_data: {event_data}")
                event_list.append(event_data)
            else:
                DEBUG(f"Event outside of specified range: {start_date} to {end_date}")
        else:
            ERR(f"Invalid or missing start date for event: {event.get('id', 'Unknown ID')}")

    return event_list