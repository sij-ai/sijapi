'''
Uses Postgres/PostGIS for location tracking (data obtained via the companion mobile Pythonista scripts), and for geocoding purposes.
'''
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import yaml
from typing import List, Tuple, Union
import traceback
from datetime import datetime, timezone
from typing import Union, List
import folium
from zoneinfo import ZoneInfo
from dateutil.parser import parse as dateutil_parse
from typing import Optional, List, Union
from datetime import datetime
from sijapi import L, DB, TZ, GEO
from sijapi.classes import Location
from sijapi.utilities import haversine

loc = APIRouter()


async def dt(
    date_time: Union[str, int, datetime],
    tz: Union[str, ZoneInfo, None] = None
) -> datetime:
    try:
        # Convert integer (epoch time) to UTC datetime
        if isinstance(date_time, int):
            date_time = datetime.utcfromtimestamp(date_time).replace(tzinfo=timezone.utc)
            L.DEBUG(f"Converted epoch time {date_time} to UTC datetime object.")
        # Convert string to datetime if necessary
        elif isinstance(date_time, str):
            date_time = dateutil_parse(date_time)
            L.DEBUG(f"Converted string '{date_time}' to datetime object.")
        
        if not isinstance(date_time, datetime):
            raise ValueError(f"Input must be a string, integer (epoch time), or datetime object. What we received: {date_time}, type {type(date_time)}")

        # Ensure the datetime is timezone-aware (UTC if not specified)
        if date_time.tzinfo is None:
            date_time = date_time.replace(tzinfo=timezone.utc)
            L.DEBUG("Added UTC timezone to naive datetime.")

        # Handle provided timezone
        if tz is not None:
            if isinstance(tz, str): 
                if tz == "local":
                    last_loc = await get_timezone_without_timezone(date_time)
                    tz = await GEO.tz_at(last_loc.latitude, last_loc.longitude)
                    L.DEBUG(f"Using local timezone: {tz}")
                else:
                    try:
                        tz = ZoneInfo(tz)
                    except Exception as e:
                        L.ERR(f"Invalid timezone string '{tz}'. Error: {e}")
                        raise ValueError(f"Invalid timezone string: {tz}")
            elif isinstance(tz, ZoneInfo):
                pass  # tz is already a ZoneInfo object
            else:
                raise ValueError(f"What we needed: tz == 'local', a string, or a ZoneInfo object. What we got: tz, a {type(tz)}, == {tz})")
            
            # Convert to the provided or determined timezone
            date_time = date_time.astimezone(tz)
            L.DEBUG(f"Converted datetime to timezone: {tz}")
        
        return date_time
    except ValueError as e:
        L.ERR(f"Error in dt: {e}")
        raise
    except Exception as e:
        L.ERR(f"Unexpected error in dt: {e}")
        raise ValueError(f"Failed to process datetime: {e}")


async def get_timezone_without_timezone(date_time):
    # This is a bit convoluted because we're trying to solve the paradox of needing to know the location in order to determine the timezone, but needing the timezone to be certain we've got the right location if this datetime coincided with inter-timezone travel. Our imperfect solution is to use UTC for an initial location query to determine roughly where we were at the time, get that timezone, then check the location again using that timezone, and if this location is different from the one using UTC, get the timezone again usng it, otherwise use the one we already sourced using UTC.
            
    # Step 1: Use UTC as an interim timezone to query location
    interim_dt = date_time.replace(tzinfo=ZoneInfo("UTC"))
    interim_loc = await fetch_last_location_before(interim_dt)
    
    # Step 2: Get a preliminary timezone based on the interim location
    interim_tz = await GEO.tz_current((interim_loc.latitude, interim_loc.longitude))
    
    # Step 3: Apply this preliminary timezone and query location again
    query_dt = date_time.replace(tzinfo=ZoneInfo(interim_tz))
    query_loc = await fetch_last_location_before(query_dt)
    
    # Step 4: Get the final timezone, reusing interim_tz if location hasn't changed
    return interim_tz if query_loc == interim_loc else await GEO.tz_current(query_loc.latitude, query_loc.longitude)
            

async def get_last_location() -> Optional[Location]:
    query_datetime = datetime.now(TZ)
    L.DEBUG(f"Query_datetime: {query_datetime}")

    this_location = await fetch_last_location_before(query_datetime)

    if this_location:
        L.DEBUG(f"location: {this_location}")
        return this_location
    
    return None


async def fetch_locations(start: datetime, end: datetime = None) -> List[Location]:
    start_datetime = await dt(start)
    if end is None:
        end_datetime = await dt(start_datetime.replace(hour=23, minute=59, second=59))
    else:
        end_datetime = await dt(end)

    if start_datetime.time() == datetime.min.time() and end_datetime.time() == datetime.min.time():
        end_datetime = end_datetime.replace(hour=23, minute=59, second=59)

    L.DEBUG(f"Fetching locations between {start_datetime} and {end_datetime}")

    async with DB.get_connection() as conn:
        locations = []
        # Check for records within the specified datetime range
        range_locations = await conn.fetch('''
            SELECT id, datetime,
            ST_X(ST_AsText(location)::geometry) AS longitude,
            ST_Y(ST_AsText(location)::geometry) AS latitude,
            ST_Z(ST_AsText(location)::geometry) AS elevation,
            city, state, zip, street,
            action, device_type, device_model, device_name, device_os
            FROM locations
            WHERE datetime >= $1 AND datetime <= $2
            ORDER BY datetime DESC
            ''', start_datetime.replace(tzinfo=None), end_datetime.replace(tzinfo=None))
        
        L.DEBUG(f"Range locations query returned: {range_locations}")
        locations.extend(range_locations)

        if not locations and (end is None or start_datetime.date() == end_datetime.date()):
            location_data = await conn.fetchrow('''
                SELECT id, datetime,
                ST_X(ST_AsText(location)::geometry) AS longitude,
                ST_Y(ST_AsText(location)::geometry) AS latitude,
                ST_Z(ST_AsText(location)::geometry) AS elevation,
                city, state, zip, street,
                action, device_type, device_model, device_name, device_os
                FROM locations
                WHERE datetime < $1
                ORDER BY datetime DESC
                LIMIT 1
                ''', start_datetime.replace(tzinfo=None))
            
            L.DEBUG(f"Fallback query returned: {location_data}")
            if location_data:
                locations.append(location_data)

    L.DEBUG(f"Locations found: {locations}")

    # Sort location_data based on the datetime field in descending order
    sorted_locations = sorted(locations, key=lambda x: x['datetime'], reverse=True)

    # Create Location objects directly from the location data
    location_objects = [
        Location(
            latitude=location['latitude'],
            longitude=location['longitude'],
            datetime=location['datetime'],
            elevation=location.get('elevation'),
            city=location.get('city'),
            state=location.get('state'),
            zip=location.get('zip'),
            street=location.get('street'),
            context={
                'action': location.get('action'),
                'device_type': location.get('device_type'),
                'device_model': location.get('device_model'),
                'device_name': location.get('device_name'),
                'device_os': location.get('device_os')
            }
        ) for location in sorted_locations if location['latitude'] is not None and location['longitude'] is not None
    ]

    return location_objects if location_objects else []

# Function to fetch the last location before the specified datetime
async def fetch_last_location_before(datetime: datetime) -> Optional[Location]:
    datetime = await dt(datetime)
    
    L.DEBUG(f"Fetching last location before {datetime}")

    async with DB.get_connection() as conn:

        location_data = await conn.fetchrow('''
            SELECT id, datetime,
                ST_X(ST_AsText(location)::geometry) AS longitude,
                ST_Y(ST_AsText(location)::geometry) AS latitude,
                ST_Z(ST_AsText(location)::geometry) AS elevation,
                city, state, zip, street, country,
                action
            FROM locations
            WHERE datetime < $1
            ORDER BY datetime DESC
            LIMIT 1
        ''', datetime.replace(tzinfo=None))
        
        await conn.close()

        if location_data:
            L.DEBUG(f"Last location found: {location_data}")
            return Location(**location_data)
        else:
            L.DEBUG("No location found before the specified datetime")
            return None

@loc.get("/map/start_date={start_date_str}&end_date={end_date_str}", response_class=HTMLResponse)
async def generate_map_endpoint(start_date_str: str, end_date_str: str):
    try:
        start_date = await dt(start_date_str)
        end_date = await dt(end_date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    html_content = await generate_map(start_date, end_date)
    return HTMLResponse(content=html_content)


@loc.get("/map", response_class=HTMLResponse)
async def generate_alltime_map_endpoint():
    try:
        start_date = await dt(datetime.fromisoformat("2022-01-01"))
        end_date =  dt(datetime.now())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    html_content = await generate_map(start_date, end_date)
    return HTMLResponse(content=html_content)
    

async def generate_map(start_date: datetime, end_date: datetime):
    locations = await fetch_locations(start_date, end_date)
    if not locations:
        raise HTTPException(status_code=404, detail="No locations found for the given date range")

    # Create a folium map centered around the first location
    map_center = [locations[0].latitude, locations[0].longitude]
    m = folium.Map(location=map_center, zoom_start=5)

    # Add markers for each location
    for location in locations:
        folium.Marker(
            location=[location.latitude, location.longitude],
            popup=f"{location.city}, {location.state}<br>Elevation: {location.elevation}m<br>Date: {location.datetime}",
            tooltip=f"{location.city}, {location.state}"
        ).add_to(m)

    # Save the map to an HTML file and return the HTML content
    map_html = "map.html"
    m.save(map_html)

    with open(map_html, 'r') as file:
        html_content = file.read()

    return html_content

async def post_location(location: Location):
    if not location.datetime:
        L.DEBUG(f"location appears to be missing datetime: {location}")
    else:
        L.DEBUG(f"post_location called with {location.datetime}")

    async with DB.get_connection() as conn:
        try:
            context = location.context or {}
            action = context.get('action', 'manual')
            device_type = context.get('device_type', 'Unknown')
            device_model = context.get('device_model', 'Unknown')
            device_name = context.get('device_name', 'Unknown')
            device_os = context.get('device_os', 'Unknown')
            
            # Parse and localize the datetime
            localized_datetime = await dt(location.datetime)

            await conn.execute('''
                INSERT INTO locations (
                    datetime, location, city, state, zip, street, action, device_type, device_model, device_name, device_os,
                    class_, type, name, display_name, amenity, house_number, road, quarter, neighbourhood, 
                    suburb, county, country_code, country
                )
                VALUES ($1, ST_SetSRID(ST_MakePoint($2, $3, $4), 4326), $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, 
                        $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
            ''', localized_datetime, location.longitude, location.latitude, location.elevation, location.city, location.state, 
                    location.zip, location.street, action, device_type, device_model, device_name, device_os, 
                    location.class_, location.type, location.name, location.display_name, 
                    location.amenity, location.house_number, location.road, location.quarter, location.neighbourhood, 
                    location.suburb, location.county, location.country_code, location.country)
                
            await conn.close()
            L.INFO(f"Successfully posted location: {location.latitude}, {location.longitude}, {location.elevation} on {localized_datetime}")
            return {
                'datetime': localized_datetime,
                'latitude': location.latitude,
                'longitude': location.longitude,
                'elevation': location.elevation,
                'city': location.city,
                'state': location.state,
                'zip': location.zip,
                'street': location.street,
                'action': action,
                'device_type': device_type,
                'device_model': device_model,
                'device_name': device_name,
                'device_os': device_os,
                'class_': location.class_,
                'type': location.type,
                'name': location.name,
                'display_name': location.display_name,
                'amenity': location.amenity,
                'house_number': location.house_number,
                'road': location.road,
                'quarter': location.quarter,
                'neighbourhood': location.neighbourhood,
                'suburb': location.suburb,
                'county': location.county,
                'country_code': location.country_code,
                'country': location.country
            }
        except Exception as e:
            L.ERR(f"Error posting location {e}")
            L.ERR(traceback.format_exc())
            return None


@loc.post("/locate")
async def post_locate_endpoint(locations: Union[Location, List[Location]]):
    if isinstance(locations, Location):
        locations = [locations]

    # Prepare locations
    for lcn in locations:
        if not lcn.datetime:
            tz = await GEO.tz_at(lcn.latitude, lcn.longitude)
            lcn.datetime = datetime.now(ZoneInfo(tz)).isoformat()
        
        if not lcn.context:
            lcn.context = {
                "action": "missing",
                "device_type": "API",
                "device_model": "Unknown",
                "device_name": "Unknown",
                "device_os": "Unknown"
            }
        L.DEBUG(f"Location received for processing: {lcn}")

    geocoded_locations = await GEO.code(locations)

    responses = []
    if isinstance(geocoded_locations, List):
        for location in geocoded_locations:
            L.DEBUG(f"Final location to be submitted to database: {location}")
            location_entry = await post_location(location)
            if location_entry:
                responses.append({"location_data": location_entry})
            else:
                L.WARN(f"Posting location to database appears to have failed.")
    else:
        L.DEBUG(f"Final location to be submitted to database: {geocoded_locations}")
        location_entry = await post_location(geocoded_locations)
        if location_entry:
            responses.append({"location_data": location_entry})
        else:
            L.WARN(f"Posting location to database appears to have failed.")

    return {"message": "Locations and weather updated", "results": responses}


@loc.get("/locate", response_model=Location)
async def get_last_location_endpoint() -> JSONResponse:
    this_location = await get_last_location()

    if this_location:
        location_dict = this_location.model_dump()
        location_dict["datetime"] = this_location.datetime.isoformat()
        return JSONResponse(content=location_dict)
    else:
        raise HTTPException(status_code=404, detail="No location found before the specified datetime")

@loc.get("/locate/{datetime_str}", response_model=List[Location])
async def get_locate(datetime_str: str, all: bool = False):
    try:
        date_time = await dt(datetime_str)
    except ValueError as e:
        L.ERR(f"Invalid datetime string provided: {datetime_str}")
        return ["ERROR: INVALID DATETIME PROVIDED. USE YYYYMMDDHHmmss or YYYYMMDD format."]
    
    locations = await fetch_locations(date_time)
    if not locations:
        raise HTTPException(status_code=404, detail="No nearby data found for this date and time")
        
    return locations if all else [locations[0]]

