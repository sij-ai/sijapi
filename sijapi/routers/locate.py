from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import requests
import json
import time
import pytz
import traceback
from datetime import datetime, timezone
from typing import Union, List
import asyncio
import pytz
import folium
import time as timer
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Union
from datetime import datetime, timedelta, time
from sijapi import LOCATION_OVERRIDES, TZ
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi.utilities import get_db_connection, haversine, localize_dt
# from osgeo import gdal
# import elevation

locate = APIRouter()


from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import requests

class Location(BaseModel):
    latitude: float
    longitude: float
    datetime: datetime
    elevation: Optional[float] = None
    altitude: Optional[float] = None
    zip: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    context: Optional[Dict[str, Any]] = None 
    class_: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    display_name: Optional[str] = None
    boundingbox: Optional[List[str]] = None
    amenity: Optional[str] = None
    house_number: Optional[str] = None
    road: Optional[str] = None
    quarter: Optional[str] = None
    neighbourhood: Optional[str] = None
    suburb: Optional[str] = None
    county: Optional[str] = None
    country_code: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }

def reverse_geocode(latitude: float, longitude: float) -> Optional[Location]:
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
    INFO(f"Calling Nominatim API at {url}")
    headers = {
        'User-Agent': 'sij.law/1.0 (sij@sij.law)',  # replace with your app name and email
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for unsuccessful requests
        data = response.json()
        
        address = data.get("address", {})
        
        location = Location(
            latitude=float(data.get("lat", latitude)),
            longitude=float(data.get("lon", longitude)),
            datetime=datetime.now(),  # You might want to adjust this based on your needs
            zip=address.get("postcode"),
            street=address.get("road"),
            city=address.get("city"),
            state=address.get("state"),
            country=address.get("country"),
            context={},  # Initialize with an empty dict, to be filled as needed
            class_=data.get("class"),
            type=data.get("type"),
            name=data.get("name"),
            display_name=data.get("display_name"),
            boundingbox=data.get("boundingbox"),
            amenity=address.get("amenity"),
            house_number=address.get("house_number"),
            road=address.get("road"),
            quarter=address.get("quarter"),
            neighbourhood=address.get("neighbourhood"),
            suburb=address.get("suburb"),
            county=address.get("county"),
            country_code=address.get("country_code")
        )
        
        INFO(f"Created Location object: {location}")

        return location
    
    except requests.exceptions.RequestException as e:
        ERR(f"Error: {e}")
        return None



## NOT YET IMPLEMENTED
async def geocode(zip_code: Optional[str] = None, latitude: Optional[float] = None, longitude: Optional[float] = None, city: Optional[str] = None, state: Optional[str] = None, country_code: str = 'US') -> Location:
    if (latitude is None or longitude is None) and (zip_code is None) and (city is None or state is None):
        ERR(f"Must provide sufficient information for geocoding!")
        return None
    
    try:
        # Establish the database connection
        conn = get_db_connection()
        
        # Build the SQL query based on the provided parameters
        query = "SELECT id, street, city, state, country, latitude, longitude, zip, elevation, datetime, date, ST_Distance(geom, ST_SetSRID(ST_MakePoint($1, $2), 4326)) AS distance FROM Locations"
        
        conditions = []
        params = []
        
        if latitude is not None and longitude is not None:
            conditions.append("ST_DWithin(geom, ST_SetSRID(ST_MakePoint($1, $2), 4326), 50000)")  # 50 km radius
            params.extend([longitude, latitude])
        
        if zip_code:
            conditions.append("zip = $3 AND country = $4")
            params.extend([zip_code, country_code])
        
        if city and state:
            conditions.append("city ILIKE $5 AND state ILIKE $6 AND country = $7")
            params.extend([city, state, country_code])
        
        if conditions:
            query += " WHERE " + " OR ".join(conditions)
        
        query += " ORDER BY distance LIMIT 1;"
        
        DEBUG(f"Executing query: {query} with params: {params}")
        
        # Execute the query with the provided parameters
        result = await conn.fetchrow(query, *params)
        
        # Close the connection
        await conn.close()
        
        if result:
            location_info = Location(
                latitude=result['latitude'],
                longitude=result['longitude'],
                datetime=result.get['datetime'],
                zip=result['zip'],
                street=result.get('street', ''),
                city=result['city'],
                state=result['state'],
                country=result['country'],
                elevation=result.get('elevation', 0),
                distance=result.get('distance')
            )
            DEBUG(f"Found location: {location_info}")
            return location_info
        else:
            DEBUG("No location found with provided parameters.")
            return Location()
    
    except Exception as e:
        ERR(f"Error occurred: {e}")
        raise Exception("An error occurred while processing your request")

def find_override_locations(lat: float, lon: float) -> Optional[str]:
    # Load the JSON file
    with open(LOCATION_OVERRIDES, 'r') as file:
        locations = json.load(file)
    
    closest_location = None
    closest_distance = float('inf')
    
    # Iterate through each location entry in the JSON
    for location in locations:
        loc_name = location.get("name")
        loc_lat = location.get("latitude")
        loc_lon = location.get("longitude")
        loc_radius = location.get("radius")
        
        # Calculate distance using haversine
        distance = haversine(lat, lon, loc_lat, loc_lon)
        
        # Check if the distance is within the specified radius
        if distance <= loc_radius:
            if distance < closest_distance:
                closest_distance = distance
                closest_location = loc_name
    
    return closest_location

def get_elevation(latitude, longitude):
    url = "https://api.open-elevation.com/api/v1/lookup"
    
    payload = {
        "locations": [
            {
                "latitude": latitude,
                "longitude": longitude
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for unsuccessful requests
        
        data = response.json()
        
        if "results" in data:
            elevation = data["results"][0]["elevation"]
            return elevation
        else:
            return None
    
    except requests.exceptions.RequestException as e:
        ERR(f"Error: {e}")
        return None


async def fetch_locations(start: datetime, end: datetime = None) -> List[Location]:
    start_datetime = localize_dt(start)
    
    if end is None:
        end_datetime = localize_dt(start_datetime.replace(hour=23, minute=59, second=59))
    else:
        end_datetime = localize_dt(end)
        if start_datetime.time() == datetime.min.time() and end.time() == datetime.min.time():
            end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
    
    DEBUG(f"Fetching locations between {start_datetime} and {end_datetime}")
    conn = await get_db_connection()
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
    
    DEBUG(f"Range locations query returned: {range_locations}")
    locations.extend(range_locations)
    
    if not locations and (end is None or start_datetime.date() == end.date()):
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
        
        DEBUG(f"Fallback query returned: {location_data}")
        if location_data:
            locations.append(location_data)
    
    await conn.close()

    DEBUG(f"Locations found: {locations}")

    # Sort location_data based on the datetime field in descending order
    sorted_locations = sorted(locations, key=lambda x: x['datetime'], reverse=True)
        
    # Create Location objects directly from the location data
    location_objects = [Location(
        latitude=loc['latitude'],
        longitude=loc['longitude'],
        datetime=loc['datetime'],
        elevation=loc.get('elevation'),
        city=loc.get('city'),
        state=loc.get('state'),
        zip=loc.get('zip'),
        street=loc.get('street'),
        context={
            'action': loc.get('action'),
            'device_type': loc.get('device_type'),
            'device_model': loc.get('device_model'),
            'device_name': loc.get('device_name'),
            'device_os': loc.get('device_os')
        }
    ) for loc in sorted_locations if loc['latitude'] is not None and loc['longitude'] is not None]

    return location_objects if location_objects else []

# Function to fetch the last location before the specified datetime
async def fetch_last_location_before(datetime: datetime) -> Optional[Location]:
    datetime = localize_dt(datetime)
    
    DEBUG(f"Fetching last location before {datetime}")
    conn = await get_db_connection()

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
        DEBUG(f"Last location found: {location_data}")
        return Location(**location_data)
    else:
        DEBUG("No location found before the specified datetime")
        return None



@locate.get("/map/start_date={start_date_str}&end_date={end_date_str}", response_class=HTMLResponse)
async def generate_map_endpoint(start_date_str: str, end_date_str: str):
    try:
        start_date = localize_dt(start_date_str)
        end_date = localize_dt(end_date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    html_content = await generate_map(start_date, end_date)
    return HTMLResponse(content=html_content)


@locate.get("/map", response_class=HTMLResponse)
async def generate_alltime_map_endpoint():
    try:
        start_date = localize_dt(datetime.fromisoformat("2022-01-01"))
        end_date =  localize_dt(datetime.now())
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
    for loc in locations:
        folium.Marker(
            location=[loc.latitude, loc.longitude],
            popup=f"{loc.city}, {loc.state}<br>Elevation: {loc.elevation}m<br>Date: {loc.datetime}",
            tooltip=f"{loc.city}, {loc.state}"
        ).add_to(m)

    # Save the map to an HTML file and return the HTML content
    map_html = "map.html"
    m.save(map_html)

    with open(map_html, 'r') as file:
        html_content = file.read()

    return html_content


async def post_location(location: Location):
    DEBUG(f"post_location called with {location.datetime}")
    conn = await get_db_connection()
    try:
        context = location.context or {}
        action = context.get('action', 'manual')
        device_type = context.get('device_type', 'Unknown')
        device_model = context.get('device_model', 'Unknown')
        device_name = context.get('device_name', 'Unknown')
        device_os = context.get('device_os', 'Unknown')
        
        # Parse and localize the datetime
        localized_datetime = localize_dt(location.datetime)
        
        await conn.execute('''
            INSERT INTO locations (datetime, location, city, state, zip, street, action, device_type, device_model, device_name, device_os)
            VALUES ($1, ST_SetSRID(ST_MakePoint($2, $3, $4), 4326), $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ''', localized_datetime, location.longitude, location.latitude, location.elevation, location.city, location.state, location.zip, location.street, action, device_type, device_model, device_name, device_os)
        await conn.close()
        INFO(f"Successfully posted location: {location.latitude}, {location.longitude} on {localized_datetime}")
        return {
            'datetime': localized_datetime,
            'latitude': location.latitude,
            'longitude': location.longitude,
            'city': location.city,
            'state': location.state,
            'zip': location.zip,
            'street': location.street,
            'elevation': location.elevation,
            'action': action,
            'device_type': device_type,
            'device_model': device_model,
            'device_name': device_name,
            'device_os': device_os
        }
    except Exception as e:
        ERR(f"Error posting location {e}")
        ERR(traceback.format_exc())
        return None

@locate.post("/locate")
async def post_locate_endpoint(locations: Union[Location, List[Location]]):
    responses = []

    if isinstance(locations, Location):
        locations = [locations]

    for location in locations:
        if not location.datetime:
            current_time = datetime.now(timezone.utc)
            location.datetime = current_time.isoformat()

        if not location.elevation:
            location.elevation = location.altitude if location.altitude else get_elevation(location.latitude, location.longitude)

        # Ensure context is a dictionary with default values if not provided
        if not location.context:
            location.context = {
                "action": "manual",
                "device_type": "Pythonista",
                "device_model": "Unknown",
                "device_name": "Unknown",
                "device_os": "Unknown"
            }

        DEBUG(f"datetime before localization: {location.datetime}")
        # Convert datetime string to timezone-aware datetime object
        location.datetime = localize_dt(location.datetime)
        DEBUG(f"datetime after localization: {location.datetime}")

        location_entry = await post_location(location)
        if location_entry:
            responses.append({"location_data": location_entry})  # Add weather data if necessary
        
        await asyncio.sleep(0.1)  # Use asyncio.sleep for async compatibility

    return {"message": "Locations and weather updated", "results": responses}



# GET endpoint to fetch the last location before the specified datetime
# @locate.get("/last_location", response_model=Union[Location, Dict[str, str]])
@locate.get("/locate", response_model=List[Location])
async def get_last_location() -> JSONResponse:
    query_datetime = datetime.now(TZ)
    DEBUG(f"Query_datetime: {query_datetime}")
    location = await fetch_last_location_before(query_datetime)
    if location:
        DEBUG(f"location: {location}")
        location_dict = location.model_dump()  # use model_dump instead of dict
        location_dict["datetime"] = location.datetime.isoformat()
        return JSONResponse(content=location_dict)
    else:
        return JSONResponse(content={"message": "No location found before the specified datetime"}, status_code=404)


@locate.get("/locate/{datetime_str}", response_model=List[Location])
async def get_locate(datetime_str: str, all: bool = False):
    try:
        date_time = localize_dt(datetime_str)
    except ValueError as e:
        ERR(f"Invalid datetime string provided: {datetime_str}")
        return ["ERROR: INVALID DATETIME PROVIDED. USE YYYYMMDDHHmmss or YYYYMMDD format."]
    
    locations = await fetch_locations(date_time)
    if not locations:
        raise HTTPException(status_code=404, detail="No nearby data found for this date and time")
        
    return locations if all else [locations[0]]






future_elevation = """
def get_elevation_srtm(latitude, longitude, srtm_file):
    try:
        # Open the SRTM dataset
        dataset = gdal.Open(srtm_file)
        
        # Get the geotransform and band information
        geotransform = dataset.GetGeoTransform()
        band = dataset.GetRasterBand(1)
        
        # Calculate the pixel coordinates from the latitude and longitude
        x = int((longitude - geotransform[0]) / geotransform[1])
        y = int((latitude - geotransform[3]) / geotransform[5])
        
        # Read the elevation value from the SRTM dataset
        elevation = band.ReadAsArray(x, y, 1, 1)[0][0]
        
        # Close the dataset
        dataset = None
        
        return elevation
    
    except Exception as e:
        ERR(f"Error: {e}")
        return None
"""

def get_elevation2(latitude: float, longitude: float) -> float:
    url = f"https://nationalmap.gov/epqs/pqs.php?x={longitude}&y={latitude}&units=Meters&output=json"

    try:
        response = requests.get(url)
        data = response.json()
        elevation = data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        return float(elevation)
    except Exception as e:
        # Handle exceptions (e.g., network errors, API changes) appropriately
        raise RuntimeError(f"Error getting elevation data: {str(e)}")
