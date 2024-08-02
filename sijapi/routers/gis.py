'''
Uses Postgres/PostGIS for location tracking (data obtained via the companion mobile Pythonista scripts), and for geocoding purposes.
'''
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import random
from pathlib import Path
import traceback
from datetime import datetime, timezone
from typing import Union, List
import folium
from folium.plugins import HeatMap, MarkerCluster, Search
from folium.plugins import Fullscreen, MiniMap, MousePosition, Geocoder, Draw, MeasureControl
from zoneinfo import ZoneInfo
from dateutil.parser import parse as dateutil_parse
from typing import Optional, List, Union
from sijapi import L, API, TZ, GEO
from sijapi.classes import Location
from sijapi.utilities import haversine, assemble_journal_path

gis = APIRouter()
logger = L.get_module_logger("gis")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

async def dt(
    date_time: Union[str, int, datetime],
    tz: Union[str, ZoneInfo, None] = None
) -> datetime:
    try:
        # Convert integer (epoch time) to UTC datetime
        if isinstance(date_time, int):
            date_time = datetime.fromtimestamp(date_time, tz=timezone.utc)
            debug(f"Converted epoch time {date_time} to UTC datetime object.")

        # Convert string to datetime if necessary
        elif isinstance(date_time, str):
            date_time = dateutil_parse(date_time)
            debug(f"Converted string '{date_time}' to datetime object.")
        
        if not isinstance(date_time, datetime):
            raise ValueError(f"Input must be a string, integer (epoch time), or datetime object. What we received: {date_time}, type {type(date_time)}")

        # Ensure the datetime is timezone-aware (UTC if not specified)
        if date_time.tzinfo is None:
            date_time = date_time.replace(tzinfo=timezone.utc)
            debug("Added UTC timezone to naive datetime.")

        # Handle provided timezone
        if tz is not None:
            if isinstance(tz, str): 
                if tz == "local":
                    last_loc = await get_timezone_without_timezone(date_time)
                    tz = await GEO.tz_at(last_loc.latitude, last_loc.longitude)
                    debug(f"Using local timezone: {tz}")
                else:
                    try:
                        tz = ZoneInfo(tz)
                    except Exception as e:
                        err(f"Invalid timezone string '{tz}'. Error: {e}")
                        raise ValueError(f"Invalid timezone string: {tz}")
            elif isinstance(tz, ZoneInfo):
                pass  # tz is already a ZoneInfo object
            else:
                raise ValueError(f"What we needed: tz == 'local', a string, or a ZoneInfo object. What we got: tz, a {type(tz)}, == {tz})")
            
            # Convert to the provided or determined timezone
            date_time = date_time.astimezone(tz)
            debug(f"Converted datetime to timezone: {tz}")
        
        return date_time
    except ValueError as e:
        err(f"Error in dt: {e}")
        raise
    except Exception as e:
        err(f"Unexpected error in dt: {e}")
        raise ValueError(f"Failed to process datetime: {e}")


async def get_timezone_without_timezone(date_time):
    # This is a bit convoluted because we're trying to solve the paradox of needing to 
    # know the location in order to determine the timezone, but needing the timezone to be 
    # certain we've chosen the correct location for a provided timezone-naive datetime 
    # (relevant, e.g., if this datetime coincided with inter-timezone travel). 
    # Our imperfect solution is to use UTC for an initial location query to determine 
    # roughly where we were at the time, get that timezone, then check the location again 
    # applying that timezone to the provided datetime. If the location changed between the
    # datetime in UTC and the localized datetime, we'll use the new location's timezone;
    # otherwise we'll use the timezone we sourced from the UTC timezone query. But at the
    # end of the day it's entirely possible to spend the end of the day twice in two different
    # timezones (or none!), so this is a best-effort solution.

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
    debug(f"Query_datetime: {query_datetime}")

    this_location = await fetch_last_location_before(query_datetime)

    if this_location:
        debug(f"location: {this_location}")
        return this_location
    
    return None

async def fetch_locations(start: Union[str, int, datetime], end: Union[str, int, datetime, None] = None) -> List[Location]:
    start_datetime = await dt(start)
    if end is None:
        end_datetime = await dt(start_datetime.replace(hour=23, minute=59, second=59))
    else:
        end_datetime = await dt(end) if not isinstance(end, datetime) else end

    if start_datetime.time() == datetime.min.time() and end_datetime.time() == datetime.min.time():
        end_datetime = await dt(end_datetime.replace(hour=23, minute=59, second=59))

    debug(f"Fetching locations between {start_datetime} and {end_datetime}")

    query = '''
        SELECT id, datetime,
        ST_X(ST_AsText(location)::geometry) AS longitude,
        ST_Y(ST_AsText(location)::geometry) AS latitude,
        ST_Z(ST_AsText(location)::geometry) AS elevation,
        city, state, zip, street,
        action, device_type, device_model, device_name, device_os
        FROM locations
        WHERE datetime >= $1 AND datetime <= $2
        ORDER BY datetime DESC
    '''
    
    locations = await API.execute_read_query(query, start_datetime.replace(tzinfo=None), end_datetime.replace(tzinfo=None), table_name="locations")
    
    debug(f"Range locations query returned: {locations}")

    if not locations and (end is None or start_datetime.date() == end_datetime.date()):
        fallback_query = '''
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
        '''
        location_data = await API.execute_read_query(fallback_query, start_datetime.replace(tzinfo=None), table_name="locations")
        debug(f"Fallback query returned: {location_data}")
        if location_data:
            locations = location_data

    debug(f"Locations found: {locations}")

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

async def fetch_last_location_before(datetime: datetime) -> Optional[Location]:
    datetime = await dt(datetime)
    
    debug(f"Fetching last location before {datetime}")

    query = '''
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
    '''
    
    location_data = await API.execute_read_query(query, datetime.replace(tzinfo=None), table_name="locations")

    if location_data:
        debug(f"Last location found: {location_data[0]}")
        return Location(**location_data[0])
    else:
        debug("No location found before the specified datetime")
        return None

@gis.get("/map", response_class=HTMLResponse)
async def generate_map_endpoint(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    max_points: int = Query(32767, description="Maximum number of points to display")
):
    try:
        if start_date and end_date:
            start_date = await dt(start_date)
            end_date = await dt(end_date)
        else:
            start_date, end_date = await get_date_range()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    info(f"Generating map for {start_date} to {end_date}")
    html_content = await generate_map(start_date, end_date, max_points)
    return HTMLResponse(content=html_content)

async def get_date_range():
    query = "SELECT MIN(datetime) as min_date, MAX(datetime) as max_date FROM locations"
    row = await API.execute_read_query(query, table_name="locations")
    if row and row[0]['min_date'] and row[0]['max_date']:
        return row[0]['min_date'], row[0]['max_date']
    else:
        return datetime(2022, 1, 1), datetime.now()

async def generate_and_save_heatmap(
    start_date: Union[str, int, datetime],
    end_date: Optional[Union[str, int, datetime]] = None,
    output_path: Optional[Path] = None
) -> Path:
    """
Generate a heatmap for the given date range and save it as a PNG file using Folium.

:param start_date: The start date for the map (or the only date if end_date is not provided)
:param end_date: The end date for the map (optional)
:param output_path: The path to save the PNG file (optional)
:return: The path where the PNG file was saved
    """
    try:
        start_date = await dt(start_date)
        if end_date:
            end_date = await dt(end_date)
        else:
            end_date = start_date.replace(hour=23, minute=59, second=59)

        # Fetch locations
        locations = await fetch_locations(start_date, end_date)
        if not locations:
            raise ValueError("No locations found for the given date range")

        # Create map
        m = folium.Map()

        # Prepare heatmap data
        heat_data = [[loc.latitude, loc.longitude] for loc in locations]

        # Add heatmap layer
        HeatMap(heat_data).add_to(m)

        # Fit the map to the bounds of all locations
        bounds = [
            [min(loc.latitude for loc in locations), min(loc.longitude for loc in locations)],
            [max(loc.latitude for loc in locations), max(loc.longitude for loc in locations)]
        ]
        m.fit_bounds(bounds)

        # Generate output path if not provided
        if output_path is None:
            output_path, relative_path = assemble_journal_path(end_date, filename="map", extension=".png", no_timestamp=True)

        # Save the map as PNG
        m.save(str(output_path))

        info(f"Heatmap saved as PNG: {output_path}")
        return output_path

    except Exception as e:
        err(f"Error generating and saving heatmap: {str(e)}")
        raise

async def generate_map(start_date: datetime, end_date: datetime, max_points: int):
    locations = await fetch_locations(start_date, end_date)
    if not locations:
        raise HTTPException(status_code=404, detail="No locations found for the given date range")

    info(f"Found {len(locations)} locations for the given date range")

    if len(locations) > max_points:
        locations = random.sample(locations, max_points)

    map_center = [sum(loc.latitude for loc in locations) / len(locations),
                  sum(loc.longitude for loc in locations) / len(locations)]
    m = folium.Map(location=map_center, zoom_start=5)

    folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
    folium.TileLayer(
        tiles='https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}',
        attr='USGS The National Map',
        name='USGS Topo'
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri World Topo'
    ).add_to(m)

    folium.TileLayer('cartodbdark_matter', name='Dark Mode').add_to(m)

    draw = Draw(
        draw_options={
            'polygon': True,
            'rectangle': True,
            'circle': True,
            'marker': True,
            'circlemarker': False,
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)

    MeasureControl(
        position='topright', 
        primary_length_unit='kilometers', 
        secondary_length_unit='miles', 
        primary_area_unit='sqmeters', 
        secondary_area_unit='acres'
    ).add_to(m)

    m.get_root().html.add_child(folium.Element("""
<script>
var drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);
map.on(L.Draw.Event.CREATED, function (event) {
    var layer = event.layer;
    drawnItems.addLayer(layer);
    var shape = layer.toGeoJSON();
    var points = [];
    markerCluster.eachLayer(function (marker) {
        if (turf.booleanPointInPolygon(marker.toGeoJSON(), shape)) {
            points.push(marker.getLatLng());
        }
    });
    if (points.length > 0) {
        alert('Selected ' + points.length + ' points');
        console.log(points);
    }
});
</script>
    """))

    # Add marker cluster
    marker_cluster = MarkerCluster(name="Markers").add_to(m)

    # Prepare data for heatmap
    heat_data = [[loc.latitude, loc.longitude] for loc in locations]

    # Add heatmap
    HeatMap(heat_data, name="Heatmap").add_to(m)

    # Add markers to cluster
    for location in locations:
        popup_content = f"""
        {location.city}, {location.state}<br>
        Elevation: {location.elevation}m<br>
        Date: {location.datetime}<br>
        Action: {location.context.get('action', 'N/A')}<br>
        Device: {location.context.get('device_name', 'N/A')} ({location.context.get('device_model', 'N/A')})
        """
        folium.Marker(
            location=[location.latitude, location.longitude],
            popup=popup_content,
            tooltip=f"{location.city}, {location.state}"
        ).add_to(marker_cluster)

    # Add controls
    Fullscreen().add_to(m)
    MiniMap().add_to(m)
    MousePosition().add_to(m)
    Geocoder().add_to(m)
    Draw().add_to(m)

    # Add search functionality
    Search(
        layer=marker_cluster,
        geom_type='Point',
        placeholder='Search for a location',
        collapsed=False,
        search_label='city'
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m.get_root().render()

async def post_location(location: Location):
    try:
        context = location.context or {}
        action = context.get('action', 'manual')
        device_type = context.get('device_type', 'Unknown')
        device_model = context.get('device_model', 'Unknown')
        device_name = context.get('device_name', 'Unknown')
        device_os = context.get('device_os', 'Unknown')
        
        # Parse and localize the datetime
        localized_datetime = await dt(location.datetime)

        query = '''
            INSERT INTO locations (
                datetime, location, city, state, zip, street, action, device_type, device_model, device_name, device_os,
                class_, type, name, display_name, amenity, house_number, road, quarter, neighbourhood, 
                suburb, county, country_code, country
            )
            VALUES ($1, ST_SetSRID(ST_MakePoint($2, $3, $4), 4326), $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, 
                    $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
        '''
        
        await API.execute_write_query(
            query,
            localized_datetime, location.longitude, location.latitude, location.elevation, location.city, location.state, 
            location.zip, location.street, action, device_type, device_model, device_name, device_os, 
            location.class_, location.type, location.name, location.display_name, 
            location.amenity, location.house_number, location.road, location.quarter, location.neighbourhood, 
            location.suburb, location.county, location.country_code, location.country,
            table_name="locations"
        )
            
        info(f"Successfully posted location: {location.latitude}, {location.longitude}, {location.elevation} on {localized_datetime}")
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
        err(f"Error posting location {e}")
        err(traceback.format_exc())
        return None


@gis.post("/locate")
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
        debug(f"Location received for processing: {lcn}")

    geocoded_locations = await GEO.code(locations)

    responses = []
    if isinstance(geocoded_locations, List):
        for location in geocoded_locations:
            debug(f"Final location to be submitted to database: {location}")
            location_entry = await post_location(location)
            if location_entry:
                responses.append({"location_data": location_entry})
            else:
                warn(f"Posting location to database appears to have failed.")
    else:
        debug(f"Final location to be submitted to database: {geocoded_locations}")
        location_entry = await post_location(geocoded_locations)
        if location_entry:
            responses.append({"location_data": location_entry})
        else:
            warn(f"Posting location to database appears to have failed.")

    return {"message": "Locations and weather updated", "results": responses}


@gis.get("/locate", response_model=Location)
async def get_last_location_endpoint() -> JSONResponse:
    this_location = await get_last_location()

    if this_location:
        location_dict = this_location.model_dump()
        return JSONResponse(content=location_dict)
    else:
        raise HTTPException(status_code=404, detail="No location found before the specified datetime")



@gis.get("/locate/{datetime_str}", response_model=List[Location])
async def get_locate(datetime_str: str, all: bool = False):
    try:
        date_time = await dt(datetime_str)
    except ValueError as e:
        err(f"Invalid datetime string provided: {datetime_str}")
        return ["ERROR: INVALID DATETIME PROVIDED. USE YYYYMMDDHHmmss or YYYYMMDD format."]
    
    locations = await fetch_locations(date_time)
    if not locations:
        raise HTTPException(status_code=404, detail="No nearby data found for this date and time")
        
    return locations if all else [locations[0]]