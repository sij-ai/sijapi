'''
Uses the VisualCrossing API and Postgres/PostGIS to source local weather forecasts and history.
'''
#routers/weather.py

import asyncio
import traceback
from fastapi import APIRouter, HTTPException, Query
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from asyncpg.cursor import Cursor
from httpx import AsyncClient
from typing import Dict
from datetime import datetime as dt_datetime, date as dt_date
from shapely.wkb import loads
from binascii import unhexlify
from sijapi import L, VISUALCROSSING_API_KEY, TZ, API, GEO
from sijapi.utilities import haversine
from sijapi.routers import gis

weather = APIRouter()
logger = L.get_module_logger("weather")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)


@weather.get("/weather/refresh", response_class=JSONResponse)
async def get_refreshed_weather(
    date: str = Query(default=dt_datetime.now().strftime("%Y-%m-%d"), description="Enter a date in YYYY-MM-DD format, otherwise it will default to today."),
    latlon: str = Query(default="None", description="Optionally enter latitude and longitude in the format 45.8411,-123.1765; if not provided it will use your recorded location."),
):
    try:
        if latlon == "None":
            date_time = await gis.dt(date)
            place = await gis.fetch_last_location_before(date_time)
            if not place:
                raise HTTPException(status_code=404, detail="No location data found for the given date")
            lat, lon = place.latitude, place.longitude
        else:
            try:
                lat, lon = map(float, latlon.split(','))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid latitude/longitude format. Use format: 45.8411,-123.1765")
            tz = await GEO.tz_at(lat, lon)
            date_time = await gis.dt(date, tz)

        debug(f"Passing date_time {date_time.strftime('%Y-%m-%d %H:%M:%S')}, {lat}/{lon} into get_weather")
        day = await get_weather(date_time, lat, lon, force_refresh=True)
        
        if day is None:
            raise HTTPException(status_code=404, detail="No weather data found for the given date and location")
        
        # Convert the day object to a JSON-serializable format
        day_dict = {}
        for k, v in day.items():
            if k == 'DailyWeather':
                day_dict[k] = {kk: vv.isoformat() if isinstance(vv, (dt_datetime, dt_date)) else vv for kk, vv in v.items()}
            elif k == 'HourlyWeather':
                day_dict[k] = [{kk: vv.isoformat() if isinstance(vv, (dt_datetime, dt_date)) else vv for kk, vv in hour.items()} for hour in v]
            else:
                day_dict[k] = v.isoformat() if isinstance(v, (dt_datetime, dt_date)) else v

        return JSONResponse(content={"weather": day_dict}, status_code=200)

    except HTTPException as e:
        err(f"HTTP Exception in get_refreshed_weather: {e.detail}")
        return JSONResponse(content={"detail": str(e.detail)}, status_code=e.status_code)

    except Exception as e:
        err(f"Unexpected error in get_refreshed_weather: {str(e)}")
        err(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(content={"detail": "An unexpected error occurred"}, status_code=500)


async def get_weather(date_time: dt_datetime, latitude: float, longitude: float, force_refresh: bool = False):
    fetch_new_data = force_refresh
    daily_weather_data = None

    if not force_refresh:
        try:
            daily_weather_data = await get_weather_from_db(date_time, latitude, longitude)
            if daily_weather_data:
                debug(f"Daily weather data from db: {daily_weather_data}")
                last_updated = str(daily_weather_data['DailyWeather'].get('last_updated'))
                last_updated = await gis.dt(last_updated)
                stored_loc_data = unhexlify(daily_weather_data['DailyWeather'].get('location'))
                stored_loc = loads(stored_loc_data)
                stored_lat, stored_lon, stored_ele = stored_loc.y, stored_loc.x, stored_loc.z
                
                hourly_weather = daily_weather_data.get('HourlyWeather')
                request_haversine = haversine(latitude, longitude, stored_lat, stored_lon)
                debug(f"\nINFO:\nlast updated {last_updated}\nstored lat: {stored_lat} - requested lat: {latitude}\nstored lon: {stored_lon} - requested lon: {longitude}\nHaversine: {request_haversine}")
                
                if last_updated and (date_time <= dt_datetime.now(TZ) and last_updated > date_time and request_haversine < 8) and hourly_weather and len(hourly_weather) > 0:
                    debug(f"Using existing data")
                    fetch_new_data = False
                else:
                    fetch_new_data = True
        except Exception as e:
            err(f"Error checking existing weather data: {e}")
            fetch_new_data = True

    if fetch_new_data:
        debug(f"Fetching new weather data")
        request_date_str = date_time.strftime("%Y-%m-%d")
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{latitude},{longitude}/{request_date_str}/{request_date_str}?unitGroup=us&key={VISUALCROSSING_API_KEY}"
        
        try:
            async with AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    weather_data = response.json()
                    store_result = await store_weather_to_db(date_time, weather_data)
                    if store_result != "SUCCESS":
                        raise HTTPException(status_code=500, detail=f"Failed to store weather data: {store_result}")

                    daily_weather_data = await get_weather_from_db(date_time, latitude, longitude)
                    if daily_weather_data is None:
                        raise HTTPException(status_code=500, detail="Weather data was not properly stored.")
                else:
                    raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch weather data: {response.text}")
        except HTTPException:
            raise
        except Exception as e:
            err(f"Exception during API call or data storage: {e}")
            err(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error fetching or storing weather data: {str(e)}")

    if daily_weather_data is None:
        raise HTTPException(status_code=404, detail="No weather data found")

    return daily_weather_data




async def store_weather_to_db(date_time: dt_datetime, weather_data: dict):
    debug(f"Starting store_weather_to_db for datetime: {date_time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        day_data = weather_data.get('days', [{}])[0]
        debug(f"RAW DAY_DATA: {day_data}")
    
        # Handle preciptype and stations as PostgreSQL arrays
        preciptype_array = day_data.get('preciptype', [])
        stations_array = day_data.get('stations', [])
        preciptype_array = [] if preciptype_array is None else preciptype_array
        stations_array = [] if stations_array is None else stations_array
    
        date_str = date_time.strftime("%Y-%m-%d")
        debug(f"Using date {date_str} for database query")
    
        # Get location details
        longitude = weather_data.get('longitude')
        latitude = weather_data.get('latitude')
        if longitude is None or latitude is None:
            raise ValueError("Missing longitude or latitude in weather data")
    
        tz = await GEO.tz_at(latitude, longitude)
        elevation = await GEO.elevation(latitude, longitude)
        location_point = f"POINTZ({longitude} {latitude} {elevation})" if elevation else None
    
        debug(f"Uncorrected datetimes: datetime={day_data.get('datetime')}, sunrise={day_data.get('sunrise')}, sunset={day_data.get('sunset')}")
        day_data['datetime'] = await gis.dt(day_data.get('datetimeEpoch'))
        day_data['sunrise'] = await gis.dt(day_data.get('sunriseEpoch'))
        day_data['sunset'] = await gis.dt(day_data.get('sunsetEpoch'))
        debug(f"Corrected datetimes: datetime={day_data['datetime']}, sunrise={day_data['sunrise']}, sunset={day_data['sunset']}")
    
        daily_weather_params = [
            day_data.get('sunrise'), day_data.get('sunriseEpoch'),
            day_data.get('sunset'), day_data.get('sunsetEpoch'),
            day_data.get('description'), day_data.get('tempmax'),
            day_data.get('tempmin'), day_data.get('uvindex'),
            day_data.get('winddir'), day_data.get('windspeed'),
            day_data.get('icon'), dt_datetime.now(tz),
            day_data.get('datetime'), day_data.get('datetimeEpoch'),
            day_data.get('temp'), day_data.get('feelslikemax'),
            day_data.get('feelslikemin'), day_data.get('feelslike'),
            day_data.get('dew'), day_data.get('humidity'),
            day_data.get('precip'), day_data.get('precipprob'),
            day_data.get('precipcover'), preciptype_array,
            day_data.get('snow'), day_data.get('snowdepth'),
            day_data.get('windgust'), day_data.get('pressure'),
            day_data.get('cloudcover'), day_data.get('visibility'),
            day_data.get('solarradiation'), day_data.get('solarenergy'),
            day_data.get('severerisk', 0), day_data.get('moonphase'),
            day_data.get('conditions'), stations_array, day_data.get('source'),
            location_point
        ]
    
        debug(f"Prepared daily_weather_params: {daily_weather_params}")
    
        daily_weather_query = '''
        INSERT INTO dailyweather (
            sunrise, sunriseepoch, sunset, sunsetepoch, description,
            tempmax, tempmin, uvindex, winddir, windspeed, icon, last_updated,
            datetime, datetimeepoch, temp, feelslikemax, feelslikemin, feelslike,
            dew, humidity, precip, precipprob, precipcover, preciptype,
            snow, snowdepth, windgust, pressure, cloudcover, visibility,
            solarradiation, solarenergy, severerisk, moonphase, conditions,
            stations, source, location
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38)
        RETURNING id
        '''
    
        daily_weather_result = await API.execute_write_query(daily_weather_query, *daily_weather_params, table_name="dailyweather")
        
        if not daily_weather_result:
            raise ValueError("Failed to insert daily weather data: no result returned")
        
        daily_weather_id = daily_weather_result[0]['id']
        debug(f"Inserted daily weather data with id: {daily_weather_id}")
    
        if 'hours' in day_data:
            debug(f"Processing {len(day_data['hours'])} hourly records")
            for hour_data in day_data['hours']:
                try:
                    hour_datetime = await gis.dt(hour_data.get('datetimeEpoch'))
                    hour_preciptype_array = hour_data.get('preciptype', []) or []
                    hour_stations_array = hour_data.get('stations', []) or []
                    hourly_weather_params = [
                        daily_weather_id,
                        hour_datetime,
                        hour_data.get('datetimeEpoch'),
                        hour_data['temp'],
                        hour_data['feelslike'],
                        hour_data['humidity'],
                        hour_data['dew'],
                        hour_data['precip'],
                        hour_data['precipprob'],
                        hour_preciptype_array,
                        hour_data['snow'],
                        hour_data['snowdepth'],
                        hour_data['windgust'],
                        hour_data['windspeed'],
                        hour_data['winddir'],
                        hour_data['pressure'],
                        hour_data['cloudcover'],
                        hour_data['visibility'],
                        hour_data['solarradiation'],
                        hour_data['solarenergy'],
                        hour_data['uvindex'],
                        hour_data.get('severerisk', 0),
                        hour_data['conditions'],
                        hour_data['icon'],
                        hour_stations_array,
                        hour_data.get('source', ''),
                    ]
    
                    hourly_weather_query = '''
                    INSERT INTO hourlyweather (
                        daily_weather_id, datetime, datetimeepoch, temp, feelslike, 
                        humidity, dew, precip, precipprob, preciptype, snow, snowdepth, 
                        windgust, windspeed, winddir, pressure, cloudcover, visibility, 
                        solarradiation, solarenergy, uvindex, severerisk, conditions, 
                        icon, stations, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, 
                              $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
                    '''
                    hourly_result = await API.execute_write_query(hourly_weather_query, *hourly_weather_params, table_name="hourlyweather")
                    debug(f"Inserted hourly weather data for {hour_datetime}")
                except Exception as e:
                    err(f"Error processing hourly data: {e}")
                    err(f"Problematic hour_data: {hour_data}")
                    raise
    
        debug("Successfully stored weather data")
        return "SUCCESS"
        
    except Exception as e:
        err(f"Error in weather storage: {e}")
        err(f"Traceback: {traceback.format_exc()}")
        return "FAILURE"

   

async def get_weather_from_db(date_time: dt_datetime, latitude: float, longitude: float):
    debug(f"Using {date_time.strftime('%Y-%m-%d %H:%M:%S')} as our datetime in get_weather_from_db.")
    query_date = date_time.date()
    try:
        # Query to get daily weather data
        daily_query = '''
            SELECT DW.* FROM dailyweather DW
            WHERE DW.datetime::date = $1
            AND ST_DWithin(DW.location::geography, ST_MakePoint($2,$3)::geography, 8046.72) 
            ORDER BY ST_Distance(DW.location, ST_MakePoint($4, $5)::geography) ASC
            LIMIT 1
        '''
    
        daily_weather_records = await API.execute_read_query(daily_query, query_date, longitude, latitude, longitude, latitude, table_name='dailyweather')
    
        if not daily_weather_records:
            debug(f"No daily weather data retrieved from database.")
            return None
    
        daily_weather_data = daily_weather_records[0]  # Get the first (and only) record
        
        # Query to get hourly weather data
        hourly_query = '''
            SELECT HW.* FROM hourlyweather HW
            WHERE HW.daily_weather_id = $1
            ORDER BY HW.datetime ASC
        '''
        
        hourly_weather_records = await API.execute_read_query(hourly_query, daily_weather_data['id'], table_name='hourlyweather')
        
        day = {
            'DailyWeather': daily_weather_data,
            'HourlyWeather': hourly_weather_records,
        }
        
        debug(f"Retrieved weather data for {date_time.date()}")
        return day
    except Exception as e:
        err(f"Unexpected error occurred in get_weather_from_db: {e}")
        err(f"Traceback: {traceback.format_exc()}")
        return None

