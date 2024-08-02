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
from datetime import datetime as dt_datetime
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
    # date = await date
    try:
        if latlon == "None":
            date_time = await gis.dt(date)
            place = await gis.fetch_last_location_before(date_time)
            lat = place.latitude
            lon = place.longitude
        else:
            lat, lon = latlon.split(',')
            tz = await GEO.tz_at(lat, lon)
            date_time = await gis.dt(date, tz)

        debug(f"passing date_time {date_time.strftime('%Y-%m-%d %H:%M:%S')}, {lat}/{lon} into get_weather")
        day = await get_weather(date_time, lat, lon, force_refresh=True)
        day_str = str(day)
        return JSONResponse(content={"weather": day_str}, status_code=200)

    except HTTPException as e:
        return JSONResponse(content={"detail": str(e.detail)}, status_code=e.status_code)

    except Exception as e:
        err(f"Error in note_weather_get: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_weather(date_time: dt_datetime, latitude: float, longitude: float, force_refresh: bool = False):
    fetch_new_data = True
    if force_refresh == False:
        daily_weather_data = await get_weather_from_db(date_time, latitude, longitude)
        if daily_weather_data:
            try:
                debug(f"Daily weather data from db: {daily_weather_data}")
                last_updated = str(daily_weather_data['DailyWeather'].get('last_updated'))
                last_updated = await gis.dt(last_updated)
                stored_loc_data = unhexlify(daily_weather_data['DailyWeather'].get('location'))
                stored_loc = loads(stored_loc_data)
                stored_lat = stored_loc.y
                stored_lon = stored_loc.x
                stored_ele = stored_loc.z
                
                hourly_weather = daily_weather_data.get('HourlyWeather')
                # debug(f"Hourly: {hourly_weather}")
                request_haversine = haversine(latitude, longitude, stored_lat, stored_lon)
                debug(f"\nINFO:\nlast updated {last_updated}\nstored lat: {stored_lat} - requested lat: {latitude}\nstored lon: {stored_lon} - requested lon: {longitude}\nHaversine: {request_haversine}")
                
                if last_updated and (date_time <= dt_datetime.now(TZ) and last_updated > date_time and request_haversine < 8) and hourly_weather and len(hourly_weather) > 0:
                    debug(f"We can use existing data... :')")
                    fetch_new_data = False
                    
            except Exception as e:
                err(f"Error in get_weather: {e}")

    if fetch_new_data:
        debug(f"We require new data!")
        request_date_str = date_time.strftime("%Y-%m-%d")
        debug(f"Using {date_time.strftime('%Y-%m-%d')} as our datetime for fetching new data.")
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{latitude},{longitude}/{request_date_str}/{request_date_str}?unitGroup=us&key={VISUALCROSSING_API_KEY}"
        try:
            async with AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    debug(f"Successfully obtained data from VC...")
                    try:
                        weather_data = response.json()
                        store_result = await store_weather_to_db(date_time, weather_data)
                        if store_result == "SUCCESS":
                            debug(f"New weather data for {request_date_str} stored in database...")
                        else:
                            err(f"Failed to store weather data for {request_date_str} in database! {store_result}")

                        debug(f"Attempting to retrieve data for {date_time}, {latitude}, {longitude}")
                        daily_weather_data = await get_weather_from_db(date_time, latitude, longitude)
                        if daily_weather_data is not None:
                            return daily_weather_data
                        else:
                            raise HTTPException(status_code=500, detail="Weather data was not properly stored.")
                    except Exception as e:
                        err(f"Problem parsing VC response or storing data: {e}")
                        raise HTTPException(status_code=500, detail="Weather data was not properly stored.")
                else:
                    err(f"Failed to fetch weather data: {response.status_code}, {response.text}")
        except Exception as e:
            err(f"Exception during API call: {e}")

    return daily_weather_data


async def store_weather_to_db(date_time: dt_datetime, weather_data: dict):
    warn(f"Using {date_time.strftime('%Y-%m-%d %H:%M:%S')} as our datetime in store_weather_to_db")
    try:
        day_data = weather_data.get('days')[0]
        debug(f"RAW DAY_DATA: {day_data}")
        # Handle preciptype and stations as PostgreSQL arrays
        preciptype_array = day_data.get('preciptype', []) or []
        stations_array = day_data.get('stations', []) or []

        date_str = date_time.strftime("%Y-%m-%d")
        warn(f"Using {date_str} in our query in store_weather_to_db.")

        # Get location details from weather data if available
        longitude = weather_data.get('longitude')
        latitude = weather_data.get('latitude')
        tz = await GEO.tz_at(latitude, longitude)
        elevation = await GEO.elevation(latitude, longitude)
        location_point = f"POINTZ({longitude} {latitude} {elevation})" if longitude and latitude and elevation else None

        warn(f"Uncorrected datetimes in store_weather_to_db: {day_data['datetime']}, sunrise: {day_data['sunrise']}, sunset: {day_data['sunset']}")
        day_data['datetime'] = await gis.dt(day_data.get('datetimeEpoch'))
        day_data['sunrise'] = await gis.dt(day_data.get('sunriseEpoch'))
        day_data['sunset'] = await gis.dt(day_data.get('sunsetEpoch'))
        warn(f"Corrected datetimes in store_weather_to_db: {day_data['datetime']}, sunrise: {day_data['sunrise']}, sunset: {day_data['sunset']}")

        daily_weather_params = (
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
        )
    except Exception as e:
        err(f"Failed to prepare database query in store_weather_to_db! {e}")
        return "FAILURE"
    
    try:
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

        daily_weather_id = await API.execute_write_query(daily_weather_query, *daily_weather_params, table_name="dailyweather")
    
        if 'hours' in day_data:
            debug(f"Processing hours now...")
            for hour_data in day_data['hours']:
                try:
                    await asyncio.sleep(0.01)
                    hour_data['datetime'] = await gis.dt(hour_data.get('datetimeEpoch'))
                    hour_preciptype_array = hour_data.get('preciptype', []) or []
                    hour_stations_array = hour_data.get('stations', []) or []
                    hourly_weather_params = (
                        daily_weather_id,
                        hour_data['datetime'],
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
                    )

                    try:
                        hourly_weather_query = '''
                        INSERT INTO hourlyweather (daily_weather_id, datetime, datetimeepoch, temp, feelslike, humidity, dew, precip, precipprob,
                        preciptype, snow, snowdepth, windgust, windspeed, winddir, pressure, cloudcover, visibility, solarradiation, solarenergy,
                        uvindex, severerisk, conditions, icon, stations, source)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
                        RETURNING id
                        '''
                        hourly_weather_id = await API.execute_write_query(hourly_weather_query, *hourly_weather_params, table_name="hourlyweather")
                        debug(f"Done processing hourly_weather_id {hourly_weather_id}")
                    except Exception as e:
                        err(f"EXCEPTION: {e}")

                except Exception as e:
                    err(f"EXCEPTION: {e}")

        return "SUCCESS"
        
    except Exception as e:
        err(f"Error in dailyweather storage: {e}")
        return "FAILURE"

   

async def get_weather_from_db(date_time: dt_datetime, latitude: float, longitude: float):
    warn(f"Using {date_time.strftime('%Y-%m-%d %H:%M:%S')} as our datetime in get_weather_from_db.")
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
        '''
        
        hourly_weather_records = await API.execute_read_query(hourly_query, daily_weather_data['id'], table_name='hourlyweather')
        
        day = {
            'DailyWeather': daily_weather_data,
            'HourlyWeather': hourly_weather_records,
        }
        
        return day
    except Exception as e:
        err(f"Unexpected error occurred in get_weather_from_db: {e}")
        err(f"Traceback: {traceback.format_exc()}")
        return None
