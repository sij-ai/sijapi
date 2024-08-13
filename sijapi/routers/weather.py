import asyncio
import traceback
import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from asyncpg.cursor import Cursor
from httpx import AsyncClient
from typing import Dict
from datetime import datetime as dt_datetime, date as dt_date
from shapely.wkb import loads
from binascii import unhexlify
from sijapi import VISUALCROSSING_API_KEY, TZ, Sys, GEO, Db
from sijapi.utilities import haversine
from sijapi.routers import gis
from sijapi.logs import get_logger
from sijapi.serialization import json_dumps, serialize

l = get_logger(__name__)

weather = APIRouter()

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

        l.debug(f"Passing date_time {date_time.strftime('%Y-%m-%d %H:%M:%S')}, {lat}/{lon} into get_weather")
        day = await get_weather(date_time, lat, lon, force_refresh=True)
        
        if day is None:
            raise HTTPException(status_code=404, detail="No weather data found for the given date and location")

        json_compatible_data = jsonable_encoder({"weather": day})
        return JSONResponse(content=json_compatible_data)

    except HTTPException as e:
        l.error(f"HTTP Exception in get_refreshed_weather: {e.detail}")
        return JSONResponse(content={"detail": str(e.detail)}, status_code=e.status_code)

    except Exception as e:
        l.error(f"Unexpected error in get_refreshed_weather: {str(e)}")
        l.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(content={"detail": "An unexpected error occurred"}, status_code=500)


async def get_weather(date_time: dt_datetime, latitude: float, longitude: float, force_refresh: bool = False):
    fetch_new_data = force_refresh
    daily_weather_data = None

    if not force_refresh:
        try:
            daily_weather_data = await get_weather_from_db(date_time, latitude, longitude)
            if daily_weather_data:
                l.debug(f"Daily weather data from db: {daily_weather_data}")
                last_updated = str(daily_weather_data['DailyWeather'].get('last_updated'))
                last_updated = await gis.dt(last_updated)
                stored_loc_data = unhexlify(daily_weather_data['DailyWeather'].get('location'))
                stored_loc = loads(stored_loc_data)
                stored_lat, stored_lon, stored_ele = stored_loc.y, stored_loc.x, stored_loc.z
                
                hourly_weather = daily_weather_data.get('HourlyWeather')
                request_haversine = haversine(latitude, longitude, stored_lat, stored_lon)
                l.debug(f"\nINFO:\nlast updated {last_updated}\nstored lat: {stored_lat} - requested lat: {latitude}\nstored lon: {stored_lon} - requested lon: {longitude}\nHaversine: {request_haversine}")
                
                if last_updated and (date_time <= dt_datetime.now(TZ) and last_updated > date_time and request_haversine < 8) and hourly_weather and len(hourly_weather) > 0:
                    l.debug(f"Using existing data")
                    fetch_new_data = False
                else:
                    fetch_new_data = True
        except Exception as e:
            l.error(f"Error checking existing weather data: {e}")
            fetch_new_data = True

    if fetch_new_data:
        l.debug(f"Fetching new weather data")
        request_date_str = date_time.strftime("%Y-%m-%d")
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{latitude},{longitude}/{request_date_str}/{request_date_str}?unitGroup=us&key={VISUALCROSSING_API_KEY}"
        
        try:
            async with AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    weather_data = response.json()

                    try:
                        store_result = await store_weather_to_db(date_time, weather_data)
                        if store_result != "SUCCESS":
                            raise HTTPException(status_code=500, detail=f"Failed to store weather data: {store_result}")
                    except Exception as e:
                        l.error(f"Error storing weather data: {str(e)}")
                        raise HTTPException(status_code=500, detail=f"Error storing weather data: {str(e)}")

                    daily_weather_data = await get_weather_from_db(date_time, latitude, longitude)
                    if daily_weather_data is None:
                        raise HTTPException(status_code=500, detail="Weather data was not properly stored.")
                else:
                    raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch weather data: {response.text}")
        except HTTPException:
            raise
        except Exception as e:
            l.error(f"Exception during API call or data storage: {e}")
            l.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error fetching or storing weather data: {str(e)}")

    if daily_weather_data is None:
        raise HTTPException(status_code=404, detail="No weather data found")

    return daily_weather_data

async def store_weather_to_db(date_time: dt_datetime, weather_data: dict):
    try:
        day_data = weather_data.get('days', [{}])[0]
        
        preciptype_array = day_data.get('preciptype', []) or []
        stations_array = day_data.get('stations', []) or []
    
        longitude = weather_data.get('longitude')
        latitude = weather_data.get('latitude')
        if longitude is None or latitude is None:
            raise ValueError("Missing longitude or latitude in weather data")
    
        tz = await GEO.tz_at(latitude, longitude)
        elevation = await GEO.elevation(latitude, longitude)
        location_point = f"POINTZ({longitude} {latitude} {elevation})" if elevation else None
    
        daily_weather_params = {
            'location': location_point,
            'sunrise': await gis.dt(day_data.get('sunriseEpoch')),
            'sunriseepoch': day_data.get('sunriseEpoch'),
            'sunset': await gis.dt(day_data.get('sunsetEpoch')),
            'sunsetepoch': day_data.get('sunsetEpoch'),
            'description': day_data.get('description'),
            'tempmax': day_data.get('tempmax'),
            'tempmin': day_data.get('tempmin'),
            'uvindex': day_data.get('uvindex'),
            'winddir': day_data.get('winddir'),
            'windspeed': day_data.get('windspeed'),
            'icon': day_data.get('icon'),
            'last_updated': dt_datetime.now(tz),
            'datetime': await gis.dt(day_data.get('datetimeEpoch')),
            'datetimeepoch': day_data.get('datetimeEpoch'),
            'temp': day_data.get('temp'),
            'feelslikemax': day_data.get('feelslikemax'),
            'feelslikemin': day_data.get('feelslikemin'),
            'feelslike': day_data.get('feelslike'),
            'dew': day_data.get('dew'),
            'humidity': day_data.get('humidity'),
            'precip': day_data.get('precip'),
            'precipprob': day_data.get('precipprob'),
            'precipcover': day_data.get('precipcover'),
            'preciptype': preciptype_array,
            'snow': day_data.get('snow'),
            'snowdepth': day_data.get('snowdepth'),
            'windgust': day_data.get('windgust'),
            'pressure': day_data.get('pressure'),
            'cloudcover': day_data.get('cloudcover'),
            'visibility': day_data.get('visibility'),
            'solarradiation': day_data.get('solarradiation'),
            'solarenergy': day_data.get('solarenergy'),
            'severerisk': day_data.get('severerisk', 0),
            'moonphase': day_data.get('moonphase'),
            'conditions': day_data.get('conditions'),
            'stations': stations_array,
            'source': day_data.get('source')
        }
    
        daily_weather_query = '''
        INSERT INTO dailyweather (
            location, sunrise, sunriseepoch, sunset, sunsetepoch, description,
            tempmax, tempmin, uvindex, winddir, windspeed, icon, last_updated,
            datetime, datetimeepoch, temp, feelslikemax, feelslikemin, feelslike,
            dew, humidity, precip, precipprob, precipcover, preciptype,
            snow, snowdepth, windgust, pressure, cloudcover, visibility,
            solarradiation, solarenergy, severerisk, moonphase, conditions,
            stations, source
        ) VALUES (
            :location, :sunrise, :sunriseepoch, :sunset, :sunsetepoch, :description,
            :tempmax, :tempmin, :uvindex, :winddir, :windspeed, :icon, :last_updated,
            :datetime, :datetimeepoch, :temp, :feelslikemax, :feelslikemin, :feelslike,
            :dew, :humidity, :precip, :precipprob, :precipcover, :preciptype,
            :snow, :snowdepth, :windgust, :pressure, :cloudcover, :visibility,
            :solarradiation, :solarenergy, :severerisk, :moonphase, :conditions,
            :stations, :source
        ) RETURNING id
        '''
    
        daily_weather_result = await Db.write(daily_weather_query, **daily_weather_params, table_name="dailyweather")
            
        if daily_weather_result is None:
            raise ValueError("Failed to insert daily weather data: no result returned")
        
        daily_weather_row = daily_weather_result.fetchone()
        if daily_weather_row is None:
            raise ValueError("Failed to retrieve inserted daily weather ID: fetchone() returned None")

        daily_weather_id = daily_weather_row[0]

        l.debug(f"Inserted daily weather data with id: {daily_weather_id}")
    
        # Hourly weather insertion
        if 'hours' in day_data:
            l.debug(f"Processing {len(day_data['hours'])} hourly records")
            for hour_data in day_data['hours']:
                hour_preciptype_array = hour_data.get('preciptype', []) or []
                hour_stations_array = hour_data.get('stations', []) or []
                hourly_weather_params = {
                    'daily_weather_id': daily_weather_id,
                    'datetime': await gis.dt(hour_data.get('datetimeEpoch')),
                    'datetimeepoch': hour_data.get('datetimeEpoch'),
                    'temp': hour_data.get('temp'),
                    'feelslike': hour_data.get('feelslike'),
                    'humidity': hour_data.get('humidity'),
                    'dew': hour_data.get('dew'),
                    'precip': hour_data.get('precip'),
                    'precipprob': hour_data.get('precipprob'),
                    'preciptype': hour_preciptype_array,
                    'snow': hour_data.get('snow'),
                    'snowdepth': hour_data.get('snowdepth'),
                    'windgust': hour_data.get('windgust'),
                    'windspeed': hour_data.get('windspeed'),
                    'winddir': hour_data.get('winddir'),
                    'pressure': hour_data.get('pressure'),
                    'cloudcover': hour_data.get('cloudcover'),
                    'visibility': hour_data.get('visibility'),
                    'solarradiation': hour_data.get('solarradiation'),
                    'solarenergy': hour_data.get('solarenergy'),
                    'uvindex': hour_data.get('uvindex'),
                    'severerisk': hour_data.get('severerisk', 0),
                    'conditions': hour_data.get('conditions'),
                    'icon': hour_data.get('icon'),
                    'stations': hour_stations_array,
                    'source': hour_data.get('source', '')
                }
    
                hourly_weather_query = '''
                INSERT INTO hourlyweather (
                    daily_weather_id, datetime, datetimeepoch, temp, feelslike, 
                    humidity, dew, precip, precipprob, preciptype, snow, snowdepth, 
                    windgust, windspeed, winddir, pressure, cloudcover, visibility, 
                    solarradiation, solarenergy, uvindex, severerisk, conditions, 
                    icon, stations, source
                ) VALUES (
                    :daily_weather_id, :datetime, :datetimeepoch, :temp, :feelslike, 
                    :humidity, :dew, :precip, :precipprob, :preciptype, :snow, :snowdepth, 
                    :windgust, :windspeed, :winddir, :pressure, :cloudcover, :visibility, 
                    :solarradiation, :solarenergy, :uvindex, :severerisk, :conditions, 
                    :icon, :stations, :source
                ) RETURNING id
                '''
                hourly_result = await Db.write(hourly_weather_query, **hourly_weather_params, table_name="hourlyweather")
                if hourly_result is None:
                    l.warning(f"Failed to insert hourly weather data for {hour_data.get('datetimeEpoch')}")
                else:
                    hourly_row = hourly_result.fetchone()
                    if hourly_row is None:
                        l.warning(f"Failed to retrieve inserted hourly weather ID for {hour_data.get('datetimeEpoch')}")
                    else:
                        hourly_id = hourly_row[0]
                        l.debug(f"Inserted hourly weather data with id: {hourly_id}")
    
        return "SUCCESS"
    except Exception as e:
        l.error(f"Error in weather storage: {e}")
        l.error(f"Traceback: {traceback.format_exc()}")
        return "FAILURE"


async def get_weather_from_db(date_time: dt_datetime, latitude: float, longitude: float):
    l.debug(f"Using {date_time.strftime('%Y-%m-%d %H:%M:%S')} as our datetime in get_weather_from_db.")
    query_date = date_time.date()
    try:
        # Query to get daily weather data
        daily_query = '''
    SELECT * FROM dailyweather
    WHERE DATE(datetime) = :query_date
    AND ST_DWithin(location::geography, ST_MakePoint(:longitude,:latitude)::geography, 8046.72) 
    ORDER BY ST_Distance(location, ST_MakePoint(:longitude2, :latitude2)::geography) ASC
    LIMIT 1
'''    
        daily_weather_records = await Db.read(daily_query, query_date=query_date, longitude=longitude, latitude=latitude, longitude2=longitude, latitude2=latitude, table_name='dailyweather')

        if not daily_weather_records:
            l.debug(f"No daily weather data retrieved from database.")
            return None
    
        daily_weather_data = daily_weather_records[0]
        
        hourly_query = '''
    SELECT * FROM hourlyweather
    WHERE daily_weather_id = :daily_weather_id
    ORDER BY datetime ASC
'''
        hourly_weather_records = await Db.read(
            hourly_query, 
            daily_weather_id=daily_weather_data['id'], 
            table_name='hourlyweather'
        )

        day = {
            'DailyWeather': daily_weather_data,
            'HourlyWeather': hourly_weather_records,
        }
        
        l.debug(f"Retrieved weather data for {date_time.date()}")
        return day
    
    except Exception as e:
        l.error(f"Unexpected error occurred in get_weather_from_db: {e}")
        l.error(f"Traceback: {traceback.format_exc()}")
        return None
