import asyncio
from fastapi import APIRouter, HTTPException
from fastapi import HTTPException
from asyncpg.cursor import Cursor
from httpx import AsyncClient
from typing import Dict
from datetime import datetime
from shapely.wkb import loads
from binascii import unhexlify
from sijapi.utilities import localize_datetime
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi import VISUALCROSSING_API_KEY, TZ
from sijapi.utilities import get_db_connection, haversine
from sijapi.routers import locate

weather = APIRouter()


async def get_weather(date_time: datetime, latitude: float, longitude: float):
    # request_date_str = date_time.strftime("%Y-%m-%d")
    DEBUG(f"Called get_weather with lat: {latitude}, lon: {longitude}, date_time: {date_time}")
    daily_weather_data = await get_weather_from_db(date_time, latitude, longitude)
    fetch_new_data = True
    if daily_weather_data:
        try:
            DEBUG(f"Daily weather data from db: {daily_weather_data}")
            last_updated = str(daily_weather_data['DailyWeather'].get('last_updated'))
            last_updated = localize_datetime(last_updated)
            stored_loc_data = unhexlify(daily_weather_data['DailyWeather'].get('location'))
            stored_loc = loads(stored_loc_data)
            stored_lat = stored_loc.y
            stored_lon = stored_loc.x
            stored_ele = stored_loc.z

            
            hourly_weather = daily_weather_data.get('HourlyWeather')

            DEBUG(f"Hourly: {hourly_weather}")

            DEBUG(f"\nINFO:\nlast updated {last_updated}\nstored lat: {stored_lat} - requested lat: {latitude}\nstored lon: {stored_lon} - requested lon: {longitude}\n")

            request_haversine = haversine(latitude, longitude, stored_lat, stored_lon)
            DEBUG(f"\nINFO:\nlast updated {last_updated}\nstored lat: {stored_lat} - requested lat: {latitude}\nstored lon: {stored_lon} - requested lon: {longitude}\nHaversine: {request_haversine}")
            
            if last_updated and (date_time <= datetime.now(TZ) and last_updated > date_time and request_haversine < 8) and hourly_weather and len(hourly_weather) > 0:
                DEBUG(f"We can use existing data... :')")
                fetch_new_data = False
                
        except Exception as e:
            ERR(f"Error in get_weather: {e}")

    if fetch_new_data:
        DEBUG(f"We require new data!")
        request_date_str = date_time.strftime("%Y-%m-%d")
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{latitude},{longitude}/{request_date_str}/{request_date_str}?unitGroup=us&key={VISUALCROSSING_API_KEY}"
        try:
            async with AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    DEBUG(f"Successfully obtained data from VC...")
                    try:
                        weather_data = response.json()
                        store_result = await store_weather_to_db(date_time, weather_data)
                        if store_result == "SUCCESS":
                            DEBUG(f"New weather data for {request_date_str} stored in database...")
                        else:
                            ERR(f"Failed to store weather data for {request_date_str} in database! {store_result}")

                        DEBUG(f"Attempting to retrieve data for {date_time}, {latitude}, {longitude}")
                        daily_weather_data = await get_weather_from_db(date_time, latitude, longitude)
                        if daily_weather_data is not None:
                            return daily_weather_data
                        else:
                            raise HTTPException(status_code=500, detail="Weather data was not properly stored.")
                    except Exception as e:
                        ERR(f"Problem parsing VC response or storing data: {e}")
                        raise HTTPException(status_code=500, detail="Weather data was not properly stored.")
                else:
                    ERR(f"Failed to fetch weather data: {response.status_code}, {response.text}")
        except Exception as e:
            ERR(f"Exception during API call: {e}")

    return daily_weather_data


async def store_weather_to_db(date_time: datetime, weather_data: dict):
    conn = await get_db_connection() 
        
    try:
        day_data = weather_data.get('days')[0]
        DEBUG(f"day_data.get('sunrise'): {day_data.get('sunrise')}")

        # Handle preciptype and stations as PostgreSQL arrays
        preciptype_array = day_data.get('preciptype', []) or []
        stations_array = day_data.get('stations', []) or []

        date_str = date_time.strftime("%Y-%m-%d")

        # Get location details from weather data if available
        longitude = weather_data.get('longitude')
        latitude = weather_data.get('latitude')
        elevation = locate.get_elevation(latitude, longitude) # 152.4  # default until we add a geocoder that can look up actual elevation; weather_data.get('elevation')  # assuming 'elevation' key, replace if different
        location_point = f"POINTZ({longitude} {latitude} {elevation})" if longitude and latitude and elevation else None

        # Correct for the datetime objects 
        day_data['datetime'] = localize_datetime(day_data.get('datetime')) #day_data.get('datetime'))
        day_data['sunrise'] = day_data['datetime'].replace(hour=int(day_data.get('sunrise').split(':')[0]), minute=int(day_data.get('sunrise').split(':')[1]))
        day_data['sunset'] = day_data['datetime'].replace(hour=int(day_data.get('sunset').split(':')[0]), minute=int(day_data.get('sunset').split(':')[1])) 

        daily_weather_params = (
            day_data.get('sunrise'), day_data.get('sunriseEpoch'),
            day_data.get('sunset'), day_data.get('sunsetEpoch'),
            day_data.get('description'), day_data.get('tempmax'),
            day_data.get('tempmin'), day_data.get('uvindex'),
            day_data.get('winddir'), day_data.get('windspeed'),
            day_data.get('icon'), datetime.now(),
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
        ERR(f"Failed to prepare database query in store_weather_to_db! {e}")
    
    try:
        daily_weather_query = '''
        INSERT INTO DailyWeather (
            sunrise, sunriseEpoch, sunset, sunsetEpoch, description,
            tempmax, tempmin, uvindex, winddir, windspeed, icon, last_updated,
            datetime, datetimeEpoch, temp, feelslikemax, feelslikemin, feelslike,
            dew, humidity, precip, precipprob, precipcover, preciptype,
            snow, snowdepth, windgust, pressure, cloudcover, visibility,
            solarradiation, solarenergy, severerisk, moonphase, conditions,
            stations, source, location
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38)
        RETURNING id
        '''
    
        # Debug logs for better insights
        # DEBUG("Executing query: %s", daily_weather_query)
        # DEBUG("With parameters: %s", daily_weather_params)

        # Execute the query to insert daily weather data
        async with conn.transaction():
            daily_weather_id = await conn.fetchval(daily_weather_query, *daily_weather_params)
            
    
        if 'hours' in day_data:
            for hour_data in day_data['hours']:
                try:
                    await asyncio.sleep(0.1)
                #    hour_data['datetime'] = parse_date(hour_data.get('datetime'))
                    hour_timestamp = date_str + ' ' + hour_data['datetime']
                    hour_data['datetime'] = localize_datetime(hour_timestamp)
                    DEBUG(f"Processing hours now...")
                    # DEBUG(f"Processing {hour_data['datetime']}")

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
                        INSERT INTO HourlyWeather (daily_weather_id, datetime, datetimeEpoch, temp, feelslike, humidity, dew, precip, precipprob,
                        preciptype, snow, snowdepth, windgust, windspeed, winddir, pressure, cloudcover, visibility, solarradiation, solarenergy,
                        uvindex, severerisk, conditions, icon, stations, source)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
                        RETURNING id
                        '''
                        # Debug logs for better insights
                        # DEBUG("Executing query: %s", hourly_weather_query)
                        # DEBUG("With parameters: %s", hourly_weather_params)

                        # Execute the query to insert hourly weather data
                        async with conn.transaction():
                            hourly_weather_id = await conn.fetchval(hourly_weather_query, *hourly_weather_params)
                        # ERR(f"\n{hourly_weather_id}")
                                
                    except Exception as e:
                        ERR(f"EXCEPTION: {e}")

                except Exception as e:
                    ERR(f"EXCEPTION: {e}")

        return "SUCCESS"
        
    except Exception as e:
        ERR(f"Error in dailyweather storage: {e}")
   


async def get_weather_from_db(date_time: datetime, latitude: float, longitude: float):
    conn = await get_db_connection()

    query_date = date_time.date()
    try:
        # Query to get daily weather data
        query = '''
            SELECT DW.* FROM DailyWeather DW
            WHERE DW.datetime::date = $1
            AND ST_DWithin(DW.location::geography, ST_MakePoint($2,$3)::geography, 8046.72) 
            ORDER BY ST_Distance(DW.location, ST_MakePoint($4, $5)::geography) ASC
            LIMIT 1
        '''

        daily_weather_data = await conn.fetchrow(query, query_date, longitude, latitude, longitude, latitude)

        if daily_weather_data is None:
            DEBUG(f"No daily weather data retrieved from database.")
            return None
        # else:
            # DEBUG(f"Daily_weather_data: {daily_weather_data}")
        # Query to get hourly weather data
        query = '''
            SELECT HW.* FROM HourlyWeather HW
            WHERE HW.daily_weather_id = $1
        '''
        hourly_weather_data = await conn.fetch(query, daily_weather_data['id'])

        day: Dict = {
            'DailyWeather': dict(daily_weather_data),
            'HourlyWeather': [dict(row) for row in hourly_weather_data],
        }
        # DEBUG(f"day: {day}")
        return day
    except Exception as e:
        ERR(f"Unexpected error occurred: {e}")


