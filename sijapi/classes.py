from pydantic import BaseModel, Field
from typing import List, Optional, Any, Tuple, Dict, Union, Tuple
from datetime import datetime, timedelta, timezone
import asyncio
import json
import os
import re
import yaml
import math
from timezonefinder import TimezoneFinder
from pathlib import Path
import asyncpg
import aiohttp
import aiofiles
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import reverse_geocoder as rg
from timezonefinder import TimezoneFinder
from srtm import get_data
from pathlib import Path
import yaml
from typing import Union, List, TypeVar, Type
from pydantic import BaseModel, create_model

from pydantic import BaseModel, Field
from typing import List, Dict
import yaml
from pathlib import Path
import os
from dotenv import load_dotenv

T = TypeVar('T', bound='Configuration')

class ModulesConfig(BaseModel):
    asr: bool = Field(alias="asr")
    calendar: bool = Field(alias="calendar")
    email: bool = Field(alias="email")
    health: bool = Field(alias="health")
    hooks: bool = Field(alias="hooks")
    llm: bool = Field(alias="llm")
    locate: bool = Field(alias="locate")
    note: bool = Field(alias="note")
    sd: bool = Field(alias="sd")
    serve: bool = Field(alias="serve")
    time: bool = Field(alias="time")
    tts: bool = Field(alias="tts")
    weather: bool = Field(alias="weather")


class APIConfig(BaseModel):
    BIND: str
    PORT: int
    URL: str
    PUBLIC: List[str]
    TRUSTED_SUBNETS: List[str]
    MODULES: ModulesConfig
    BaseTZ: Optional[str] = 'UTC'
    KEYS: List[str]

    @classmethod
    def load_from_yaml(cls, config_path: Path, secrets_path: Path):
        # Load main configuration
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        print(f"Loaded main config: {config_data}")  # Debug print
        
        # Load secrets
        try:
            with open(secrets_path, 'r') as file:
                secrets_data = yaml.safe_load(file)
            print(f"Loaded secrets: {secrets_data}")  # Debug print
        except FileNotFoundError:
            print(f"Secrets file not found: {secrets_path}")
            secrets_data = {}
        except yaml.YAMLError as e:
            print(f"Error parsing secrets YAML: {e}")
            secrets_data = {}
        
        # Handle KEYS placeholder
        if isinstance(config_data.get('KEYS'), list) and len(config_data['KEYS']) == 1:
            placeholder = config_data['KEYS'][0]
            if placeholder.startswith('{{') and placeholder.endswith('}}'):
                key = placeholder[2:-2].strip()  # Remove {{ }} and whitespace
                parts = key.split('.')
                if len(parts) == 2 and parts[0] == 'SECRET':
                    secret_key = parts[1]
                    if secret_key in secrets_data:
                        config_data['KEYS'] = secrets_data[secret_key]
                        print(f"Replaced KEYS with secret: {config_data['KEYS']}")  # Debug print
                    else:
                        print(f"Secret key '{secret_key}' not found in secrets file")
                else:
                    print(f"Invalid secret placeholder format: {placeholder}")
        
        # Convert 'on'/'off' to boolean for MODULES if they are strings
        for key, value in config_data['MODULES'].items():
            if isinstance(value, str):
                config_data['MODULES'][key] = value.lower() == 'on'
            elif isinstance(value, bool):
                config_data['MODULES'][key] = value
            else:
                raise ValueError(f"Invalid value for module {key}: {value}. Must be 'on', 'off', True, or False.")
        
        return cls(**config_data)

class Configuration(BaseModel):
    @classmethod
    def load_config(cls: Type[T], yaml_path: Union[str, Path]) -> Union[T, List[T]]:
        yaml_path = Path(yaml_path)
        with yaml_path.open('r') as file:
            config_data = yaml.safe_load(file)
        
        # Load environment variables
        load_dotenv()
        
        # Resolve placeholders
        config_data = cls.resolve_placeholders(config_data)
        
        if isinstance(config_data, list):
            return [cls.create_dynamic_model(**cfg) for cfg in config_data]
        elif isinstance(config_data, dict):
            return cls.create_dynamic_model(**config_data)
        else:
            raise ValueError(f"Unsupported YAML structure in {yaml_path}")

    @classmethod
    def resolve_placeholders(cls, data):
        if isinstance(data, dict):
            return {k: cls.resolve_placeholders(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.resolve_placeholders(v) for v in data]
        elif isinstance(data, str):
            return cls.resolve_string_placeholders(data)
        else:
            return data

    @classmethod
    def resolve_string_placeholders(cls, value):
        pattern = r'\{\{\s*([^}]+)\s*\}\}'
        matches = re.findall(pattern, value)
        
        for match in matches:
            parts = match.split('.')
            if len(parts) == 2:
                category, key = parts
                if category == 'DIR':
                    replacement = str(Path(os.getenv(key, '')))
                elif category == 'SECRET':
                    replacement = os.getenv(key, '')
                else:
                    replacement = os.getenv(match, '')
                
                value = value.replace('{{' + match + '}}', replacement)
        
        return value

    @classmethod
    def create_dynamic_model(cls, **data):
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = cls.create_dynamic_model(**value)
        
        DynamicModel = create_model(
            f'Dynamic{cls.__name__}',
            __base__=cls,
            **{k: (type(v), v) for k, v in data.items()}
        )
        return DynamicModel(**data)

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

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


class Geocoder:
    def __init__(self, named_locs: Union[str, Path] = None, cache_file: Union[str, Path] = 'timezone_cache.json'):
        self.tf = TimezoneFinder()
        self.srtm_data = get_data()
        self.named_locs = Path(named_locs) if named_locs else None
        self.cache_file = Path(cache_file)
        self.last_timezone: str = "America/Los_Angeles"
        self.last_update: Optional[datetime] = None
        self.last_location: Optional[Tuple[float, float]] = None
        self.executor = ThreadPoolExecutor()
        self.override_locations = self.load_override_locations()

    def load_override_locations(self):
        if self.named_locs and self.named_locs.exists():
            with open(self.named_locs, 'r') as file:
                return yaml.safe_load(file)
        return []

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def find_override_location(self, lat: float, lon: float) -> Optional[str]:
        closest_location = None
        closest_distance = float('inf')
        
        for location in self.override_locations:
            loc_name = location.get("name")
            loc_lat = location.get("latitude")
            loc_lon = location.get("longitude")
            loc_radius = location.get("radius")
            
            distance = self.haversine(lat, lon, loc_lat, loc_lon)
            
            if distance <= loc_radius:
                if distance < closest_distance:
                    closest_distance = distance
                    closest_location = loc_name
        
        return closest_location

    async def location(self, lat: float, lon: float):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, rg.search, [(lat, lon)])
        override = self.find_override_location(lat, lon)
        if override:
            result[0]['override_name'] = override
        return result

    async def elevation(self, latitude: float, longitude: float, unit: str = "m") -> float:
        loop = asyncio.get_running_loop()
        elevation = await loop.run_in_executor(self.executor, self.srtm_data.get_elevation, latitude, longitude)
        
        if unit == "m":
            return elevation
        elif unit == "km":
            return elevation / 1000
        elif unit == "ft" or unit == "'":
            return elevation * 3.280839895
        else:
            raise ValueError(f"Unsupported unit: {unit}")


    async def timezone(self, lat: float, lon: float):
        loop = asyncio.get_running_loop()
        timezone = await loop.run_in_executor(self.executor, lambda: self.tf.timezone_at(lat=lat, lng=lon))
        return timezone if timezone else 'Unknown'


    async def lookup(self, lat: float, lon: float):
        city, state, country = (await self.location(lat, lon))[0]['name'], (await self.location(lat, lon))[0]['admin1'], (await self.location(lat, lon))[0]['cc']
        elevation = await self.elevation(lat, lon)
        timezone = await self.timezone(lat, lon)
        
        return {
            "city": city,
            "state": state,
            "country": country,
            "elevation": elevation,
            "timezone": timezone
        }

    async def code(self, locations: Union[Location, Tuple[float, float], List[Union[Location, Tuple[float, float]]]]) -> Union[Location, List[Location]]:
        if isinstance(locations, (Location, tuple)):
            locations = [locations]
        
        processed_locations = []
        for loc in locations:
            if isinstance(loc, tuple):
                processed_locations.append(Location(latitude=loc[0], longitude=loc[1]))
            elif isinstance(loc, Location):
                processed_locations.append(loc)
            else:
                raise ValueError(f"Unsupported location type: {type(loc)}")

        coordinates = [(location.latitude, location.longitude) for location in processed_locations]
        
        geocode_results = await asyncio.gather(*[self.location(lat, lon) for lat, lon in coordinates])
        elevations = await asyncio.gather(*[self.elevation(lat, lon) for lat, lon in coordinates])
        timezones = await asyncio.gather(*[self.timezone(lat, lon) for lat, lon in coordinates])

        geocoded_locations = []
        for location, result, elevation, timezone in zip(processed_locations, geocode_results, elevations, timezones):
            result = result[0]  # Unpack the first result
            override_name = result.get('override_name')
            geocoded_location = Location(
                latitude=location.latitude,
                longitude=location.longitude,
                elevation=elevation,
                datetime=location.datetime or datetime.now(timezone.utc),
                zip=result.get("admin2"),
                city=result.get("name"),
                state=result.get("admin1"),
                country=result.get("cc"),
                context=location.context or {},
                name=override_name or result.get("name"),
                display_name=f"{override_name or result.get('name')}, {result.get('admin1')}, {result.get('cc')}",
                country_code=result.get("cc"),
                timezone=timezone
            )

            # Merge original location data with geocoded data
            for field in location.__fields__:
                if getattr(location, field) is None:
                    setattr(location, field, getattr(geocoded_location, field))

            geocoded_locations.append(location)

        return geocoded_locations[0] if len(geocoded_locations) == 1 else geocoded_locations

    async def geocode_osm(self, latitude: float, longitude: float, email: str):
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
        headers = {
            'User-Agent': f'sijapi/1.0 ({email})',  # replace with your app name and email
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
        
        address = data.get("address", {})
        elevation = await self.elevation(latitude, longitude)
        return Location(
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            datetime=datetime.now(timezone.utc),
            zip=address.get("postcode"),
            street=address.get("road"),
            city=address.get("city"),
            state=address.get("state"),
            country=address.get("country"),
            context={}, 
            class_=data.get("class"),
            type=data.get("type"),
            name=data.get("name"),
            display_name=data.get("display_name"),
            amenity=address.get("amenity"),
            house_number=address.get("house_number"),
            road=address.get("road"),
            quarter=address.get("quarter"),
            neighbourhood=address.get("neighbourhood"),
            suburb=address.get("suburb"),
            county=address.get("county"),
            country_code=address.get("country_code"),
            timezone=await self.timezone(latitude, longitude)
        )


    def round_coords(self, lat: float, lon: float, decimal_places: int = 2) -> Tuple[float, float]:
        return (round(lat, decimal_places), round(lon, decimal_places))

    def coords_equal(self, coord1: Tuple[float, float], coord2: Tuple[float, float], tolerance: float = 1e-5) -> bool:
        return math.isclose(coord1[0], coord2[0], abs_tol=tolerance) and math.isclose(coord1[1], coord2[1], abs_tol=tolerance)

    async def refresh_timezone(self, location: Union[Location, Tuple[float, float]], force: bool = False) -> str:
        if isinstance(location, Location):
            lat, lon = location.latitude, location.longitude
        else:
            lat, lon = location

        rounded_location = self.round_coords(lat, lon)
        current_time = datetime.now()

        if (force or
            not self.last_update or
            current_time - self.last_update > timedelta(hours=1) or
            not self.coords_equal(rounded_location, self.round_coords(*self.last_location) if self.last_location else (None, None))):
            
            new_timezone = await self.timezone(lat, lon)
            self.last_timezone = new_timezone
            self.last_update = current_time
            self.last_location = (lat, lon)  # Store the original, non-rounded coordinates
            await self.tz_save()
        
        return self.last_timezone

    async def tz_save(self):
        cache_data = {
            'last_timezone': self.last_timezone,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'last_location': self.last_location
        }
        async with aiofiles.open(self.cache_file, 'w') as f:
            await f.write(json.dumps(cache_data))

    async def tz_cached(self):
        try:
            async with aiofiles.open(self.cache_file, 'r') as f:
                cache_data = json.loads(await f.read())
            self.last_timezone = cache_data.get('last_timezone')
            self.last_update = datetime.fromisoformat(cache_data['last_update']) if cache_data.get('last_update') else None
            self.last_location = tuple(cache_data['last_location']) if cache_data.get('last_location') else None
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is invalid, we'll start fresh
            pass

    async def tz_current(self, location: Union[Location, Tuple[float, float]]) -> str:
        await self.tz_cached()
        return await self.refresh_timezone(location)

    async def tz_last(self) -> Optional[str]:
        await self.tz_cached()
        return self.last_timezone
    

    async def tz_at(self, lat: float, lon: float) -> str:
        """
        Get the timezone at a specific latitude and longitude without affecting the cache.
        
        :param lat: Latitude
        :param lon: Longitude
        :return: Timezone string
        """
        return await self.timezone(lat, lon)

    def __del__(self):
        self.executor.shutdown()


class Database(BaseModel):
    host: str = Field(..., description="Database host")
    port: int = Field(5432, description="Database port")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    database: str = Field(..., description="Database name")
    db_schema: Optional[str] = Field(None, description="Database schema")

    @asynccontextmanager
    async def get_connection(self):
        conn = await asyncpg.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )
        try:
            if self.db_schema:
                await conn.execute(f"SET search_path TO {self.db_schema}")
            yield conn
        finally:
            await conn.close()

    @classmethod
    def from_env(cls):
        import os
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            db_schema=os.getenv("DB_SCHEMA")
        )

    def to_dict(self):
        return self.dict(exclude_none=True)


class IMAPConfig(BaseModel):
    username: str
    password: str
    host: str
    port: int
    encryption: str = None

class SMTPConfig(BaseModel):
    username: str
    password: str
    host: str
    port: int
    encryption: str = None

class AutoResponder(BaseModel):
    name: str
    style: str
    context: str
    ollama_model: str = "llama3"
    whitelist: List[str]
    blacklist: List[str]
    image_prompt: Optional[str] = None
    image_scene:  Optional[str] = None
    smtp: SMTPConfig
    
class EmailAccount(BaseModel):
    name: str
    refresh: int
    fullname: Optional[str]
    bio: Optional[str]
    summarize: bool = False
    podcast: bool = False
    imap: IMAPConfig
    autoresponders: Optional[List[AutoResponder]]

class EmailContact(BaseModel):
    email: str
    name: Optional[str] = None

class IncomingEmail(BaseModel):
    sender: str
    datetime_received: datetime
    recipients: List[EmailContact]
    subject: str
    body: str
    attachments: List[dict] = []
