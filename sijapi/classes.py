from pydantic import BaseModel
from typing import List, Optional, Any, Tuple, Dict, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import asyncpg
import json
from pydantic import BaseModel, Field
from typing import Optional
import asyncpg
import os
from typing import Optional, Tuple, Union
from datetime import datetime, timedelta
import json
from timezonefinder import TimezoneFinder
from pathlib import Path

from pydantic import BaseModel, Field
from typing import Optional

from pydantic import BaseModel, Field
from typing import Optional
import asyncpg

from pydantic import BaseModel, Field
from typing import Optional
import asyncpg
from contextlib import asynccontextmanager

import reverse_geocoder as rg
from timezonefinder import TimezoneFinder
from srtm import get_data

class PyGeolocator:
    def __init__(self):
        self.tf = TimezoneFinder()
        self.srtm_data = get_data()

    def get_location(self, lat, lon):
        result = rg.search((lat, lon))
        return result[0]['name'], result[0]['admin1'], result[0]['cc']

    def get_elevation(self, lat, lon):
        return self.srtm_data.get_elevation(lat, lon)

    def get_timezone(self, lat, lon):
        return self.tf.timezone_at(lat=lat, lng=lon)

    def lookup(self, lat, lon):
        city, state, country = self.get_location(lat, lon)
        elevation = self.get_elevation(lat, lon)
        timezone = self.get_timezone(lat, lon)
        
        return {
            "city": city,
            "state": state,
            "country": country,
            "elevation": elevation,
            "timezone": timezone
        }

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


class AutoResponder(BaseModel):
    name: str
    style: str
    context: str
    ollama_model: str = "llama3"
    whitelist: List[str]
    blacklist: List[str]
    image_prompt: Optional[str] = None

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

class EmailAccount(BaseModel):
    name: str
    refresh: int
    fullname: Optional[str]
    bio: Optional[str]
    summarize: bool = False
    podcast: bool = False
    imap: IMAPConfig
    smtp: SMTPConfig
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



class TimezoneTracker:
    def __init__(self, cache_file: Union[str, Path] = 'timezone_cache.json'):
        self.cache_file = Path(cache_file)
        self.last_timezone: str = "America/Los_Angeles"
        self.last_update: Optional[datetime] = None
        self.last_location: Optional[Tuple[float, float]] = None
        self.tf = TimezoneFinder()

    def find(self, lat: float, lon: float) -> str:
        timezone = self.tf.timezone_at(lat=lat, lng=lon)
        return timezone if timezone else 'Unknown'

    async def refresh(self, location: Union[Location, Tuple[float, float]], force: bool = False) -> str:
        if isinstance(location, Location):
            lat, lon = location.latitude, location.longitude
        else:
            lat, lon = location

        current_time = datetime.now()
        if (force or
            not self.last_update or
            current_time - self.last_update > timedelta(hours=1) or
            self.last_location != (lat, lon)):
            new_timezone = self.find(lat, lon)
            self.last_timezone = new_timezone
            self.last_update = current_time
            self.last_location = (lat, lon)
            await self.save_to_cache()
        return self.last_timezone

    async def save_to_cache(self):
        cache_data = {
            'last_timezone': self.last_timezone,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'last_location': self.last_location
        }
        with self.cache_file.open('w') as f:
            json.dump(cache_data, f)

    async def load_from_cache(self):
        try:
            with self.cache_file.open('r') as f:
                cache_data = json.load(f)
            self.last_timezone = cache_data.get('last_timezone')
            self.last_update = datetime.fromisoformat(cache_data['last_update']) if cache_data.get('last_update') else None
            self.last_location = tuple(cache_data['last_location']) if cache_data.get('last_location') else None
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is invalid, we'll start fresh
            pass

    async def get_current(self, location: Union[Location, Tuple[float, float]]) -> str:
        await self.load_from_cache()
        return await self.refresh(location)

    async def get_last(self) -> Optional[str]:
        await self.load_from_cache()
        return self.last_timezone
