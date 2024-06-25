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

from pydantic import BaseModel, Field
from typing import Optional

from pydantic import BaseModel, Field
from typing import Optional
import asyncpg

from pydantic import BaseModel, Field
from typing import Optional
import asyncpg
from contextlib import asynccontextmanager

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
    def __init__(self, db_config: Database, cache_file: str = 'timezone_cache.json'):
        self.db_config = db_config
        self.cache_file = cache_file
        self.last_timezone: str = "America/Los_Angeles"
        self.last_update: Optional[datetime] = None
        self.last_location: Optional[Tuple[float, float]] = None

    async def find(self, lat: float, lon: float) -> str:
        query = """
        SELECT tzid
        FROM timezones
        WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint($1, $2), 4326))
        LIMIT 1;
        """
        async with await self.db_config.get_connection() as conn:
            result = await conn.fetchrow(query, lon, lat)
            return result['tzid'] if result else 'Unknown'

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
            
            new_timezone = await self.find(lat, lon)
            
            self.last_timezone = new_timezone
            self.last_update = current_time
            self.last_location = (lat, lon)
            
            await self.save_to_cache()
            
            return new_timezone
        
        return self.last_timezone

    async def save_to_cache(self):
        cache_data = {
            'last_timezone': self.last_timezone,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'last_location': self.last_location
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f)

    async def load_from_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
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
