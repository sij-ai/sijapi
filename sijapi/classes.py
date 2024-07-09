# classes.py
import asyncio
import json
import math
import multiprocessing
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Type
from zoneinfo import ZoneInfo
import aiofiles
import aiohttp
import asyncpg
from typing import Union, Any
from pydantic import BaseModel, Field, ConfigDict
import reverse_geocoder as rg
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model, ConfigDict, validator
from srtm import get_data
from timezonefinder import TimezoneFinder
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, Type
import yaml
from typing import List, Optional
from dotenv import load_dotenv

T = TypeVar('T', bound='Configuration')

class HierarchicalPath(os.PathLike):
    def __init__(self, path=None, base=None, home=None):
        self.home = Path(home).expanduser() if home else Path.home()
        self.base = Path(base).resolve() if base else self._find_base()
        self.path = self._resolve_path(path) if path else self.base

    def _find_base(self):
        current = Path(__file__).resolve().parent
        while current.name != 'sijapi' and current != current.parent:
            current = current.parent
        return current

    def _resolve_path(self, path):
        if isinstance(path, HierarchicalPath):
            return path.path
        if isinstance(path, Path):
            return path
        path = self._resolve_placeholders(path)
        if path.startswith(('~', 'HOME')):
            return self.home / path.lstrip('~').lstrip('HOME').lstrip('/')
        if path.startswith('/'):
            return Path(path)
        return self._resolve_relative_path(self.base / path)

    def _resolve_placeholders(self, path):
        placeholders = {
            'HOME': str(self.home),
            'BASE': str(self.base),
        }
        pattern = r'\{\{\s*([^}]+)\s*\}\}'
        return re.sub(pattern, lambda m: placeholders.get(m.group(1).strip(), m.group(0)), path)

    def _resolve_relative_path(self, path):
        if path.is_file():
            return path
        if path.is_dir():
            return path
        yaml_path = path.with_suffix('.yaml')
        if yaml_path.is_file():
            return yaml_path
        return path

    def __truediv__(self, other):
        return HierarchicalPath(self.path / other, base=self.base, home=self.home)

    def __getattr__(self, name):
        return HierarchicalPath(self.path / name, base=self.base, home=self.home)

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return f"HierarchicalPath('{self.path}')"

    def __fspath__(self):
        return os.fspath(self.path)

    def __eq__(self, other):
        if isinstance(other, (HierarchicalPath, Path, str)):
            return str(self.path) == str(other)
        return False

    def __lt__(self, other):
        if isinstance(other, (HierarchicalPath, Path, str)):
            return str(self.path) < str(other)
        return False

    def __le__(self, other):
        if isinstance(other, (HierarchicalPath, Path, str)):
            return str(self.path) <= str(other)
        return False

    def __gt__(self, other):
        if isinstance(other, (HierarchicalPath, Path, str)):
            return str(self.path) > str(other)
        return False

    def __ge__(self, other):
        if isinstance(other, (HierarchicalPath, Path, str)):
            return str(self.path) >= str(other)
        return False

    def __hash__(self):
        return hash(self.path)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.path, name)


class Dir(BaseModel):
    HOME: HierarchicalPath = Field(default_factory=lambda: HierarchicalPath(Path.home()))
    BASE: HierarchicalPath | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def determine_base(cls) -> HierarchicalPath:
        return HierarchicalPath(HierarchicalPath()._find_base())

    def __init__(self, **data):
        super().__init__(**data)
        if self.BASE is None:
            self.BASE = self.determine_base()

    @classmethod
    def load(cls, yaml_path: Union[str, Path] = None) -> 'Dir':
        yaml_path = cls._resolve_path(yaml_path) if yaml_path else None
        if yaml_path:
            with open(yaml_path, 'r') as file:
                config_data = yaml.safe_load(file)
            print(f"Loaded directory configuration from {yaml_path}")
            resolved_data = cls.resolve_placeholders(config_data)
        else:
            resolved_data = {}
        return cls(**resolved_data)

    @classmethod
    def _resolve_path(cls, path: Union[str, Path]) -> Path:
        base_path = cls.determine_base().path.parent
        path = Path(path)
        if not path.suffix:
            path = base_path / 'sijapi' / 'config' / f"{path.name}.yaml"
        elif not path.is_absolute():
            path = base_path / path
        return path

    @classmethod
    def resolve_placeholders(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: cls.resolve_placeholders(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.resolve_placeholders(v) for v in data]
        elif isinstance(data, str):
            return cls.resolve_string_placeholders(data)
        return data

    @classmethod
    def resolve_string_placeholders(cls, value: str) -> Any:
        if value.startswith('{{') and value.endswith('}}'):
            parts = value.strip('{}').strip().split('.')
            result = cls.HOME
            for part in parts:
                result = getattr(result, part)
            return result
        elif value == '*~*':
            return cls.HOME
        return HierarchicalPath(value)

    def __getattr__(self, name):
        return HierarchicalPath(self.BASE / name.lower(), base=self.BASE.path, home=self.HOME.path)

    def model_dump(self, *args, **kwargs):
        d = super().model_dump(*args, **kwargs)
        return {k: str(v) for k, v in d.items()}



class Configuration(BaseModel):
    HOME: Path = Field(default_factory=Path.home)
    _dir_config: Optional['Configuration'] = None
    dir: Dir = Field(default_factory=Dir)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # This allows extra fields

    @classmethod
    def load(cls, yaml_path: Union[str, Path], secrets_path: Optional[Union[str, Path]] = None, dir_config: Optional['Configuration'] = None) -> 'Configuration':
        yaml_path = cls._resolve_path(yaml_path, 'config')
        if secrets_path:
            secrets_path = cls._resolve_path(secrets_path, 'config')

        try:
            with yaml_path.open('r') as file:
                config_data = yaml.safe_load(file)
            
            print(f"Loaded configuration data from {yaml_path}")
            
            if secrets_path:
                with secrets_path.open('r') as file:
                    secrets_data = yaml.safe_load(file)
                print(f"Loaded secrets data from {secrets_path}")
                config_data.update(secrets_data)
            
            instance = cls(**config_data)
            instance._dir_config = dir_config or instance

            resolved_data = instance.resolve_placeholders(config_data)
            
            return cls._create_nested_config(resolved_data)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            raise

    @classmethod
    def _create_nested_config(cls, data):
        if isinstance(data, dict):
            return cls(**{k: cls._create_nested_config(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [cls._create_nested_config(item) for item in data]
        else:
            return data

    def __getattr__(self, name):
        value = self.__dict__.get(name)
        if isinstance(value, dict):
            return Configuration(**value)
        return value

    @classmethod
    def _resolve_path(cls, path: Union[str, Path], default_dir: str) -> Path:
        base_path = Path(__file__).parent.parent  # This will be two levels up from this file
        path = Path(path)
        if not path.suffix:
            path = base_path / 'sijapi' / default_dir / f"{path.name}.yaml"
        elif not path.is_absolute():
            path = base_path / path
        return path

    def resolve_placeholders(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self.resolve_placeholders(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.resolve_placeholders(v) for v in data]
        elif isinstance(data, str):
            return self.resolve_string_placeholders(data)
        else:
            return data

    def resolve_string_placeholders(self, value: str) -> Any:
        pattern = r'\{\{\s*([^}]+)\s*\}\}'
        matches = re.findall(pattern, value)
        
        for match in matches:
            parts = match.split('.')
            if len(parts) == 1:  # Internal reference
                replacement = getattr(self._dir_config, parts[0], str(Path.home() / parts[0].lower()))
            elif len(parts) == 2 and parts[0] == 'Dir':
                replacement = getattr(self._dir_config, parts[1], str(Path.home() / parts[1].lower()))
            elif len(parts) == 2 and parts[0] == 'ENV':
                replacement = os.getenv(parts[1], '')
            else:
                replacement = value
            
            value = value.replace('{{' + match + '}}', str(replacement))
        
        # Convert to Path if it looks like a file path
        if isinstance(value, str) and (value.startswith(('/', '~')) or (':' in value and value[1] == ':')):
            return Path(value).expanduser()
        return value


class APIConfig(BaseModel):
    HOST: str
    PORT: int
    BIND: str
    URL: str
    PUBLIC: List[str]
    TRUSTED_SUBNETS: List[str]
    MODULES: Any  # This will be replaced with a dynamic model
    TZ: str
    KEYS: List[str]
    MAX_CPU_CORES: int = Field(default_factory=lambda: min(
        int(os.getenv("MAX_CPU_CORES", multiprocessing.cpu_count() // 2)), multiprocessing.cpu_count()
    ))

    @classmethod
    def load(cls, config_path: Union[str, Path], secrets_path: Union[str, Path]):
        config_path = cls._resolve_path(config_path, 'config')
        secrets_path = cls._resolve_path(secrets_path, 'config')

        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        print(f"Loaded main config: {config_data}")
        
        # Load secrets
        try:
            with open(secrets_path, 'r') as file:
                secrets_data = yaml.safe_load(file)
            print(f"Loaded secrets: {secrets_data}")
        except FileNotFoundError:
            print(f"Secrets file not found: {secrets_path}")
            secrets_data = {}
        except yaml.YAMLError as e:
            print(f"Error parsing secrets YAML: {e}")
            secrets_data = {}
        
        # Resolve internal placeholders
        config_data = cls.resolve_placeholders(config_data)
        
        print(f"Resolved config: {config_data}")
        
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
                        print(f"Replaced KEYS with secret: {config_data['KEYS']}")
                    else:
                        print(f"Secret key '{secret_key}' not found in secrets file")
                else:
                    print(f"Invalid secret placeholder format: {placeholder}")
        
        # Create dynamic ModulesConfig
        modules_data = config_data.get('MODULES', {})
        modules_fields = {}
        for key, value in modules_data.items():
            if isinstance(value, str):
                modules_fields[key] = (bool, value.lower() == 'on')
            elif isinstance(value, bool):
                modules_fields[key] = (bool, value)
            else:
                raise ValueError(f"Invalid value for module {key}: {value}. Must be 'on', 'off', True, or False.")

        DynamicModulesConfig = create_model('DynamicModulesConfig', **modules_fields)
        config_data['MODULES'] = DynamicModulesConfig(**modules_data)

        return cls(**config_data)

    @classmethod
    def _resolve_path(cls, path: Union[str, Path], default_dir: str) -> Path:
        base_path = Path(__file__).parent.parent  # This will be two levels up from this file
        path = Path(path)
        if not path.suffix:
            path = base_path / "sijapi" / default_dir / f"{path.name}.yaml"
        elif not path.is_absolute():
            path = base_path / path
        return path

    @classmethod
    def resolve_placeholders(cls, config_data: Dict[str, Any]) -> Dict[str, Any]:
        def resolve_value(value):
            if isinstance(value, str):
                pattern = r'\{\{\s*([^}]+)\s*\}\}'
                matches = re.findall(pattern, value)
                for match in matches:
                    if match in config_data:
                        value = value.replace(f'{{{{ {match} }}}}', str(config_data[match]))
            return value

        resolved_data = {}
        for key, value in config_data.items():
            if isinstance(value, dict):
                resolved_data[key] = cls.resolve_placeholders(value)
            elif isinstance(value, list):
                resolved_data[key] = [resolve_value(item) for item in value]
            else:
                resolved_data[key] = resolve_value(value)

        # Resolve BIND separately to ensure HOST and PORT are used
        if 'BIND' in resolved_data:
            resolved_data['BIND'] = resolved_data['BIND'].replace('{{ HOST }}', str(resolved_data['HOST']))
            resolved_data['BIND'] = resolved_data['BIND'].replace('{{ PORT }}', str(resolved_data['PORT']))

        return resolved_data

    def __getattr__(self, name: str) -> Any:
        if name == 'MODULES':
            return self.__dict__['MODULES']
        return super().__getattr__(name)

    @property
    def active_modules(self) -> List[str]:
        return [module for module, is_active in self.MODULES.__dict__.items() if is_active]

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

    async def timezone(self, lat: float, lon: float) -> Optional[ZoneInfo]:
        loop = asyncio.get_running_loop()
        timezone_str = await loop.run_in_executor(self.executor, lambda: self.tf.timezone_at(lat=lat, lng=lon))
        return ZoneInfo(timezone_str) if timezone_str else None


    
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
                processed_locations.append(Location(
                    latitude=loc[0], 
                    longitude=loc[1],
                    datetime=datetime.now(timezone.utc)
                ))
            elif isinstance(loc, Location):
                if loc.datetime is None:
                    loc.datetime = datetime.now(timezone.utc)
                processed_locations.append(loc)
            else:
                raise ValueError(f"Unsupported location type: {type(loc)}")

        coordinates = [(location.latitude, location.longitude) for location in processed_locations]
        
        geocode_results = await asyncio.gather(*[self.location(lat, lon) for lat, lon in coordinates])
        elevations = await asyncio.gather(*[self.elevation(lat, lon) for lat, lon in coordinates])
        timezone_results = await asyncio.gather(*[self.timezone(lat, lon) for lat, lon in coordinates])

                
        def create_display_name(override_name, result):
            parts = []
            if override_name:
                parts.append(override_name)
            if result.get('name') and result['name'] != override_name:
                parts.append(result['name'])
            if result.get('admin1'):
                parts.append(result['admin1'])
            if result.get('cc'):
                parts.append(result['cc'])
            return ', '.join(filter(None, parts))

        geocoded_locations = []
        for location, result, elevation, tz_result in zip(processed_locations, geocode_results, elevations, timezone_results):
            result = result[0]  # Unpack the first result
            override_name = result.get('override_name')
            geocoded_location = Location(
                latitude=location.latitude,
                longitude=location.longitude,
                elevation=elevation,
                datetime=location.datetime,
                zip=result.get("admin2"),
                city=result.get("name"),
                state=result.get("admin1"),
                country=result.get("cc"),
                context=location.context or {},
                name=override_name or result.get("name"),
                display_name=create_display_name(override_name, result),
                country_code=result.get("cc"),
                timezone=tz_result
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

    async def refresh_timezone(self, location: Union[Location, Tuple[float, float]], force: bool = False) -> Optional[ZoneInfo]:
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
            'last_timezone': str(self.last_timezone) if self.last_timezone else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'last_location': self.last_location
        }
        async with aiofiles.open(self.cache_file, 'w') as f:
            await f.write(json.dumps(cache_data))

    async def tz_cached(self):
        try:
            async with aiofiles.open(self.cache_file, 'r') as f:
                cache_data = json.loads(await f.read())
            self.last_timezone = ZoneInfo(cache_data['last_timezone']) if cache_data.get('last_timezone') else None
            self.last_update = datetime.fromisoformat(cache_data['last_update']) if cache_data.get('last_update') else None
            self.last_location = tuple(cache_data['last_location']) if cache_data.get('last_location') else None
        
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is invalid, we'll start fresh
            self.last_timezone = None
            self.last_update = None
            self.last_location = None

    async def tz_current(self, location: Union[Location, Tuple[float, float]]) -> Optional[ZoneInfo]:
        await self.tz_cached()
        return await self.refresh_timezone(location)

    async def tz_last(self) -> Optional[ZoneInfo]:
        await self.tz_cached()
        return self.last_timezone

    async def tz_at(self, lat: float, lon: float) -> Optional[ZoneInfo]:
        """
        Get the timezone at a specific latitude and longitude without affecting the cache.
        
        :param lat: Latitude
        :param lon: Longitude
        :return: ZoneInfo object representing the timezone
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
    def from_yaml(cls, yaml_path: Union[str, Path]):
        yaml_path = Path(yaml_path)
        if not yaml_path.is_absolute():
            yaml_path = Path(__file__).parent / 'config' / yaml_path

        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls(**config)

    def to_dict(self):
        return self.dict(exclude_none=True)



class IMAPConfig(BaseModel):
    username: str
    password: str
    host: str
    port: int
    encryption: Optional[str]

class SMTPConfig(BaseModel):
    username: str
    password: str
    host: str
    port: int
    encryption: Optional[str]

class AutoResponder(BaseModel):
    name: str
    style: str
    context: str
    ollama_model: str = "llama3"
    image_prompt: Optional[str] = None
    image_scene: Optional[str] = None

class AccountAutoResponder(BaseModel):
    name: str
    smtp: str
    whitelist: List[str]
    blacklist: List[str]

class EmailAccount(BaseModel):
    name: str
    fullname: Optional[str]
    bio: Optional[str]
    refresh: int
    summarize: bool = False
    podcast: bool = False
    imap: str
    autoresponders: List[AccountAutoResponder]

class EmailConfiguration(Configuration):
    imaps: List[IMAPConfig]
    smtps: List[SMTPConfig]
    autoresponders: List[AutoResponder]
    accounts: List[EmailAccount]

    def get_imap(self, username: str) -> Optional[IMAPConfig]:
        return next((imap for imap in self.imaps if imap.username == username), None)

    def get_smtp(self, username: str) -> Optional[SMTPConfig]:
        return next((smtp for smtp in self.smtps if smtp.username == username), None)

    def get_autoresponder(self, name: str) -> Optional[AutoResponder]:
        return next((ar for ar in self.autoresponders if ar.name == name), None)

    def get_account(self, name: str) -> Optional[EmailAccount]:
        return next((account for account in self.accounts if account.name == name), None)

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