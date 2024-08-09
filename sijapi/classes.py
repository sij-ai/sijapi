# classes.py
import json
import yaml
import math
import os
import re
import uuid
import time
import aiofiles
import aiohttp
import asyncio
import asyncpg
import socket
import traceback
from tqdm.asyncio import tqdm
import reverse_geocoder as rg
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, ClassVar
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model, PrivateAttr
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
from srtm import get_data
import os
import sys
from loguru import logger

# Custom logger class
class Logger:
    def __init__(self, name, logs_dir):
        self.logs_dir = logs_dir
        self.name = name
        self.logger = logger
        self.debug_modules = set()

    def setup_from_args(self, args):
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        self.logger.remove()
        
        log_format = "{time:YYYY-MM-DD HH:mm:ss} - {name} - <level>{level: <8}</level> - <level>{message}</level>"
        
        # File handler
        self.logger.add(os.path.join(self.logs_dir, 'app.log'), rotation="2 MB", level="DEBUG", format=log_format)
        
        # Set debug modules
        self.debug_modules = set(args.debug)
        
        # Console handler with custom filter
        def module_filter(record):
            return (record["level"].no >= logger.level(args.log.upper()).no or
                    record["name"] in self.debug_modules)
        
        self.logger.add(sys.stdout, level="DEBUG", format=log_format, filter=module_filter, colorize=True)
        
        # Custom color and style mappings
        self.logger.level("CRITICAL", color="<yellow><bold><MAGENTA>")
        self.logger.level("ERROR", color="<red><bold>")
        self.logger.level("WARNING", color="<yellow><bold>")
        self.logger.level("DEBUG", color="<green><bold>")
        
        self.logger.info(f"Debug modules: {self.debug_modules}")

    def get_module_logger(self, module_name):
        return self.logger.bind(name=module_name)

L = Logger("classes", "classes")
logger = L.get_module_logger("classes")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(ENV_PATH)
TS_ID = os.environ.get('TS_ID')
T = TypeVar('T', bound='Configuration')


class Configuration(BaseModel):
    HOME: Path = Path.home()
    _dir_config: Optional['Configuration'] = None

    @classmethod
    def load(cls, yaml_path: Union[str, Path], secrets_path: Optional[Union[str, Path]] = None, dir_config: Optional['Configuration'] = None) -> 'Configuration':
        yaml_path = cls._resolve_path(yaml_path, 'config')
        if secrets_path:
            secrets_path = cls._resolve_path(secrets_path, 'config')

        try:
            with yaml_path.open('r') as file:
                config_data = yaml.safe_load(file)

            debug(f"Loaded configuration data from {yaml_path}")
            if secrets_path:
                with secrets_path.open('r') as file:
                    secrets_data = yaml.safe_load(file)
                debug(f"Loaded secrets data from {secrets_path}")
                if isinstance(config_data, list):
                    for item in config_data:
                        if isinstance(item, dict):
                            item.update(secrets_data)
                else:
                    config_data.update(secrets_data)
            if isinstance(config_data, list):
                config_data = {"configurations": config_data}
            if config_data.get('HOME') is None:
                config_data['HOME'] = str(Path.home())
                warn(f"HOME was None in config, set to default: {config_data['HOME']}")

            load_dotenv()
            instance = cls.create_dynamic_model(**config_data)
            instance._dir_config = dir_config or instance
            resolved_data = instance.resolve_placeholders(config_data)
            instance = cls.create_dynamic_model(**resolved_data)
            instance._dir_config = dir_config or instance
            return instance

        except Exception as e:
            err(f"Error loading configuration: {str(e)}")
            raise

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
            resolved_data = {k: self.resolve_placeholders(v) for k, v in data.items()}
            home_dir = Path(resolved_data.get('HOME', self.HOME)).expanduser()
            base_dir = Path(__file__).parent.parent
            data_dir = base_dir / "data"
            resolved_data['HOME'] = str(home_dir)
            resolved_data['BASE'] = str(base_dir)
            resolved_data['DATA'] = str(data_dir)
            return resolved_data
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
                replacement = getattr(self, parts[0], str(Path.home() / parts[0].lower()))
            elif len(parts) == 2 and parts[0] == 'Dir':
                replacement = getattr(self, parts[1], str(Path.home() / parts[1].lower()))
            elif len(parts) == 2 and parts[0] == 'ENV':
                replacement = os.getenv(parts[1], '')
            else:
                replacement = value

            value = value.replace('{{' + match + '}}', str(replacement))

        # Convert to Path if it looks like a file path
        if isinstance(value, str) and (value.startswith(('/', '~')) or (':' in value and value[1] == ':')):
            return Path(value).expanduser()
        return value

    @classmethod
    def create_dynamic_model(cls, **data):
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = cls.create_dynamic_model(**value)
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                data[key] = [cls.create_dynamic_model(**item) for item in value]

        DynamicModel = create_model(
            f'Dynamic{cls.__name__}',
            __base__=cls,
            **{k: (Any, v) for k, v in data.items()}
        )
        return DynamicModel(**data)

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


class DirConfig(BaseModel):
    HOME: Path = Path.home()

    @classmethod
    def load(cls, yaml_path: Union[str, Path]) -> 'DirConfig':
        yaml_path = cls._resolve_path(yaml_path, 'config')

        try:
            with yaml_path.open('r') as file:
                config_data = yaml.safe_load(file)

            print(f"Loaded configuration data from {yaml_path}")

            # Ensure HOME is set
            if 'HOME' not in config_data:
                config_data['HOME'] = str(Path.home())
                print(f"HOME was not in config, set to default: {config_data['HOME']}")

            instance = cls.create_dynamic_model(**config_data)
            resolved_data = instance.resolve_placeholders(config_data)
            return cls.create_dynamic_model(**resolved_data)

        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            raise

    @classmethod
    def _resolve_path(cls, path: Union[str, Path], default_dir: str) -> Path:
        base_path = Path(__file__).parent.parent
        path = Path(path)
        if not path.suffix:
            path = base_path / 'sijapi' / default_dir / f"{path.name}.yaml"
        elif not path.is_absolute():
            path = base_path / path
        return path

    def resolve_placeholders(self, data: Any) -> Any:
        if isinstance(data, dict):
            resolved_data = {k: self.resolve_placeholders(v) for k, v in data.items()}
            home_dir = Path(resolved_data.get('HOME', self.HOME)).expanduser()
            base_dir = Path(__file__).parent.parent
            data_dir = base_dir / "data"
            resolved_data['HOME'] = str(home_dir)
            resolved_data['BASE'] = str(base_dir)
            resolved_data['DATA'] = str(data_dir)
            return resolved_data
        elif isinstance(data, list):
            return [self.resolve_placeholders(v) for v in data]
        elif isinstance(data, str):
            return self.resolve_string_placeholders(data)
        else:
            return data

    def resolve_string_placeholders(self, value: str) -> Path:
        pattern = r'\{\{\s*([^}]+)\s*\}\}'
        matches = re.findall(pattern, value)

        for match in matches:
            if match == 'HOME':
                replacement = str(self.HOME)
            elif hasattr(self, match):
                replacement = str(getattr(self, match))
            else:
                replacement = value

            value = value.replace('{{' + match + '}}', replacement)

        return Path(value).expanduser()

    @classmethod
    def create_dynamic_model(cls, **data):
        DynamicModel = create_model(
            f'Dynamic{cls.__name__}',
            __base__=cls,
            **{k: (Path, v) for k, v in data.items()}
        )
        return DynamicModel(**data)

    class Config:
        arbitrary_types_allowed = True



# Configuration class for API & Database methods.
class APIConfig(BaseModel):
    HOST: str
    PORT: int
    BIND: str
    URL: str
    PUBLIC: List[str]
    TRUSTED_SUBNETS: List[str]
    MODULES: Any
    POOL: List[Dict[str, Any]]
    EXTENSIONS: Any
    TZ: str
    KEYS: List[str]
    GARBAGE: Dict[str, Any]
    SPECIAL_TABLES: ClassVar[List[str]] = ['spatial_ref_sys']
    db_pools: Dict[str, Any] = Field(default_factory=dict)
    offline_servers: Dict[str, float] = Field(default_factory=dict)
    offline_timeout: float = Field(default=30.0)  # 30 second timeout for offline servers
    online_hosts_cache: Dict[str, Tuple[List[Dict[str, Any]], float]] = Field(default_factory=dict)
    online_hosts_cache_ttl: float = Field(default=30.0)  # Cache TTL in seconds

    def __init__(self, **data):
        super().__init__(**data)
        self.db_pools = {}
        self.online_hosts_cache = {}  # Initialize the cache
        self._sync_tasks = {}

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def load(cls, config_path: Union[str, Path], secrets_path: Union[str, Path]):
        config_path = cls._resolve_path(config_path, 'config')
        secrets_path = cls._resolve_path(secrets_path, 'config')

        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)

        debug(f"Loaded main config: {config_data}")

        try:
            with open(secrets_path, 'r') as file:
                secrets_data = yaml.safe_load(file)
        except FileNotFoundError:
            warn(f"Secrets file not found: {secrets_path}")
            secrets_data = {}
        except yaml.YAMLError as e:
            err(f"Error parsing secrets YAML: {e}")
            secrets_data = {}

        config_data = cls.resolve_placeholders(config_data)
        debug(f"Resolved config: {config_data}")
        if isinstance(config_data.get('KEYS'), list) and len(config_data['KEYS']) == 1:
            placeholder = config_data['KEYS'][0]
            if placeholder.startswith('{{') and placeholder.endswith('}}'):
                key = placeholder[2:-2].strip()
                parts = key.split('.')
                if len(parts) == 2 and parts[0] == 'SECRET':
                    secret_key = parts[1]
                    if secret_key in secrets_data:
                        config_data['KEYS'] = secrets_data[secret_key]
                        debug(f"Replaced KEYS with secret: {config_data['KEYS']}")
                    else:
                        warn(f"Secret key '{secret_key}' not found in secrets file")
                else:
                    warn(f"Invalid secret placeholder format: {placeholder}")

        config_data['MODULES'] = cls._create_dynamic_config(config_data.get('MODULES', {}), 'DynamicModulesConfig')
        config_data['EXTENSIONS'] = cls._create_dynamic_config(config_data.get('EXTENSIONS', {}), 'DynamicExtensionsConfig')
        return cls(**config_data)

    @classmethod
    def _create_dynamic_config(cls, data: Dict[str, Any], model_name: str):
        fields = {}
        for key, value in data.items():
            if isinstance(value, str):
                fields[key] = (bool, value.lower() == 'on')
            elif isinstance(value, bool):
                fields[key] = (bool, value)
            else:
                raise ValueError(f"Invalid value for {key}: {value}. Must be 'on', 'off', True, or False.")
    
        DynamicConfig = create_model(model_name, **fields)
        instance_data = {k: (v.lower() == 'on' if isinstance(v, str) else v) for k, v in data.items()}
        return DynamicConfig(**instance_data)


    @classmethod
    def _resolve_path(cls, path: Union[str, Path], default_dir: str) -> Path:
        base_path = Path(__file__).parent.parent
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

        if 'BIND' in resolved_data:
            resolved_data['BIND'] = resolved_data['BIND'].replace('{{ HOST }}', str(resolved_data['HOST']))
            resolved_data['BIND'] = resolved_data['BIND'].replace('{{ PORT }}', str(resolved_data['PORT']))

        return resolved_data

    def __getattr__(self, name: str) -> Any:
        if name in ['MODULES', 'EXTENSIONS']:
            return self.__dict__[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def active_modules(self) -> List[str]:
        return [module for module, is_active in self.MODULES.__dict__.items() if is_active]

    @property
    def active_extensions(self) -> List[str]:
        return [extension for extension, is_active in self.EXTENSIONS.__dict__.items() if is_active]

    @property
    def local_db(self):
        ts_id = os.environ.get('TS_ID')
        local_db = next((db for db in self.POOL if db['ts_id'] == ts_id), None)
        if local_db is None:
            raise ValueError(f"No database configuration found for TS_ID: {ts_id}")
        return local_db

    async def get_online_hosts(self) -> List[Dict[str, Any]]:
        current_time = time.time()
        cache_key = "online_hosts"

        if cache_key in self.online_hosts_cache:
            cached_hosts, cache_time = self.online_hosts_cache[cache_key]
            if current_time - cache_time < self.online_hosts_cache_ttl:
                return cached_hosts

        online_hosts = []
        local_ts_id = os.environ.get('TS_ID')

        for pool_entry in self.POOL:
            # omit self from online hosts
            # if pool_entry['ts_id'] != local_ts_id:
            pool_key = f"{pool_entry['ts_ip']}:{pool_entry['db_port']}"
            if pool_key in self.offline_servers:
                if current_time - self.offline_servers[pool_key] < self.offline_timeout:
                    continue
                else:
                    del self.offline_servers[pool_key]

            conn = await self.get_connection(pool_entry)
            if conn is not None:
                online_hosts.append(pool_entry)
                await conn.close()

        self.online_hosts_cache[cache_key] = (online_hosts, current_time)
        return online_hosts

    async def get_connection(self, pool_entry: Dict[str, Any] = None):
        if pool_entry is None:
            pool_entry = self.local_db

        pool_key = f"{pool_entry['ts_ip']}:{pool_entry['db_port']}"

        # Check if the server is marked as offline
        if pool_key in self.offline_servers:
            if time.time() - self.offline_servers[pool_key] < self.offline_timeout:
                return None
            else:
                del self.offline_servers[pool_key]

        if pool_key not in self.db_pools:
            try:
                self.db_pools[pool_key] = await asyncpg.create_pool(
                    host=pool_entry['ts_ip'],
                    port=pool_entry['db_port'],
                    user=pool_entry['db_user'],
                    password=pool_entry['db_pass'],
                    database=pool_entry['db_name'],
                    min_size=1,
                    max_size=10,
                    timeout=5
                )
            except Exception as e:
                warn(f"Failed to create connection pool for {pool_key}: {str(e)}")
                self.offline_servers[pool_key] = time.time()
                return None

        try:
            return await asyncio.wait_for(self.db_pools[pool_key].acquire(), timeout=5)
        except asyncio.TimeoutError:
            warn(f"Timeout acquiring connection from pool for {pool_key}")
            self.offline_servers[pool_key] = time.time()
            return None
        except Exception as e:
            warn(f"Failed to acquire connection for {pool_key}: {str(e)}")
            self.offline_servers[pool_key] = time.time()
            return None


    async def initialize_sync(self):
        local_ts_id = os.environ.get('TS_ID')
        online_hosts = await self.get_online_hosts()

        for pool_entry in online_hosts:
            if pool_entry['ts_id'] == local_ts_id:
                continue  # Skip local database
            try:
                conn = await self.get_connection(pool_entry)
                if conn is None:
                    continue  # Skip this database if connection failed

                debug(f"Starting sync initialization for {pool_entry['ts_ip']}...")

                # Check PostGIS installation
                postgis_installed = await self.check_postgis(conn)
                if not postgis_installed:
                    warn(f"PostGIS is not installed on {pool_entry['ts_id']} ({pool_entry['ts_ip']}). Some spatial operations may fail.")

                tables = await conn.fetch("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                """)

                for table in tables:
                    table_name = table['tablename']
                    await self.ensure_sync_columns(conn, table_name)

                debug(f"Sync initialization complete for {pool_entry['ts_ip']}. All tables now have necessary sync columns and triggers.")

            except Exception as e:
                err(f"Error initializing sync for {pool_entry['ts_ip']}: {str(e)}")
                err(f"Traceback: {traceback.format_exc()}")

    def _schedule_sync_task(self, table_name: str, pk_value: Any, version: int, server_id: str):
        # Use a background task manager to handle syncing
        task_key = f"{table_name}:{pk_value}" if pk_value else table_name
        if task_key not in self._sync_tasks:
            self._sync_tasks[task_key] = asyncio.create_task(self._sync_changes(table_name, pk_value, version, server_id))


    async def ensure_sync_columns(self, conn, table_name):
        if conn is None or table_name in self.SPECIAL_TABLES:
            return None

        try:
            # Check if primary key exists
            primary_key = await conn.fetchval(f"""
                SELECT a.attname
                FROM   pg_index i
                JOIN   pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE  i.indrelid = '{table_name}'::regclass AND i.indisprimary;
            """)

            if not primary_key:
                # Add an id column as primary key if it doesn't exist
                await conn.execute(f"""
                    ALTER TABLE "{table_name}"
                    ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY;
                """)
                primary_key = 'id'

            # Ensure version column exists
            await conn.execute(f"""
                ALTER TABLE "{table_name}"
                ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;
            """)

            # Ensure server_id column exists
            await conn.execute(f"""
                ALTER TABLE "{table_name}"
                ADD COLUMN IF NOT EXISTS server_id TEXT DEFAULT '{os.environ.get('TS_ID')}';
            """)

            # Create or replace the trigger function
            await conn.execute(f"""
                CREATE OR REPLACE FUNCTION update_version_and_server_id()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.version = COALESCE(OLD.version, 0) + 1;
                    NEW.server_id = '{os.environ.get('TS_ID')}';
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Check if the trigger exists and create it if it doesn't
            trigger_exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_trigger
                    WHERE tgname = 'update_version_and_server_id_trigger'
                    AND tgrelid = '{table_name}'::regclass
                )
            """)

            if not trigger_exists:
                await conn.execute(f"""
                    CREATE TRIGGER update_version_and_server_id_trigger
                    BEFORE INSERT OR UPDATE ON "{table_name}"
                    FOR EACH ROW EXECUTE FUNCTION update_version_and_server_id();
                """)

            debug(f"Successfully ensured sync columns and trigger for table {table_name}")
            return primary_key

        except Exception as e:
            err(f"Error ensuring sync columns for table {table_name}: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
            return None

    async def check_postgis(self, conn):
        if conn is None:
            debug(f"Skipping offline server...")
            return None

        try:
            result = await conn.fetchval("SELECT PostGIS_version();")
            if result:
                debug(f"PostGIS version: {result}")
                return True
            else:
                warn("PostGIS is not installed or not working properly")
                return False
        except Exception as e:
            err(f"Error checking PostGIS: {str(e)}")
            return False


    async def pull_changes(self, source_pool_entry, batch_size=10000):
        if source_pool_entry['ts_id'] == os.environ.get('TS_ID'):
            debug("Skipping self-sync")
            return 0

        total_changes = 0
        source_id = source_pool_entry['ts_id']
        source_ip = source_pool_entry['ts_ip']
        dest_id = os.environ.get('TS_ID')
        dest_ip = self.local_db['ts_ip']

        info(f"Starting sync from source {source_id} ({source_ip}) to destination {dest_id} ({dest_ip})")

        source_conn = None
        dest_conn = None
        try:
            source_conn = await self.get_connection(source_pool_entry)
            if source_conn is None:
                warn(f"Unable to connect to source {source_id} ({source_ip}). Skipping sync.")
                return 0

            dest_conn = await self.get_connection(self.local_db)
            if dest_conn is None:
                warn(f"Unable to connect to local database. Skipping sync.")
                return 0

            tables = await source_conn.fetch("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
            """)

            for table in tables:
                table_name = table['tablename']
                try:
                    if table_name in self.SPECIAL_TABLES:
                        await self.sync_special_table(source_conn, dest_conn, table_name)
                    else:
                        primary_key = await self.ensure_sync_columns(dest_conn, table_name)
                        last_synced_version = await self.get_last_synced_version(dest_conn, table_name, source_id)

                        while True:
                            changes = await source_conn.fetch(f"""
                                SELECT * FROM "{table_name}"
                                WHERE version > $1 AND server_id = $2
                                ORDER BY version ASC
                                LIMIT $3
                            """, last_synced_version, source_id, batch_size)

                            if not changes:
                                break  # No more changes for this table

                            changes_count = await self.apply_batch_changes(dest_conn, table_name, changes, primary_key)
                            total_changes += changes_count

                            if changes_count > 0:
                                info(f"Synced batch for {table_name}: {changes_count} changes. Total so far: {total_changes}")

                            last_synced_version = changes[-1]['version']  # Update last synced version

                except Exception as e:
                    err(f"Error syncing table {table_name}: {str(e)}")
                    err(f"Traceback: {traceback.format_exc()}")

            info(f"Sync complete from {source_id} ({source_ip}) to {dest_id} ({dest_ip}). Total changes: {total_changes}")

        except Exception as e:
            err(f"Error during sync process: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")

        finally:
            if source_conn:
                await source_conn.close()
            if dest_conn:
                await dest_conn.close()

        info(f"Sync summary:")
        info(f"  Total changes: {total_changes}")
        info(f"  Tables synced: {len(tables) if 'tables' in locals() else 0}")
        info(f"  Source: {source_id} ({source_ip})")
        info(f"  Destination: {dest_id} ({dest_ip})")

        return total_changes


    async def get_last_synced_version(self, conn, table_name, server_id):
        if conn is None:
            debug(f"Skipping offline server...")
            return 0

        if table_name in self.SPECIAL_TABLES:
            debug(f"Skipping get_last_synced_version because {table_name} is special.")
            return 0  # Special handling for tables without version column

        try:
            last_version = await conn.fetchval(f"""
                SELECT COALESCE(MAX(version), 0)
                FROM "{table_name}"
                WHERE server_id = $1
            """, server_id)
            return last_version
        except Exception as e:
            err(f"Error getting last synced version for table {table_name}: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
            return 0


    async def get_most_recent_source(self):
        most_recent_source = None
        max_version = -1
        local_ts_id = os.environ.get('TS_ID')
        online_hosts = await self.get_online_hosts()
        num_online_hosts = len(online_hosts)

        if num_online_hosts > 0:
            online_ts_ids = [host['ts_id'] for host in online_hosts if host['ts_id'] != local_ts_id]
            crit(f"Online hosts: {', '.join(online_ts_ids)}")

            for pool_entry in online_hosts:
                if pool_entry['ts_id'] == local_ts_id:
                    continue  # Skip local database

                try:
                    conn = await self.get_connection(pool_entry)
                    if conn is None:
                        warn(f"Unable to connect to {pool_entry['ts_id']}. Skipping.")
                        continue

                    tables = await conn.fetch("""
                        SELECT tablename FROM pg_tables
                        WHERE schemaname = 'public'
                    """)

                    for table in tables:
                        table_name = table['tablename']
                        if table_name in self.SPECIAL_TABLES:
                            continue
                        try:
                            result = await conn.fetchrow(f"""
                                SELECT MAX(version) as max_version, server_id
                                FROM "{table_name}"
                                WHERE version = (SELECT MAX(version) FROM "{table_name}")
                                GROUP BY server_id
                                ORDER BY MAX(version) DESC
                                LIMIT 1
                            """)
                            if result:
                                version, server_id = result['max_version'], result['server_id']
                                info(f"Max version for {pool_entry['ts_id']}, table {table_name}: {version} (from server {server_id})")
                                if version > max_version:
                                    max_version = version
                                    most_recent_source = pool_entry
                            else:
                                debug(f"No data in table {table_name} for {pool_entry['ts_id']}")
                        except asyncpg.exceptions.UndefinedColumnError:
                            warn(f"Version or server_id column does not exist in table {table_name} for {pool_entry['ts_id']}. Skipping.")
                        except Exception as e:
                            err(f"Error checking version for {pool_entry['ts_id']}, table {table_name}: {str(e)}")

                except asyncpg.exceptions.ConnectionFailureError:
                    warn(f"Failed to connect to database: {pool_entry['ts_ip']}:{pool_entry['db_port']}. Skipping.")
                except Exception as e:
                    err(f"Unexpected error occurred while checking version for {pool_entry['ts_id']}: {str(e)}")
                    err(f"Traceback: {traceback.format_exc()}")
                finally:
                    if conn:
                        await conn.close()

        if most_recent_source is None:
            if num_online_hosts > 0:
                warn("Could not determine most recent source. Using first available online host.")
                most_recent_source = next(host for host in online_hosts if host['ts_id'] != local_ts_id)
            else:
                crit("No other online hosts available for sync.")

        return most_recent_source



    async def _sync_changes(self, table_name: str, primary_key: str):
        try:
            local_conn = await self.get_connection()
            if local_conn is None:
                return

            # Get the latest changes
            changes = await local_conn.fetch(f"""
                SELECT * FROM "{table_name}"
                WHERE version > (SELECT COALESCE(MAX(version), 0) FROM "{table_name}" WHERE server_id != $1)
                OR (version = (SELECT COALESCE(MAX(version), 0) FROM "{table_name}" WHERE server_id != $1) AND server_id = $1)
                ORDER BY version ASC
            """, os.environ.get('TS_ID'))

            if changes:
                online_hosts = await self.get_online_hosts()
                for pool_entry in online_hosts:
                    if pool_entry['ts_id'] != os.environ.get('TS_ID'):
                        remote_conn = await self.get_connection(pool_entry)
                        if remote_conn is None:
                            continue
                        try:
                            await self.apply_batch_changes(remote_conn, table_name, changes, primary_key)
                        finally:
                            await remote_conn.close()
        except Exception as e:
            err(f"Error syncing changes for {table_name}: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
        finally:
            if 'local_conn' in locals():
                await local_conn.close()


    async def execute_read_query(self, query: str, *args, table_name: str):
        online_hosts = await self.get_online_hosts()
        results = []
        max_version = -1
        latest_result = None
        
        for pool_entry in online_hosts:
            conn = await self.get_connection(pool_entry)
            if conn is None:
                warn(f"Unable to connect to {pool_entry['ts_id']}. Skipping read.")
                continue
        
            try:
                # Execute the query
                result = await conn.fetch(query, *args)
                
                if not result:
                    continue
        
                # Check version if it's not a special table
                if table_name not in self.SPECIAL_TABLES:
                    try:
                        version_query = f"""
                            SELECT MAX(version) as max_version, server_id
                            FROM "{table_name}"
                            WHERE version = (SELECT MAX(version) FROM "{table_name}")
                            GROUP BY server_id
                            ORDER BY MAX(version) DESC
                            LIMIT 1
                        """
                        version_result = await conn.fetchrow(version_query)
                        if version_result:
                            version = version_result['max_version']
                            server_id = version_result['server_id']
                            info(f"Max version for {pool_entry['ts_id']}, table {table_name}: {version} (from server {server_id})")
                            if version > max_version:
                                max_version = version
                                latest_result = result
                        else:
                            debug(f"No version data in table {table_name} for {pool_entry['ts_id']}")
                            if latest_result is None:
                                latest_result = result
                    except asyncpg.exceptions.UndefinedColumnError:
                        warn(f"Version column does not exist in table {table_name} for {pool_entry['ts_id']}. Using result without version check.")
                        if latest_result is None:
                            latest_result = result
                else:
                    # For special tables, just use the first result
                    if latest_result is None:
                        latest_result = result
        
                results.append((pool_entry['ts_id'], result))
        
            except Exception as e:
                err(f"Error executing read query on {pool_entry['ts_id']}: {str(e)}")
                err(f"Traceback: {traceback.format_exc()}")
            finally:
                await conn.close()
        
        if not latest_result:
            warn(f"No results found for query on table {table_name}")
            return []
        
        # Log results from all databases
        for ts_id, result in results:
            info(f"Read result from {ts_id}: {result}")
        
        return [dict(r) for r in latest_result]  # Convert Record objects to dictionaries


    async def execute_write_query(self, query: str, *args, table_name: str):
        conn = await self.get_connection(self.local_db)
        if conn is None:
            err(f"Unable to connect to local database. Write operation failed.")
            return []
        
        try:
            # Execute the original query
            result = await conn.fetch(query, *args)
            
            if result:
                # Update version and server_id
                update_query = f"""
                UPDATE {table_name}
                SET version = COALESCE(version, 0) + 1,
                    server_id = $1
                WHERE id = $2
                RETURNING id, version
                """
                update_result = await conn.fetch(update_query, os.environ.get('TS_ID'), result[0]['id'])
                return update_result
            return result
        except Exception as e:
            err(f"Error executing write query: {str(e)}")
            err(f"Query: {query}")
            err(f"Args: {args}")
            err(f"Traceback: {traceback.format_exc()}")
            return []
        finally:
            await conn.close()


    
    async def get_table_columns(self, conn, table_name: str) -> List[str]:
        query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = $1
        ORDER BY ordinal_position
        """
        columns = await conn.fetch(query, table_name)
        return [col['column_name'] for col in columns]

    
    async def get_primary_key(self, conn, table_name: str) -> str:
        query = """
        SELECT a.attname
        FROM   pg_index i
        JOIN   pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE  i.indrelid = $1::regclass AND i.indisprimary
        """
        result = await conn.fetchval(query, table_name)
        return result



    async def _sync_write_to_other_dbs(self, query: str, args: tuple, table_name: str):
        local_ts_id = os.environ.get('TS_ID')
        online_hosts = await self.get_online_hosts()
        
        for pool_entry in online_hosts:
            if pool_entry['ts_id'] != local_ts_id:
                remote_conn = await self.get_connection(pool_entry)
                if remote_conn is None:
                    warn(f"Unable to connect to {pool_entry['ts_id']}. Skipping write operation.")
                    continue
        
                try:
                    await remote_conn.execute(query, *args)
                    debug(f"Successfully synced write operation to {pool_entry['ts_id']} for table {table_name}")
                except Exception as e:
                    err(f"Error executing write query on {pool_entry['ts_id']}: {str(e)}")
                finally:
                    await remote_conn.close()


    async def _run_sync_tasks(self, tasks):
        for task in tasks:
            try:
                await task
            except Exception as e:
                err(f"Error during background sync: {str(e)}")
                err(f"Traceback: {traceback.format_exc()}")


    async def push_change(self, table_name: str, pk_value: Any, version: int, server_id: str):
        asyncio.create_task(self._push_change_background(table_name, pk_value, version, server_id))

    async def _push_change_background(self, table_name: str, pk_value: Any, version: int, server_id: str):
        online_hosts = await self.get_online_hosts()
        successful_pushes = 0
        failed_pushes = 0

        for pool_entry in online_hosts:
            if pool_entry['ts_id'] != os.environ.get('TS_ID'):
                remote_conn = await self.get_connection(pool_entry)
                if remote_conn is None:
                    continue

                try:
                    local_conn = await self.get_connection()
                    if local_conn is None:
                        continue

                    try:
                        updated_row = await local_conn.fetchrow(f'SELECT * FROM "{table_name}" WHERE "{self.get_primary_key(table_name)}" = $1', pk_value)
                    finally:
                        await local_conn.close()

                    if updated_row:
                        columns = updated_row.keys()
                        placeholders = [f'${i+1}' for i in range(len(columns))]
                        primary_key = self.get_primary_key(table_name)

                        remote_version = await remote_conn.fetchval(f"""
                            SELECT version FROM "{table_name}"
                            WHERE "{primary_key}" = $1
                        """, updated_row[primary_key])

                        if remote_version is not None and remote_version >= updated_row['version']:
                            debug(f"Remote version for {table_name} in {pool_entry['ts_id']} is already up to date. Skipping push.")
                            successful_pushes += 1
                            continue

                        insert_query = f"""
                            INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
                            VALUES ({', '.join(placeholders)})
                            ON CONFLICT ("{primary_key}") DO UPDATE SET
                            {', '.join(f'"{col}" = EXCLUDED."{col}"' for col in columns if col != primary_key)},
                            version = EXCLUDED.version,
                            server_id = EXCLUDED.server_id
                            WHERE "{table_name}".version < EXCLUDED.version
                            OR ("{table_name}".version = EXCLUDED.version AND "{table_name}".server_id < EXCLUDED.server_id)
                        """
                        await remote_conn.execute(insert_query, *updated_row.values())
                        successful_pushes += 1
                except Exception as e:
                    err(f"Error pushing change to {pool_entry['ts_id']}: {str(e)}")
                    failed_pushes += 1
                finally:
                    if remote_conn:
                        await remote_conn.close()

        if successful_pushes > 0:
            info(f"Successfully pushed changes to {successful_pushes} server(s) for {table_name}")
        if failed_pushes > 0:
            warn(f"Failed to push changes to {failed_pushes} server(s) for {table_name}")


    async def push_all_changes(self, table_name: str):
        online_hosts = await self.get_online_hosts()
        tasks = []

        for pool_entry in online_hosts:
            if pool_entry['ts_id'] != os.environ.get('TS_ID'):
                task = asyncio.create_task(self._push_changes_to_host(pool_entry, table_name))
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_pushes = sum(1 for r in results if r is True)
        failed_pushes = sum(1 for r in results if r is False or isinstance(r, Exception))

        info(f"Push all changes summary for {table_name}: Successful: {successful_pushes}, Failed: {failed_pushes}")
        if failed_pushes > 0:
            warn(f"Failed to push all changes to {failed_pushes} server(s). Data may be out of sync.")

    async def _push_changes_to_host(self, pool_entry: Dict[str, Any], table_name: str) -> bool:
        remote_conn = None
        try:
            remote_conn = await self.get_connection(pool_entry)
            if remote_conn is None:
                warn(f"Unable to connect to {pool_entry['ts_id']}. Skipping push.")
                return False

            local_conn = await self.get_connection()
            if local_conn is None:
                warn(f"Unable to connect to local database. Skipping push.")
                return False

            try:
                all_rows = await local_conn.fetch(f'SELECT * FROM "{table_name}"')
            finally:
                await local_conn.close()

            if all_rows:
                columns = list(all_rows[0].keys())
                placeholders = [f'${i+1}' for i in range(len(columns))]

                insert_query = f"""
                    INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
                    VALUES ({', '.join(placeholders)})
                    ON CONFLICT DO NOTHING
                """

                async with remote_conn.transaction():
                    for row in all_rows:
                        await remote_conn.execute(insert_query, *row.values())

                return True
            else:
                debug(f"No rows to push for table {table_name}")
                return True
        except Exception as e:
            err(f"Error pushing all changes to {pool_entry['ts_id']}: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            if remote_conn:
                await remote_conn.close()

    async def _execute_special_table_write(self, conn, query: str, *args, table_name: str):
        if table_name == 'spatial_ref_sys':
            return await self._execute_spatial_ref_sys_write(conn, query, *args)

    async def _execute_spatial_ref_sys_write(self, local_conn, query: str, *args):
        # Execute the query locally
        result = await local_conn.execute(query, *args)

        # Sync the entire spatial_ref_sys table with all online hosts
        online_hosts = await self.get_online_hosts()
        for pool_entry in online_hosts:
            if pool_entry['ts_id'] != os.environ.get('TS_ID'):
                remote_conn = await self.get_connection(pool_entry)
                if remote_conn is None:
                    continue
                try:
                    await self.sync_spatial_ref_sys(local_conn, remote_conn)
                except Exception as e:
                    err(f"Error syncing spatial_ref_sys to {pool_entry['ts_id']}: {str(e)}")
                    err(f"Traceback: {traceback.format_exc()}")
                finally:
                    await remote_conn.close()

        return result

    async def apply_batch_changes(self, conn, table_name, changes, primary_key):
        if conn is None or not changes:
            debug(f"Skipping apply_batch_changes because conn is none or there are no changes.")
            return 0

        try:
            columns = list(changes[0].keys())
            placeholders = [f'${i+1}' for i in range(len(columns))]

            if primary_key:
                insert_query = f"""
                    INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
                    VALUES ({', '.join(placeholders)})
                    ON CONFLICT ("{primary_key}") DO UPDATE SET
                    {', '.join(f'"{col}" = EXCLUDED."{col}"' for col in columns if col not in [primary_key, 'version', 'server_id'])},
                    version = EXCLUDED.version,
                    server_id = EXCLUDED.server_id
                    WHERE "{table_name}".version < EXCLUDED.version
                    OR ("{table_name}".version = EXCLUDED.version AND "{table_name}".server_id < EXCLUDED.server_id)
                """
            else:
                warn(f"Possible source of issue #4")
                # For tables without a primary key, we'll use all columns for conflict resolution
                insert_query = f"""
                    INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
                    VALUES ({', '.join(placeholders)})
                    ON CONFLICT DO NOTHING
                """

            affected_rows = 0
            async for change in tqdm(changes, desc=f"Syncing {table_name}", unit="row"):
                values = [change[col] for col in columns]
                result = await conn.execute(insert_query, *values)
                affected_rows += int(result.split()[-1])

            return affected_rows

        except Exception as e:
            err(f"Error applying batch changes to {table_name}: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
            return 0

    async def sync_special_table(self, source_conn, dest_conn, table_name):
        if table_name == 'spatial_ref_sys':
            return await self.sync_spatial_ref_sys(source_conn, dest_conn)
        # Add more special cases as needed

    async def sync_spatial_ref_sys(self, source_conn, dest_conn):
        try:
            # Get all entries from the source
            source_entries = await source_conn.fetch("""
                SELECT * FROM spatial_ref_sys
                ORDER BY srid
            """)

            # Get all entries from the destination
            dest_entries = await dest_conn.fetch("""
                SELECT * FROM spatial_ref_sys
                ORDER BY srid
            """)

            # Convert to dictionaries for easier comparison
            source_dict = {entry['srid']: entry for entry in source_entries}
            dest_dict = {entry['srid']: entry for entry in dest_entries}

            updates = 0
            inserts = 0

            for srid, source_entry in source_dict.items():
                if srid not in dest_dict:
                    # Insert new entry
                    columns = source_entry.keys()
                    placeholders = [f'${i+1}' for i in range(len(columns))]
                    insert_query = f"""
                        INSERT INTO spatial_ref_sys ({', '.join(f'"{col}"' for col in columns)})
                        VALUES ({', '.join(placeholders)})
                    """
                    await dest_conn.execute(insert_query, *source_entry.values())
                    inserts += 1
                elif source_entry != dest_dict[srid]:
                    # Update existing entry
                    update_query = f"""
                        UPDATE spatial_ref_sys
                        SET auth_name = $1::text,
                            auth_srid = $2::integer,
                            srtext = $3::text,
                            proj4text = $4::text
                        WHERE srid = $5::integer
                    """
                    await dest_conn.execute(update_query,
                        source_entry['auth_name'],
                        source_entry['auth_srid'],
                        source_entry['srtext'],
                        source_entry['proj4text'],
                        srid
                    )
                    updates += 1

            info(f"spatial_ref_sys sync complete. Inserts: {inserts}, Updates: {updates}")
            return inserts + updates

        except Exception as e:
            err(f"Error syncing spatial_ref_sys table: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
            return 0




    async def add_primary_keys_to_local_tables(self):
        conn = await self.get_connection()

        debug(f"Adding primary keys to existing tables...")
        if conn is None:
            raise ConnectionError("Failed to connect to local database")

        try:
            tables = await conn.fetch("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
            """)

            for table in tables:
                table_name = table['tablename']
                if table_name not in self.SPECIAL_TABLES:
                    await self.ensure_sync_columns(conn, table_name)
        finally:
            await conn.close()

    async def add_primary_keys_to_remote_tables(self):
        online_hosts = await self.get_online_hosts()

        for pool_entry in online_hosts:
            conn = await self.get_connection(pool_entry)
            if conn is None:
                warn(f"Unable to connect to {pool_entry['ts_id']}. Skipping primary key addition.")
                continue

            try:
                info(f"Adding primary keys to existing tables on {pool_entry['ts_id']}...")
                tables = await conn.fetch("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                """)

                for table in tables:
                    table_name = table['tablename']
                    if table_name not in self.SPECIAL_TABLES:
                        primary_key = await self.ensure_sync_columns(conn, table_name)
                        if primary_key:
                            info(f"Added/ensured primary key '{primary_key}' for table '{table_name}' on {pool_entry['ts_id']}")
                        else:
                            warn(f"Failed to add/ensure primary key for table '{table_name}' on {pool_entry['ts_id']}")

                info(f"Completed adding primary keys to existing tables on {pool_entry['ts_id']}")
            except Exception as e:
                err(f"Error adding primary keys to existing tables on {pool_entry['ts_id']}: {str(e)}")
                err(f"Traceback: {traceback.format_exc()}")
            finally:
                await conn.close()


    async def close_db_pools(self):
        info("Closing database connection pools...")
        close_tasks = []
        for pool_key, pool in self.db_pools.items():
            close_tasks.append(self.close_pool_with_timeout(pool, pool_key))

        await asyncio.gather(*close_tasks)
        self.db_pools.clear()
        info("All database connection pools closed.")

    async def close_pool_with_timeout(self, pool, pool_key, timeout=10):
        try:
            await asyncio.wait_for(pool.close(), timeout=timeout)
            debug(f"Closed pool for {pool_key}")
        except asyncio.TimeoutError:
            err(f"Timeout closing pool for {pool_key}")
        except Exception as e:
            err(f"Error closing pool for {pool_key}: {str(e)}")


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
    class_: Optional[str] = Field(None, alias="class")
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
        populate_by_name = True

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
            'User-Agent': f'sijapi/1.0 ({email})',
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

class WidgetUpdate(BaseModel):
    text: Optional[str] = None
    progress: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    url: Optional[str] = None
    shortcut: Optional[str] = None
    graph: Optional[str] = None