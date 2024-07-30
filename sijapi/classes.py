# classes.py
import json
import yaml
import math
import os
import re
import uuid
import aiofiles
import aiohttp
import asyncio
import asyncpg
import socket
import traceback
from tqdm.asyncio import tqdm
import reverse_geocoder as rg
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
from srtm import get_data
from .logs import Logger

L = Logger("classes", "classes")
logger = L.get_module_logger("classes")

def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

T = TypeVar('T', bound='Configuration')

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(ENV_PATH)
TS_ID = os.environ.get('TS_ID')

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

            info(f"Loaded configuration data from {yaml_path}")
            if secrets_path:
                with secrets_path.open('r') as file:
                    secrets_data = yaml.safe_load(file)
                info(f"Loaded secrets data from {secrets_path}")
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
            return {k: self.resolve_placeholders(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.resolve_placeholders(v) for v in data]
        elif isinstance(data, str):
            return self.resolve_string_placeholders(data)
        else:
            return data

    def resolve_placeholders(self, data: Any) -> Any:
        if isinstance(data, dict):
            resolved_data = {k: self.resolve_placeholders(v) for k, v in data.items()}
            home = Path(resolved_data.get('HOME', self.HOME)).expanduser()
            sijapi = home / "workshop" / "sijapi"
            data_dir = sijapi / "data"
            resolved_data['HOME'] = str(home)
            resolved_data['SIJAPI'] = str(sijapi)
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
    _db_pools: Dict[str, asyncpg.Pool] = {}

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
        return DynamicConfig(**data)

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

    @asynccontextmanager
    async def get_connection(self, pool_entry: Dict[str, Any] = None):
        if pool_entry is None:
            pool_entry = self.local_db
        
        info(f"Attempting to connect to database: {pool_entry['ts_ip']}:{pool_entry['db_port']}")
        try:
            conn = await asyncpg.connect(
                host=pool_entry['ts_ip'],
                port=pool_entry['db_port'],
                user=pool_entry['db_user'],
                password=pool_entry['db_pass'],
                database=pool_entry['db_name'],
                timeout=5  # Add a timeout to prevent hanging
            )
            info(f"Successfully connected to {pool_entry['ts_ip']}:{pool_entry['db_port']}")
            try:
                yield conn
            finally:
                await conn.close()
                info(f"Closed connection to {pool_entry['ts_ip']}:{pool_entry['db_port']}")
        except asyncpg.exceptions.ConnectionDoesNotExistError:
            err(f"Failed to connect to database: {pool_entry['ts_ip']}:{pool_entry['db_port']} - Connection does not exist")
            raise
        except asyncpg.exceptions.ConnectionFailureError:
            err(f"Failed to connect to database: {pool_entry['ts_ip']}:{pool_entry['db_port']} - Connection failure")
            raise
        except asyncpg.exceptions.PostgresError as e:
            err(f"PostgreSQL error when connecting to {pool_entry['ts_ip']}:{pool_entry['db_port']}: {str(e)}")
            raise
        except Exception as e:
            err(f"Unexpected error when connecting to {pool_entry['ts_ip']}:{pool_entry['db_port']}: {str(e)}")
            raise



    async def close_db_pools(self):
        info("Closing database connection pools...")
        for pool_key, pool in self._db_pools.items():
            try:
                await pool.close()
                info(f"Closed pool for {pool_key}")
            except Exception as e:
                err(f"Error closing pool for {pool_key}: {str(e)}")
        self._db_pools.clear()
        info("All database connection pools closed.")


    async def initialize_sync(self):
        local_ts_id = os.environ.get('TS_ID')
        for pool_entry in self.POOL:
            if pool_entry['ts_id'] == local_ts_id:
                continue  # Skip local database
            try:
                async with self.get_connection(pool_entry) as conn:
                    info(f"Starting sync initialization for {pool_entry['ts_ip']}...")
                    
                    # Check PostGIS installation
                    postgis_installed = await self.check_postgis(conn)
                    if not postgis_installed:
                        warn(f"PostGIS is not installed on {pool_entry['ts_id']} ({pool_entry['ts_ip']}). Some spatial operations may fail.")
                    
                    # Initialize sync_status table
                    await self.initialize_sync_status_table(conn)
                    
                    # Continue with sync initialization
                    tables = await conn.fetch("""
                        SELECT tablename FROM pg_tables 
                        WHERE schemaname = 'public'
                    """)
                    
                    all_tables_synced = True
                    for table in tables:
                        table_name = table['tablename']
                        if not await self.ensure_sync_columns(conn, table_name):
                            all_tables_synced = False
                    
                    if all_tables_synced:
                        info(f"Sync initialization complete for {pool_entry['ts_ip']}. All tables now have version and server_id columns with appropriate triggers.")
                    else:
                        warn(f"Sync initialization partially complete for {pool_entry['ts_ip']}. Some tables may be missing version or server_id columns.")
                    
            except Exception as e:
                err(f"Error initializing sync for {pool_entry['ts_ip']}: {str(e)}")
                err(f"Traceback: {traceback.format_exc()}")



    async def initialize_sync_status_table(self, conn):
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_status (
                table_name TEXT,
                server_id TEXT,
                last_synced_version INTEGER,
                last_sync_time TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (table_name, server_id)
            )
        """)

        # Check if the last_sync_time column exists, and add it if it doesn't
        column_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'sync_status' AND column_name = 'last_sync_time'
            )
        """)

        if not column_exists:
            await conn.execute("""
                ALTER TABLE sync_status
                ADD COLUMN last_sync_time TIMESTAMP WITH TIME ZONE
            """)




    async def ensure_sync_structure(self, conn):
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        
        for table in tables:
            table_name = table['tablename']
            await self.ensure_sync_columns(conn, table_name)
            await self.ensure_sync_trigger(conn, table_name)


    async def ensure_sync_columns(self, conn, table_name):
        try:
            # Check if the table has a primary key
            has_primary_key = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.table_constraints
                    WHERE table_name = '{table_name}'
                    AND constraint_type = 'PRIMARY KEY'
                )
            """)

            # Check if version column exists
            version_exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = '{table_name}' AND column_name = 'version'
                )
            """)

            # Check if server_id column exists
            server_id_exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = '{table_name}' AND column_name = 'server_id'
                )
            """)

            # Add version column if it doesn't exist
            if not version_exists:
                await conn.execute(f"""
                    ALTER TABLE "{table_name}" ADD COLUMN version INTEGER DEFAULT 1
                """)

            # Add server_id column if it doesn't exist
            if not server_id_exists:
                await conn.execute(f"""
                    ALTER TABLE "{table_name}" ADD COLUMN server_id TEXT DEFAULT '{os.environ.get('TS_ID')}'
                """)

            # Create or replace the trigger function
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_version_and_server_id()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.version = COALESCE(OLD.version, 0) + 1;
                    NEW.server_id = $1;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Create the trigger if it doesn't exist
            trigger_exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT 1 FROM pg_trigger 
                    WHERE tgname = 'update_version_and_server_id_trigger' 
                    AND tgrelid = '{table_name}'::regclass
                )
            """)

            if not trigger_exists:
                await conn.execute(f"""
                    CREATE TRIGGER update_version_and_server_id_trigger
                    BEFORE INSERT OR UPDATE ON "{table_name}"
                    FOR EACH ROW EXECUTE FUNCTION update_version_and_server_id('{os.environ.get('TS_ID')}')
                """)
            
            info(f"Successfully ensured sync columns and trigger for table {table_name}. Has primary key: {has_primary_key}")
            return has_primary_key

        except Exception as e:
            err(f"Error ensuring sync columns for table {table_name}: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
            return False

    async def ensure_sync_trigger(self, conn, table_name):
        await conn.execute(f"""
            CREATE OR REPLACE FUNCTION update_version_and_server_id()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.version = COALESCE(OLD.version, 0) + 1;
                NEW.server_id = '{os.environ.get('TS_ID')}';
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            DROP TRIGGER IF EXISTS update_version_and_server_id_trigger ON "{table_name}";

            CREATE TRIGGER update_version_and_server_id_trigger
            BEFORE INSERT OR UPDATE ON "{table_name}"
            FOR EACH ROW EXECUTE FUNCTION update_version_and_server_id();
        """)

    async def get_most_recent_source(self):
        most_recent_source = None
        max_version = -1
        local_ts_id = os.environ.get('TS_ID')
        
        for pool_entry in self.POOL:
            if pool_entry['ts_id'] == local_ts_id:
                continue  # Skip local database
            
            if not await self.is_server_accessible(pool_entry['ts_ip'], pool_entry['db_port']):
                warn(f"Server {pool_entry['ts_id']} ({pool_entry['ts_ip']}:{pool_entry['db_port']}) is not accessible. Skipping.")
                continue

            try:
                async with self.get_connection(pool_entry) as conn:
                    tables = await conn.fetch("""
                        SELECT tablename FROM pg_tables 
                        WHERE schemaname = 'public'
                    """)
                    
                    for table in tables:
                        table_name = table['tablename']
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
                                info(f"No data in table {table_name} for {pool_entry['ts_id']}")
                        except asyncpg.exceptions.UndefinedColumnError:
                            warn(f"Version or server_id column does not exist in table {table_name} for {pool_entry['ts_id']}. Attempting to add...")
                            await self.ensure_sync_columns(conn, table_name)
                        except Exception as e:
                            err(f"Error checking version for {pool_entry['ts_id']}, table {table_name}: {str(e)}")

            except asyncpg.exceptions.ConnectionFailureError as e:
                err(f"Failed to establish database connection with {pool_entry['ts_id']} ({pool_entry['ts_ip']}:{pool_entry['db_port']}): {str(e)}")
            except Exception as e:
                err(f"Unexpected error occurred while checking version for {pool_entry['ts_id']}: {str(e)}")
                err(f"Traceback: {traceback.format_exc()}")
        
        return most_recent_source



    
    async def is_server_accessible(self, host, port, timeout=2):
        try:
            future = asyncio.open_connection(host, port)
            await asyncio.wait_for(future, timeout=timeout)
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, socket.gaierror):
            return False

    async def check_version_column_exists(self, conn):
        try:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND column_name = 'version'
                    AND table_name IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public')
                )
            """)
            if not result:
                tables_without_version = await conn.fetch("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename NOT IN (
                        SELECT table_name 
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' AND column_name = 'version'
                    )
                """)
                table_names = ", ".join([t['tablename'] for t in tables_without_version])
                warn(f"Tables without 'version' column: {table_names}")
            return result
        except Exception as e:
            err(f"Error checking for 'version' column existence: {str(e)}")
            return False


    async def pull_changes(self, source_pool_entry, batch_size=10000):
        if source_pool_entry['ts_id'] == os.environ.get('TS_ID'):
            info("Skipping self-sync")
            return 0

        total_changes = 0
        source_id = source_pool_entry['ts_id']
        source_ip = source_pool_entry['ts_ip']
        dest_id = os.environ.get('TS_ID')
        dest_ip = self.local_db['ts_ip']

        info(f"Starting sync from source {source_id} ({source_ip}) to destination {dest_id} ({dest_ip})")

        try:
            async with self.get_connection(source_pool_entry) as source_conn:
                async with self.get_connection(self.local_db) as dest_conn:
                    tables = await source_conn.fetch("""
                        SELECT tablename FROM pg_tables 
                        WHERE schemaname = 'public'
                    """)
                    
                    for table in tables:
                        table_name = table['tablename']
                        try:
                            debug(f"Processing table: {table_name}")
                            has_primary_key = await self.ensure_sync_columns(dest_conn, table_name)
                            debug(f"Table {table_name} has primary key: {has_primary_key}")
                            last_synced_version = await self.get_last_synced_version(dest_conn, table_name, source_id)
                            debug(f"Last synced version for {table_name}: {last_synced_version}")
                            
                            changes = await source_conn.fetch(f"""
                                SELECT * FROM "{table_name}"
                                WHERE version > $1 AND server_id = $2
                                ORDER BY version ASC
                                LIMIT $3
                            """, last_synced_version, source_id, batch_size)
                            
                            debug(f"Number of changes for {table_name}: {len(changes)}")
                            
                            if changes:
                                changes_count = await self.apply_batch_changes(dest_conn, table_name, changes, has_primary_key)
                                total_changes += changes_count
                                
                                if changes_count > 0:
                                    last_synced_version = changes[-1]['version']
                                    await self.update_sync_status(dest_conn, table_name, source_id, last_synced_version)
                                
                                info(f"Synced batch for {table_name}: {changes_count} changes. Total so far: {total_changes}")
                            else:
                                info(f"No changes to sync for {table_name}")

                        except Exception as e:
                            err(f"Error syncing table {table_name}: {str(e)}")
                            err(f"Traceback: {traceback.format_exc()}")
                            # Continue with the next table

            info(f"Sync complete from {source_id} ({source_ip}) to {dest_id} ({dest_ip}). Total changes: {total_changes}")

        except Exception as e:
            err(f"Error during sync process: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")

        info(f"Sync summary:")
        info(f"  Total changes: {total_changes}")
        info(f"  Tables synced: {len(tables)}")
        info(f"  Source: {source_id} ({source_ip})")
        info(f"  Destination: {dest_id} ({dest_ip})")

        return total_changes


    async def apply_batch_changes(self, conn, table_name, changes, has_primary_key):
        if not changes:
            return 0

        try:
            columns = list(changes[0].keys())
            placeholders = [f'${i}' for i in range(1, len(columns) + 1)]
            
            if has_primary_key:
                insert_query = f"""
                    INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
                    VALUES ({', '.join(placeholders)})
                    ON CONFLICT ON CONSTRAINT {table_name}_pkey DO UPDATE SET
                    {', '.join(f'"{col}" = EXCLUDED."{col}"' for col in columns if col not in ['version', 'server_id'])},
                    version = EXCLUDED.version,
                    server_id = EXCLUDED.server_id
                    WHERE "{table_name}".version < EXCLUDED.version
                    OR ("{table_name}".version = EXCLUDED.version AND "{table_name}".server_id < EXCLUDED.server_id)
                """
            else:
                # For tables without a primary key, we'll use all columns for conflict detection
                insert_query = f"""
                    INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
                    VALUES ({', '.join(placeholders)})
                    ON CONFLICT ({', '.join(f'"{col}"' for col in columns if col not in ['version', 'server_id'])}) DO UPDATE SET
                    version = EXCLUDED.version,
                    server_id = EXCLUDED.server_id
                    WHERE "{table_name}".version < EXCLUDED.version
                    OR ("{table_name}".version = EXCLUDED.version AND "{table_name}".server_id < EXCLUDED.server_id)
                """

            debug(f"Generated insert query for {table_name}: {insert_query}")

            # Prepare the statement
            stmt = await conn.prepare(insert_query)

            # Convert list of dicts to list of tuples
            values_list = [tuple(change[col] for col in columns) for change in changes]
            
            # Use executemany for batch insert
            result = await stmt.executemany(values_list)
            affected_rows = len(values_list)  # Assume all rows were affected

            return affected_rows

        except Exception as e:
            err(f"Error applying batch changes to {table_name}: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
            return 0



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
                    debug(f"Inserting new entry for srid {srid}: {insert_query}")
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
                    debug(f"Updating entry for srid {srid}: {update_query}")
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


    async def push_changes_to_all(self):
        for pool_entry in self.POOL:
            if pool_entry['ts_id'] != os.environ.get('TS_ID'):
                try:
                    await self.push_changes_to_one(pool_entry)
                except Exception as e:
                    err(f"Error pushing changes to {pool_entry['ts_id']}: {str(e)}")

    async def push_changes_to_one(self, pool_entry):
        try:
            async with self.get_connection() as local_conn:
                async with self.get_connection(pool_entry) as remote_conn:
                    tables = await local_conn.fetch("""
                        SELECT tablename FROM pg_tables 
                        WHERE schemaname = 'public'
                    """)
                    
                    for table in tables:
                        table_name = table['tablename']
                        try:
                            last_synced_version = await self.get_last_synced_version(remote_conn, table_name, os.environ.get('TS_ID'))
                            
                            changes = await local_conn.fetch(f"""
                                SELECT * FROM "{table_name}"
                                WHERE version > $1 AND server_id = $2
                                ORDER BY version ASC
                            """, last_synced_version, os.environ.get('TS_ID'))
                            
                            if changes:
                                debug(f"Pushing changes for table {table_name}")
                                debug(f"Columns: {', '.join(changes[0].keys())}")
                                
                                columns = list(changes[0].keys())
                                placeholders = [f'${i+1}' for i in range(len(columns))]
                                
                                insert_query = f"""
                                    INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
                                    VALUES ({', '.join(placeholders)})
                                    ON CONFLICT (id) DO UPDATE SET
                                    {', '.join(f'"{col}" = EXCLUDED."{col}"' for col in columns if col != 'id')}
                                """
                                
                                debug(f"Insert query: {insert_query}")
                                
                                for change in changes:
                                    values = [change[col] for col in columns]
                                    await remote_conn.execute(insert_query, *values)
                                
                                await self.update_sync_status(remote_conn, table_name, os.environ.get('TS_ID'), changes[-1]['version'])
                        
                        except Exception as e:
                            err(f"Error pushing changes for table {table_name}: {str(e)}")
                            err(f"Traceback: {traceback.format_exc()}")
            
            info(f"Successfully pushed changes to {pool_entry['ts_id']}")
        except Exception as e:
            err(f"Error pushing changes to {pool_entry['ts_id']}: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")



    async def update_sync_status(self, conn, table_name, server_id, version):
        await conn.execute("""
            INSERT INTO sync_status (table_name, server_id, last_synced_version, last_sync_time)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (table_name, server_id) DO UPDATE
            SET last_synced_version = EXCLUDED.last_synced_version,
                last_sync_time = EXCLUDED.last_sync_time
        """, table_name, server_id, version)


    async def get_last_synced_version(self, conn, table_name, server_id):
        return await conn.fetchval(f"""
            SELECT COALESCE(MAX(version), 0)
            FROM "{table_name}"
            WHERE server_id = $1
        """, server_id)

    async def update_last_synced_version(self, conn, table_name, server_id, version):
        await conn.execute(f"""
            INSERT INTO "{table_name}" (server_id, version)
            VALUES ($1, $2)
            ON CONFLICT (server_id) DO UPDATE
            SET version = EXCLUDED.version
            WHERE "{table_name}".version < EXCLUDED.version
        """, server_id, version)

    async def get_schema_version(self, pool_entry):
        async with self.get_connection(pool_entry) as conn:
            return await conn.fetchval("""
                SELECT COALESCE(MAX(version), 0) FROM (
                    SELECT MAX(version) as version FROM pg_tables
                    WHERE schemaname = 'public'
                ) as subquery
            """)

    async def create_sequence_if_not_exists(self, conn, sequence_name):
        await conn.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_sequences WHERE schemaname = 'public' AND sequencename = '{sequence_name}') THEN
                CREATE SEQUENCE {sequence_name};
            END IF;
        END $$;
        """)

    async def check_postgis(self, conn):
        try:
            result = await conn.fetchval("SELECT PostGIS_version();")
            if result:
                info(f"PostGIS version: {result}")
                return True
            else:
                warn("PostGIS is not installed or not working properly")
                return False
        except Exception as e:
            err(f"Error checking PostGIS: {str(e)}")
            return False




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

    def model_dump(self):
            data = self.dict()
            data["datetime"] = self.datetime.isoformat() if self.datetime else None
            return data


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

class WidgetUpdate(BaseModel):
    text: Optional[str] = None
    progress: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    url: Optional[str] = None
    shortcut: Optional[str] = None
    graph: Optional[str] = None