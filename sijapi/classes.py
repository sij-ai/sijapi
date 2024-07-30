# classes.py
import asyncio
import json
import yaml
import math
import os
import re
import aiofiles
import aiohttp
import asyncpg
import traceback
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
        
        info(f"Attempting to connect to database: {pool_entry}")
        try:
            conn = await asyncpg.connect(
                host=pool_entry['ts_ip'],
                port=pool_entry['db_port'],
                user=pool_entry['db_user'],
                password=pool_entry['db_pass'],
                database=pool_entry['db_name']
            )
            try:
                yield conn
            finally:
                await conn.close()
        except Exception as e:
            warn(f"Failed to connect to database: {pool_entry['ts_ip']}:{pool_entry['db_port']}")
            err(f"Error: {str(e)}")
            raise

    async def initialize_sync(self):
        async with self.get_connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_status (
                    table_name TEXT,
                    server_id TEXT,
                    last_synced_version INTEGER,
                    PRIMARY KEY (table_name, server_id)
                )
            """)
            
            tables = await conn.fetch("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            
            for table in tables:
                table_name = table['tablename']
                await conn.execute(f"""
                    ALTER TABLE "{table_name}" 
                    ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1,
                    ADD COLUMN IF NOT EXISTS server_id TEXT DEFAULT '{os.environ.get('TS_ID')}';

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

                    INSERT INTO sync_status (table_name, server_id, last_synced_version)
                    VALUES ('{table_name}', '{os.environ.get('TS_ID')}', 0)
                    ON CONFLICT (table_name, server_id) DO NOTHING;
                """)

    async def get_most_recent_source(self):
        most_recent_source = None
        max_version = -1
        
        for pool_entry in self.POOL:
            if pool_entry['ts_id'] == os.environ.get('TS_ID'):
                continue
            
            try:
                async with self.get_connection(pool_entry) as conn:
                    version = await conn.fetchval("""
                        SELECT COALESCE(MAX(last_synced_version), -1) FROM sync_status
                    """)
                    if version > max_version:
                        max_version = version
                        most_recent_source = pool_entry
            except Exception as e:
                err(f"Error checking version for {pool_entry['ts_id']}: {str(e)}")
        
        return most_recent_source

    async def pull_changes(self, source_pool_entry):
        total_inserts = 0
        total_updates = 0
        table_changes = {}

        source_id = source_pool_entry['ts_id']
        source_ip = source_pool_entry['ts_ip']
        dest_id = os.environ.get('TS_ID')
        dest_ip = self.local_db['ts_ip']

        info(f"Starting comprehensive sync from source {source_id} ({source_ip}) to destination {dest_id} ({dest_ip})")

        try:
            async with self.get_connection(source_pool_entry) as source_conn:
                async with self.get_connection(self.local_db) as dest_conn:  # Connect to local DB explicitly
                    # Compare tables
                    source_tables = await self.get_tables(source_conn)
                    dest_tables = await self.get_tables(dest_conn)
                    
                    tables_only_in_source = set(source_tables) - set(dest_tables)
                    tables_only_in_dest = set(dest_tables) - set(source_tables)
                    common_tables = set(source_tables) & set(dest_tables)

                    info(f"Tables only in source: {tables_only_in_source}")
                    info(f"Tables only in destination: {tables_only_in_dest}")
                    info(f"Common tables: {common_tables}")

                    for table in common_tables:
                        await self.compare_table_structure(source_conn, dest_conn, table)
                        inserts, updates = await self.compare_and_sync_data(source_conn, dest_conn, table, source_id)
                        
                        total_inserts += inserts
                        total_updates += updates
                        table_changes[table] = {'inserts': inserts, 'updates': updates}

                    # Optionally, handle tables only in source
                    for table in tables_only_in_source:
                        warn(f"Table '{table}' exists in source but not in destination. Consider manual migration.")

            info(f"Comprehensive sync complete from {source_id} ({source_ip}) to {dest_id} ({dest_ip})")
            info(f"Total changes: {total_inserts} inserts, {total_updates} updates")
            info("Changes by table:")
            for table, changes in table_changes.items():
                info(f"  {table}: {changes['inserts']} inserts, {changes['updates']} updates")

        except Exception as e:
            err(f"Error during sync process: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")

        return total_inserts + total_updates



    async def get_tables(self, conn):
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        return [table['tablename'] for table in tables]
    
    async def compare_table_structure(self, source_conn, dest_conn, table_name):
        source_columns = await self.get_table_structure(source_conn, table_name)
        dest_columns = await self.get_table_structure(dest_conn, table_name)

        columns_only_in_source = set(source_columns.keys()) - set(dest_columns.keys())
        columns_only_in_dest = set(dest_columns.keys()) - set(source_columns.keys())
        common_columns = set(source_columns.keys()) & set(dest_columns.keys())

        info(f"Table {table_name}:")
        info(f"  Columns only in source: {columns_only_in_source}")
        info(f"  Columns only in destination: {columns_only_in_dest}")
        info(f"  Common columns: {common_columns}")

        for col in common_columns:
            if source_columns[col] != dest_columns[col]:
                warn(f"  Column {col} has different types: source={source_columns[col]}, dest={dest_columns[col]}")

    async def get_table_structure(self, conn, table_name):
        columns = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = $1
        """, table_name)
        return {col['column_name']: col['data_type'] for col in columns}
    
    async def compare_and_sync_data(self, source_conn, dest_conn, table_name, source_id):
        inserts = 0
        updates = 0
        error_count = 0

        try:
            primary_keys = await self.get_primary_keys(dest_conn, table_name)
            if not primary_keys:
                warn(f"Table {table_name} has no primary keys. Using all columns for comparison.")
                columns = await self.get_table_columns(dest_conn, table_name)
                primary_keys = columns  # Use all columns if no primary key
            
            last_synced_version = await self.get_last_synced_version(table_name, source_id)

            changes = await source_conn.fetch(f"""
                SELECT * FROM "{table_name}"
                WHERE version > $1 AND server_id = $2
                ORDER BY version ASC
            """, last_synced_version, source_id)

            for change in changes:
                columns = list(change.keys())
                values = [change[col] for col in columns]

                conflict_clause = f"({', '.join(primary_keys)})"
                update_clause = ', '.join(f"{col} = EXCLUDED.{col}" for col in columns if col not in primary_keys)

                insert_query = f"""
                    INSERT INTO "{table_name}" ({', '.join(columns)})
                    VALUES ({', '.join(f'${i+1}' for i in range(len(columns)))})
                    ON CONFLICT {conflict_clause} DO UPDATE SET
                    {update_clause}
                """

                try:
                    result = await dest_conn.execute(insert_query, *values)
                    if 'UPDATE' in result:
                        updates += 1
                    else:
                        inserts += 1
                except Exception as e:
                    if error_count < 10:  # Limit error logging
                        err(f"Error syncing data for table {table_name}: {str(e)}")
                        error_count += 1
                    elif error_count == 10:
                        err(f"Suppressing further errors for table {table_name}")
                        error_count += 1

            if changes:
                await self.update_sync_status(table_name, source_id, changes[-1]['version'])

            info(f"Synced {table_name}: {inserts} inserts, {updates} updates")
            if error_count > 10:
                info(f"Total of {error_count} errors occurred for table {table_name}")

        except Exception as e:
            err(f"Error processing table {table_name}: {str(e)}")

        return inserts, updates

    async def get_table_columns(self, conn, table_name):
        columns = await conn.fetch("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """, table_name)
        return [col['column_name'] for col in columns]

    async def get_primary_keys(self, conn, table_name):
        primary_keys = await conn.fetch("""
            SELECT a.attname
            FROM   pg_index i
            JOIN   pg_attribute a ON a.attrelid = i.indrelid
                                AND a.attnum = ANY(i.indkey)
            WHERE  i.indrelid = $1::regclass
            AND    i.indisprimary
        """, table_name)
        return [pk['attname'] for pk in primary_keys]


    async def push_changes_to_all(self):
        async with self.get_connection() as local_conn:
            tables = await local_conn.fetch("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            
            for pool_entry in self.POOL:
                if pool_entry['ts_id'] == os.environ.get('TS_ID'):
                    continue
                
                try:
                    async with self.get_connection(pool_entry) as remote_conn:
                        for table in tables:
                            table_name = table['tablename']
                            last_synced_version = await self.get_last_synced_version(table_name, pool_entry['ts_id'])
                            
                            changes = await local_conn.fetch(f"""
                                SELECT * FROM "{table_name}"
                                WHERE version > $1 AND server_id = $2
                                ORDER BY version ASC
                            """, last_synced_version, os.environ.get('TS_ID'))
                            
                            for change in changes:
                                columns = change.keys()
                                values = [change[col] for col in columns]
                                await remote_conn.execute(f"""
                                    INSERT INTO "{table_name}" ({', '.join(columns)})
                                    VALUES ({', '.join(f'${i+1}' for i in range(len(columns)))})
                                    ON CONFLICT (id) DO UPDATE SET
                                    {', '.join(f"{col} = EXCLUDED.{col}" for col in columns if col != 'id')}
                                """, *values)
                            
                            if changes:
                                await self.update_sync_status(table_name, pool_entry['ts_id'], changes[-1]['version'])
                    
                    info(f"Successfully pushed changes to {pool_entry['ts_id']}")
                except Exception as e:
                    err(f"Error pushing changes to {pool_entry['ts_id']}: {str(e)}")

    async def get_last_synced_version(self, table_name, server_id):
        async with self.get_connection() as conn:
            return await conn.fetchval("""
                SELECT last_synced_version FROM sync_status
                WHERE table_name = $1 AND server_id = $2
            """, table_name, server_id) or 0


    async def update_sync_status(self, table_name, server_id, version):
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO sync_status (table_name, server_id, last_synced_version)
                VALUES ($1, $2, $3)
                ON CONFLICT (table_name, server_id) DO UPDATE
                SET last_synced_version = EXCLUDED.last_synced_version
            """, table_name, server_id, version)


    async def sync_schema(self):
        local_id = os.environ.get('TS_ID')
        source_entry = self.local_db
        source_schema = await self.get_schema(source_entry)
        
        for pool_entry in self.POOL:
            if pool_entry['ts_id'] != local_id:  # Skip the local instance
                try:
                    target_schema = await self.get_schema(pool_entry)
                    await self.apply_schema_changes(pool_entry, source_schema, target_schema)
                    info(f"Synced schema to {pool_entry['ts_ip']}")
                except Exception as e:
                    err(f"Failed to sync schema to {pool_entry['ts_ip']}: {str(e)}")

    async def get_schema(self, pool_entry: Dict[str, Any]):
        async with self.get_connection(pool_entry) as conn:
            tables = await conn.fetch("""
                SELECT table_name, column_name, data_type, character_maximum_length,
                    is_nullable, column_default, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
            
            indexes = await conn.fetch("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
            """)
            
            constraints = await conn.fetch("""
                SELECT conname, contype, conrelid::regclass::text as table_name,
                    pg_get_constraintdef(oid) as definition
                FROM pg_constraint
                WHERE connamespace = 'public'::regnamespace
            """)
            
            return {
                'tables': tables,
                'indexes': indexes,
                'constraints': constraints
            }

    async def apply_schema_changes(self, pool_entry: Dict[str, Any], source_schema, target_schema):
        async with self.get_connection(pool_entry) as conn:
            source_tables = {t['table_name']: t for t in source_schema['tables']}
            target_tables = {t['table_name']: t for t in target_schema['tables']}

            def get_column_type(data_type):
                if data_type == 'ARRAY':
                    return 'text[]'
                elif data_type == 'USER-DEFINED':
                    return 'geometry'
                else:
                    return data_type

            for table_name, source_table in source_tables.items():
                try:
                    if table_name not in target_tables:
                        columns = []
                        for t in source_schema['tables']:
                            if t['table_name'] == table_name:
                                col_type = get_column_type(t['data_type'])
                                col_def = f"\"{t['column_name']}\" {col_type}"
                                if t['character_maximum_length']:
                                    col_def += f"({t['character_maximum_length']})"
                                if t['is_nullable'] == 'NO':
                                    col_def += " NOT NULL"
                                if t['column_default']:
                                    if 'nextval' in t['column_default']:
                                        sequence_name = t['column_default'].split("'")[1]
                                        await self.create_sequence_if_not_exists(conn, sequence_name)
                                    col_def += f" DEFAULT {t['column_default']}"
                                columns.append(col_def)
                        
                        primary_key_constraint = next(
                            (con['definition'] for con in source_schema['constraints'] if con['table_name'] == table_name and con['contype'] == 'p'), 
                            None
                        )
                        
                        sql = f'CREATE TABLE "{table_name}" ({", ".join(columns)}'
                        if primary_key_constraint:
                            sql += f', {primary_key_constraint}'
                        sql += ')'
                        
                        info(f"Executing SQL: {sql}")
                        await conn.execute(sql)
                    else:
                        target_table = target_tables[table_name]
                        source_columns = {t['column_name']: t for t in source_schema['tables'] if t['table_name'] == table_name}
                        target_columns = {t['column_name']: t for t in target_schema['tables'] if t['table_name'] == table_name}

                        for col_name, source_col in source_columns.items():
                            if col_name not in target_columns:
                                col_type = get_column_type(source_col['data_type'])
                                col_def = f"\"{col_name}\" {col_type}" + \
                                        (f"({source_col['character_maximum_length']})" if source_col['character_maximum_length'] else "") + \
                                        (" NOT NULL" if source_col['is_nullable'] == 'NO' else "") + \
                                        (f" DEFAULT {source_col['column_default']}" if source_col['column_default'] else "")
                                sql = f'ALTER TABLE "{table_name}" ADD COLUMN {col_def}'
                                debug(f"Executing SQL: {sql}")
                                await conn.execute(sql)
                            else:
                                target_col = target_columns[col_name]
                                if source_col != target_col:
                                    col_type = get_column_type(source_col['data_type'])
                                    sql = f'ALTER TABLE "{table_name}" ALTER COLUMN "{col_name}" TYPE {col_type}'
                                    debug(f"Executing SQL: {sql}")
                                    await conn.execute(sql)
                                    if source_col['is_nullable'] != target_col['is_nullable']:
                                        null_constraint = "DROP NOT NULL" if source_col['is_nullable'] == 'YES' else "SET NOT NULL"
                                        sql = f'ALTER TABLE "{table_name}" ALTER COLUMN "{col_name}" {null_constraint}'
                                        debug(f"Executing SQL: {sql}")
                                        await conn.execute(sql)
                                    if source_col['column_default'] != target_col['column_default']:
                                        default_clause = f"SET DEFAULT {source_col['column_default']}" if source_col['column_default'] else "DROP DEFAULT"
                                        sql = f'ALTER TABLE "{table_name}" ALTER COLUMN "{col_name}" {default_clause}'
                                        debug(f"Executing SQL: {sql}")
                                        await conn.execute(sql)
                        
                        # Ensure primary key constraint exists
                        primary_key_constraint = next(
                            (con['definition'] for con in source_schema['constraints'] if con['table_name'] == table_name and con['contype'] == 'p'), 
                            None
                        )
                        if primary_key_constraint and primary_key_constraint not in target_schema['constraints']:
                            sql = f'ALTER TABLE "{table_name}" ADD CONSTRAINT {primary_key_constraint}'
                            debug(f"Executing SQL: {sql}")
                            await conn.execute(sql)
                except Exception as e:
                    err(f"Error processing table {table_name}: {str(e)}")

            try:
                source_indexes = {idx['indexname']: idx['indexdef'] for idx in source_schema['indexes']}
                target_indexes = {idx['indexname']: idx['indexdef'] for idx in target_schema['indexes']}

                for idx_name, idx_def in source_indexes.items():
                    if idx_name not in target_indexes:
                        debug(f"Executing SQL: {idx_def}")
                        await conn.execute(idx_def)
                    elif idx_def != target_indexes[idx_name]:
                        sql = f'DROP INDEX IF EXISTS "{idx_name}"'
                        debug(f"Executing SQL: {sql}")
                        await conn.execute(sql)
                        debug(f"Executing SQL: {idx_def}")
                        await conn.execute(idx_def)
            except Exception as e:
                err(f"Error processing indexes: {str(e)}")

            try:
                source_constraints = {con['conname']: con for con in source_schema['constraints']}
                target_constraints = {con['conname']: con for con in target_schema['constraints']}

                for con_name, source_con in source_constraints.items():
                    if con_name not in target_constraints:
                        sql = f'ALTER TABLE "{source_con["table_name"]}" ADD CONSTRAINT "{con_name}" {source_con["definition"]}'
                        debug(f"Executing SQL: {sql}")
                        await conn.execute(sql)
                    elif source_con != target_constraints[con_name]:
                        sql = f'ALTER TABLE "{source_con["table_name"]}" DROP CONSTRAINT IF EXISTS "{con_name}"'
                        debug(f"Executing SQL: {sql}")
                        await conn.execute(sql)
                        sql = f'ALTER TABLE "{source_con["table_name"]}" ADD CONSTRAINT "{con_name}" {source_con["definition"]}'
                        debug(f"Executing SQL: {sql}")
                        await conn.execute(sql)
            except Exception as e:
                err(f"Error processing constraints: {str(e)}")

        info(f"Schema synchronization completed for {pool_entry['ts_ip']}")

    async def create_sequence_if_not_exists(self, conn, sequence_name):
        await conn.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_sequences WHERE schemaname = 'public' AND sequencename = '{sequence_name}') THEN
                CREATE SEQUENCE {sequence_name};
            END IF;
        END $$;
        """)



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