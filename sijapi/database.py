# database.py

import json
import yaml
import time
import aiohttp
import asyncio
import traceback
from datetime import datetime as dt_datetime, date
from tqdm.asyncio import tqdm
import reverse_geocoder as rg
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, ClassVar
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model, PrivateAttr
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from srtm import get_data
import os
import sys
from loguru import logger
from sqlalchemy import text, select, func, and_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text
from sqlalchemy.dialects.postgresql import JSONB
from urllib.parse import urljoin
import hashlib
import random
from .logs import get_logger
from .serialization import json_dumps, json_serial, serialize

l = get_logger(__name__)

Base = declarative_base()

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(ENV_PATH)
TS_ID = os.environ.get('TS_ID')

class QueryTracking(Base):
    __tablename__ = 'query_tracking'

    id = Column(Integer, primary_key=True)
    ts_id = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    args = Column(JSONB)
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_by = Column(JSONB, default={})
    result_checksum = Column(String)

class Database:
    @classmethod
    def init(cls, config_name: str):
        return cls(config_name)

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.engines: Dict[str, Any] = {}
        self.sessions: Dict[str, Any] = {}
        self.online_servers: set = set()
        self.local_ts_id = self.get_local_ts_id()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        base_path = Path(__file__).parent.parent
        full_path = base_path / "sijapi" / "config" / f"{config_path}.yaml"
        
        with open(full_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config

    def get_local_ts_id(self) -> str:
        return os.environ.get('TS_ID')

    async def initialize_engines(self):
        for db_info in self.config['POOL']:
            url = f"postgresql+asyncpg://{db_info['db_user']}:{db_info['db_pass']}@{db_info['ts_ip']}:{db_info['db_port']}/{db_info['db_name']}"
            try:
                engine = create_async_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
                self.engines[db_info['ts_id']] = engine
                self.sessions[db_info['ts_id']] = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
                l.info(f"Initialized engine and session for {db_info['ts_id']}")
                
                # Create tables if they don't exist
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                l.info(f"Ensured tables exist for {db_info['ts_id']}")
            except Exception as e:
                l.error(f"Failed to initialize engine for {db_info['ts_id']}: {str(e)}")

        if self.local_ts_id not in self.sessions:
            l.error(f"Failed to initialize session for local server {self.local_ts_id}")

    async def get_online_servers(self) -> List[str]:
        online_servers = []
        for ts_id, engine in self.engines.items():
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                online_servers.append(ts_id)
            except OperationalError:
                pass
        self.online_servers = set(online_servers)
        return online_servers

    async def read(self, query: str, **kwargs):
        if self.local_ts_id not in self.sessions:
            l.error(f"No session found for local server {self.local_ts_id}. Database may not be properly initialized.")
            return None

        async with self.sessions[self.local_ts_id]() as session:
            try:
                result = await session.execute(text(query), kwargs)
                rows = result.fetchall()
                if rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    return []
            except Exception as e:
                l.error(f"Failed to execute read query: {str(e)}")
                return None


    async def write(self, query: str, **kwargs):
        if self.local_ts_id not in self.sessions:
            l.error(f"No session found for local server {self.local_ts_id}. Database may not be properly initialized.")
            return None

        async with self.sessions[self.local_ts_id]() as session:
            try:
                # Serialize the kwargs
                serialized_kwargs = {key: serialize(value) for key, value in kwargs.items()}

                # Execute the write query
                result = await session.execute(text(query), serialized_kwargs)
                
                # Log the query
                new_query = QueryTracking(
                    ts_id=self.local_ts_id,
                    query=query,
                    args=json_dumps(kwargs)  # Use json_dumps for logging
                )
                session.add(new_query)
                await session.flush()
                query_id = new_query.id

                await session.commit()
                l.info(f"Successfully executed write query: {query[:50]}...")

                checksum = await self._local_compute_checksum(query, serialized_kwargs)

                # Update query_tracking with checksum
                await self.update_query_checksum(query_id, checksum)

                # Perform sync operations asynchronously
                asyncio.create_task(self._async_sync_operations(query_id, query, serialized_kwargs, checksum))

                return result
            
            except Exception as e:
                l.error(f"Failed to execute write query: {str(e)}")
                l.error(f"Query: {query}")
                l.error(f"Kwargs: {kwargs}")
                l.error(f"Serialized kwargs: {serialized_kwargs}")
                l.error(f"Traceback: {traceback.format_exc()}")
                return None

    async def _async_sync_operations(self, query_id: int, query: str, params: dict, checksum: str):
        try:
            await self.sync_query_tracking()
        except Exception as e:
            l.error(f"Failed to sync query_tracking: {str(e)}")

        try:
            await self.call_db_sync_on_servers()
        except Exception as e:
            l.error(f"Failed to call db_sync on other servers: {str(e)}")

        # Replicate write to other servers
        online_servers = await self.get_online_servers()
        for ts_id in online_servers:
            if ts_id != self.local_ts_id:
                try:
                    await self._replicate_write(ts_id, query_id, query, params, checksum)
                except Exception as e:
                    l.error(f"Failed to replicate write to {ts_id}: {str(e)}")


    async def get_primary_server(self) -> str:
        url = urljoin(self.config['URL'], '/id')
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        primary_ts_id = await response.text()
                        return primary_ts_id.strip()
                    else:
                        l.error(f"Failed to get primary server. Status: {response.status}")
                        return None
            except aiohttp.ClientError as e:
                l.error(f"Error connecting to load balancer: {str(e)}")
                return None

    async def get_checksum_server(self) -> dict:
        primary_ts_id = await self.get_primary_server()
        online_servers = await self.get_online_servers()
        
        checksum_servers = [server for server in self.config['POOL'] if server['ts_id'] in online_servers and server['ts_id'] != primary_ts_id]
        
        if not checksum_servers:
            return next(server for server in self.config['POOL'] if server['ts_id'] == primary_ts_id)
        
        return random.choice(checksum_servers)

    async def _local_compute_checksum(self, query: str, params: dict):
        async with self.sessions[self.local_ts_id]() as session:
            result = await session.execute(text(query), params)
            if result.returns_rows:
                data = result.fetchall()
            else:
                data = str(result.rowcount) + query + str(params)
            checksum = hashlib.md5(str(data).encode()).hexdigest()
            return checksum

    async def _delegate_compute_checksum(self, server: Dict[str, Any], query: str, params: dict):
        url = f"http://{server['ts_ip']}:{server['app_port']}/sync/checksum"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json={"query": query, "params": params}) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['checksum']
                    else:
                        l.error(f"Failed to get checksum from {server['ts_id']}. Status: {response.status}")
                        return await self._local_compute_checksum(query, params)
            except aiohttp.ClientError as e:
                l.error(f"Error connecting to {server['ts_id']} for checksum: {str(e)}")
                return await self._local_compute_checksum(query, params)

    async def update_query_checksum(self, query_id: int, checksum: str):
        async with self.sessions[self.local_ts_id]() as session:
            await session.execute(
                text("UPDATE query_tracking SET result_checksum = :checksum WHERE id = :id"),
                {"checksum": checksum, "id": query_id}
            )
            await session.commit()

    async def _replicate_write(self, ts_id: str, query_id: int, query: str, params: dict, expected_checksum: str):
        try:
            async with self.sessions[ts_id]() as session:
                await session.execute(text(query), params)
                actual_checksum = await self._local_compute_checksum(query, params)
                if actual_checksum != expected_checksum:
                    raise ValueError(f"Checksum mismatch on {ts_id}")
                await self.mark_query_completed(query_id, ts_id)
                await session.commit()
                l.info(f"Successfully replicated write to {ts_id}")
        except Exception as e:
            l.error(f"Failed to replicate write on {ts_id}")
            l.debug(f"Failed to replicate write on {ts_id}: {str(e)}")

    async def mark_query_completed(self, query_id: int, ts_id: str):
        async with self.sessions[self.local_ts_id]() as session:
            query = await session.get(QueryTracking, query_id)
            if query:
                completed_by = query.completed_by or {}
                completed_by[ts_id] = True
                query.completed_by = completed_by
                await session.commit()

    async def sync_local_server(self):
        async with self.sessions[self.local_ts_id]() as session:
            last_synced = await session.execute(
                text("SELECT MAX(id) FROM query_tracking WHERE completed_by ? :ts_id"),
                {"ts_id": self.local_ts_id}
            )
            last_synced_id = last_synced.scalar() or 0

            unexecuted_queries = await session.execute(
                text("SELECT * FROM query_tracking WHERE id > :last_id ORDER BY id"),
                {"last_id": last_synced_id}
            )

            for query in unexecuted_queries:
                try:
                    params = json.loads(query.args)
                    await session.execute(text(query.query), params)
                    actual_checksum = await self._local_compute_checksum(query.query, params)
                    if actual_checksum != query.result_checksum:
                        raise ValueError(f"Checksum mismatch for query ID {query.id}")
                    await self.mark_query_completed(query.id, self.local_ts_id)
                except Exception as e:
                    l.error(f"Failed to execute query ID {query.id} during local sync: {str(e)}")

            await session.commit()
            l.info(f"Local server sync completed. Executed {unexecuted_queries.rowcount} queries.")

    async def purge_completed_queries(self):
        async with self.sessions[self.local_ts_id]() as session:
            all_ts_ids = [db['ts_id'] for db in self.config['POOL']]
            
            result = await session.execute(
                text("""
                    DELETE FROM query_tracking
                    WHERE id <= (
                        SELECT MAX(id)
                        FROM query_tracking
                        WHERE completed_by ?& :ts_ids
                    )
                """),
                {"ts_ids": all_ts_ids}
            )
            await session.commit()
            
            deleted_count = result.rowcount
            l.info(f"Purged {deleted_count} completed queries.")

    async def sync_query_tracking(self):
        """Combinatorial sync method for the query_tracking table."""
        try:
            online_servers = await self.get_online_servers()
            
            for ts_id in online_servers:
                if ts_id == self.local_ts_id:
                    continue
                
                try:
                    async with self.sessions[ts_id]() as remote_session:
                        local_max_id = await self.get_max_query_id(self.local_ts_id)
                        remote_max_id = await self.get_max_query_id(ts_id)
                        
                        # Sync from remote to local
                        remote_new_queries = await remote_session.execute(
                            select(QueryTracking).where(QueryTracking.id > local_max_id)
                        )
                        for query in remote_new_queries:
                            await self.add_or_update_query(query)
                        
                        # Sync from local to remote
                        async with self.sessions[self.local_ts_id]() as local_session:
                            local_new_queries = await local_session.execute(
                                select(QueryTracking).where(QueryTracking.id > remote_max_id)
                            )
                            for query in local_new_queries:
                                await self.add_or_update_query_remote(ts_id, query)
                except Exception as e:
                    l.error(f"Error syncing with {ts_id}: {str(e)}")
        except Exception as e:
            l.error(f"Error in sync_query_tracking: {str(e)}")
            l.error(f"Traceback: {traceback.format_exc()}")


    async def get_max_query_id(self, ts_id):
        async with self.sessions[ts_id]() as session:
            result = await session.execute(select(func.max(QueryTracking.id)))
            return result.scalar() or 0

    async def add_or_update_query(self, query):
        async with self.sessions[self.local_ts_id]() as session:
            existing_query = await session.get(QueryTracking, query.id)
            if existing_query:
                existing_query.completed_by = {**existing_query.completed_by, **query.completed_by}
            else:
                session.add(query)
            await session.commit()

    async def add_or_update_query_remote(self, ts_id, query):
        async with self.sessions[ts_id]() as session:
            existing_query = await session.get(QueryTracking, query.id)
            if existing_query:
                existing_query.completed_by = {**existing_query.completed_by, **query.completed_by}
            else:
                new_query = QueryTracking(
                    id=query.id,
                    ts_id=query.ts_id,
                    query=query.query,
                    args=query.args,
                    executed_at=query.executed_at,
                    completed_by=query.completed_by,
                    result_checksum=query.result_checksum
                )
                session.add(new_query)
            await session.commit()

    async def ensure_query_tracking_table(self):
        for ts_id, engine in self.engines.items():
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                l.info(f"Ensured query_tracking table exists for {ts_id}")
            except Exception as e:
                l.error(f"Failed to create query_tracking table for {ts_id}: {str(e)}")


    async def call_db_sync_on_servers(self):
        """Call /db/sync on all online servers."""
        online_servers = await self.get_online_servers()
        tasks = []
        for server in self.config['POOL']:
            if server['ts_id'] in online_servers and server['ts_id'] != self.local_ts_id:
                url = f"http://{server['ts_ip']}:{server['app_port']}/db/sync"
                tasks.append(self.call_db_sync(server))
        await asyncio.gather(*tasks)


    async def call_db_sync(self, server):
        url = f"http://{server['ts_ip']}:{server['app_port']}/db/sync"
        headers = {
            "Authorization": f"Bearer {server['api_key']}"
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        l.info(f"Successfully called /db/sync on {url}")
                    else:
                        l.warning(f"Failed to call /db/sync on {url}. Status: {response.status}")
            except asyncio.TimeoutError:
                l.debug(f"Timeout while calling /db/sync on {url}")
            except Exception as e:
                l.error(f"Error calling /db/sync on {url}: {str(e)}")


    async def close(self):
        for engine in self.engines.values():
            await engine.dispose()
