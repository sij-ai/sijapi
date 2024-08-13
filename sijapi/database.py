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
        self.last_sync_time = 0

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
                l.debug(f"Server {ts_id} is online")
            except OperationalError:
                l.warning(f"Server {ts_id} is offline")
        self.online_servers = set(online_servers)
        l.info(f"Online servers: {', '.join(online_servers)}")
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
                # Execute the write query locally
                serialized_kwargs = {key: serialize(value) for key, value in kwargs.items()}
                result = await session.execute(text(query), serialized_kwargs)
                await session.commit()
                
                # Initiate async operations
                asyncio.create_task(self._async_sync_operations(query, kwargs))

                # Return the result
                return result
            
            except Exception as e:
                l.error(f"Failed to execute write query: {str(e)}")
                l.error(f"Query: {query}")
                l.error(f"Kwargs: {kwargs}")
                l.error(f"Serialized kwargs: {serialized_kwargs}")
                l.error(f"Traceback: {traceback.format_exc()}")
                return None

    async def _async_sync_operations(self, query: str, kwargs: dict):
        try:
            # Add the write query to the query_tracking table
            await self.add_query_to_tracking(query, kwargs)

            # Call /db/sync on all online servers
            await self.call_db_sync_on_servers()
        except Exception as e:
            l.error(f"Error in async sync operations: {str(e)}")
            l.error(f"Traceback: {traceback.format_exc()}")

    async def add_query_to_tracking(self, query: str, kwargs: dict):
        async with self.sessions[self.local_ts_id]() as session:
            new_query = QueryTracking(
                ts_id=self.local_ts_id,
                query=query,
                args=json_dumps(kwargs),
                completed_by={self.local_ts_id: True}
            )
            session.add(new_query)
            await session.commit()
        l.info(f"Added query to tracking: {query[:50]}...")

    async def sync_db(self):
        current_time = time.time()
        if current_time - self.last_sync_time < 30:
            l.info("Skipping sync, last sync was less than 30 seconds ago")
            return

        try:
            l.info("Starting database synchronization")
            await self.pull_query_tracking_from_all_servers()
            await self.execute_unexecuted_queries()
            self.last_sync_time = current_time
            l.info("Database synchronization completed successfully")
        except Exception as e:
            l.error(f"Error during database sync: {str(e)}")
            l.error(f"Traceback: {traceback.format_exc()}")

    async def pull_query_tracking_from_all_servers(self):
        online_servers = await self.get_online_servers()
        l.info(f"Pulling query tracking from {len(online_servers)} online servers")
        
        for server_id in online_servers:
            if server_id == self.local_ts_id:
                continue  # Skip local server
            
            l.info(f"Pulling queries from server: {server_id}")
            async with self.sessions[server_id]() as remote_session:
                queries = await remote_session.execute(select(QueryTracking))
                queries = queries.fetchall()

            l.info(f"Retrieved {len(queries)} queries from server {server_id}")
            async with self.sessions[self.local_ts_id]() as local_session:
                for query in queries:
                    existing = await local_session.execute(
                        select(QueryTracking).where(QueryTracking.id == query.id)
                    )
                    existing = existing.scalar_one_or_none()
                    
                    if existing:
                        existing.completed_by = {**existing.completed_by, **query.completed_by}
                        l.debug(f"Updated existing query: {query.id}")
                    else:
                        local_session.add(query)
                        l.debug(f"Added new query: {query.id}")
                await local_session.commit()
        l.info("Finished pulling queries from all servers")

    async def execute_unexecuted_queries(self):
        async with self.sessions[self.local_ts_id]() as session:
            unexecuted_queries = await session.execute(
                select(QueryTracking).where(~QueryTracking.completed_by.has_key(self.local_ts_id)).order_by(QueryTracking.executed_at)
            )
            unexecuted_queries = unexecuted_queries.fetchall()

            l.info(f"Executing {len(unexecuted_queries)} unexecuted queries")
            for query in unexecuted_queries:
                try:
                    params = json.loads(query.args)
                    await session.execute(text(query.query), params)
                    query.completed_by[self.local_ts_id] = True
                    await session.commit()
                    l.info(f"Successfully executed query ID {query.id}")
                except Exception as e:
                    l.error(f"Failed to execute query ID {query.id}: {str(e)}")
                    await session.rollback()
        l.info("Finished executing unexecuted queries")

    async def call_db_sync_on_servers(self):
        """Call /db/sync on all online servers."""
        online_servers = await self.get_online_servers()
        l.info(f"Calling /db/sync on {len(online_servers)} online servers")
        for server in self.config['POOL']:
            if server['ts_id'] in online_servers and server['ts_id'] != self.local_ts_id:
                try:
                    await self.call_db_sync(server)
                except Exception as e:
                    l.error(f"Failed to call /db/sync on {server['ts_id']}: {str(e)}")
        l.info("Finished calling /db/sync on all servers")

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

    async def ensure_query_tracking_table(self):
        for ts_id, engine in self.engines.items():
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                l.info(f"Ensured query_tracking table exists for {ts_id}")
            except Exception as e:
                l.error(f"Failed to create query_tracking table for {ts_id}: {str(e)}")
    
    async def close(self):
        for engine in self.engines.values():
            await engine.dispose()
        l.info("Closed all database connections")
