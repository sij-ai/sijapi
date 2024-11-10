# database.py
import yaml
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
from loguru import logger
from sqlalchemy import text, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text
import uuid
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import os
from .logs import get_logger
from .serialization import serialize

l = get_logger(__name__)

Base = declarative_base()

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(ENV_PATH)
TS_ID = os.environ.get('TS_ID')

class Database:
    @classmethod
    def init(cls, config_name: str):
        return cls(config_name)

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.engines: Dict[str, Any] = {}
        self.sessions: Dict[str, Any] = {}
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
                
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                l.info(f"Ensured tables exist for {db_info['ts_id']}")
            except Exception as e:
                l.error(f"Failed to initialize engine for {db_info['ts_id']}: {str(e)}")

        if self.local_ts_id not in self.sessions:
            l.error(f"Failed to initialize session for local server {self.local_ts_id}")

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
                serialized_kwargs = {key: serialize(value) for key, value in kwargs.items()}
                result = await session.execute(text(query), serialized_kwargs)
                await session.commit()
                return result
            except Exception as e:
                l.error(f"Failed to execute write query: {str(e)}")
                l.error(f"Query: {query}")
                l.error(f"Kwargs: {kwargs}")
                l.error(f"Serialized kwargs: {serialized_kwargs}")
                return None

    async def close(self):
        for engine in self.engines.values():
            await engine.dispose()
        l.info("Closed all database connections")
