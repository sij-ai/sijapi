# database.py
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from .logs import get_logger
from .serialization import serialize

l = get_logger(__name__)

Base = declarative_base()

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(ENV_PATH)


class Database:
    @classmethod
    def init(cls, config_name: str):
        return cls(config_name)

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.engine: Optional[Any] = None
        self.async_session: Optional[Any] = None

    def load_config(self, config_path: str) -> Dict[str, Any]:
        base_path = Path(__file__).parent.parent
        full_path = base_path / "sijapi" / "config" / f"{config_path}.yaml"
        
        with open(full_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config

    async def initialize(self):
        """Initialize the database connection."""
        db = self.config.get('DATABASE', {})
        
        host = db.get('host', 'localhost')
        port = db.get('port', 5432)
        name = db.get('name', 'sijapi')
        user = db.get('user', 'sijapi')
        password = db.get('password', '')
        
        url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
        
        try:
            self.engine = create_async_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
            self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
            l.info(f"Initialized database connection to {host}:{port}/{name}")
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            l.info("Ensured database tables exist")
        except Exception as e:
            l.error(f"Failed to initialize database: {str(e)}")
            raise

    async def read(self, query: str, **kwargs):
        """Execute a read query and return results."""
        if not self.async_session:
            l.error("Database not initialized. Call initialize() first.")
            return None

        async with self.async_session() as session:
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
        """Execute a write query."""
        if not self.async_session:
            l.error("Database not initialized. Call initialize() first.")
            return None

        async with self.async_session() as session:
            try:
                serialized_kwargs = {key: serialize(value) for key, value in kwargs.items()}
                result = await session.execute(text(query), serialized_kwargs)
                await session.commit()
                return result
            except Exception as e:
                l.error(f"Failed to execute write query: {str(e)}")
                l.error(f"Query: {query}")
                l.error(f"Kwargs: {kwargs}")
                return None

    async def close(self):
        """Close the database connection."""
        if self.engine:
            await self.engine.dispose()
            l.info("Closed database connection")
