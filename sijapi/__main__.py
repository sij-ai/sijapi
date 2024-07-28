#!/Users/sij/miniforge3/envs/api/bin/python
#__main__.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import ClientDisconnect
from hypercorn.asyncio import serve
from hypercorn.config import Config as HypercornConfig
import sys
import os
import traceback
import asyncio 
import httpx
import argparse
import json
import ipaddress
import importlib
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import argparse
from . import L, API, ROUTER_DIR

parser = argparse.ArgumentParser(description='Personal API.')
parser.add_argument('--log', type=str, default='INFO', help='Set overall log level (e.g., DEBUG, INFO, WARNING)')
parser.add_argument('--debug', nargs='+', default=[], help='Set DEBUG log level for specific modules')
parser.add_argument('--test', type=str, help='Load only the specified module.')
args = parser.parse_args()

L.setup_from_args(args)
print(f"Debug modules after setup: {L.debug_modules}")
logger = L.get_module_logger("main")
def debug(text: str): logger.debug(text)
debug(f"Debug message.")
def info(text: str): logger.info(text)
info(f"Info message.")
def warn(text: str): logger.warning(text)
warn(f"Warning message.")
def err(text: str): logger.error(text)
err(f"Error message.")
def crit(text: str): logger.critical(text)
crit(f"Critical message.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    crit("sijapi launched")
    crit(f"Arguments: {args}")

    # Load routers
    if args.test:
        load_router(args.test)
    else:
        for module_name in API.MODULES.__fields__:
            if getattr(API.MODULES, module_name):
                load_router(module_name)

    crit("Starting database synchronization...")
    try:
        # Initialize sync structures
        await API.initialize_sync()

        # Sync schema across all databases
        await API.sync_schema()
        crit("Schema synchronization complete.")
        
        # Check if other instances have more recent data
        source = await API.get_most_recent_source()
        if source:
            crit(f"Pulling changes from {source['ts_id']} ({source['ts_ip']})...")
            total_changes = await API.pull_changes(source)
            crit(f"Data pull complete. Total changes: {total_changes}")
        else:
            crit("No instances with more recent data found.")


    except Exception as e:
        crit(f"Error during startup: {str(e)}")
        crit(f"Traceback: {traceback.format_exc()}")

    yield  # This is where the app runs

    # Shutdown
    crit("Shutting down...")
    # Perform any cleanup operations here if needed

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class SimpleAPIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = ipaddress.ip_address(request.client.host)
        if request.method == "OPTIONS":
            # Allow CORS preflight requests
            return JSONResponse(status_code=200)
        if request.url.path not in API.PUBLIC:
            trusted_subnets = [ipaddress.ip_network(subnet) for subnet in API.TRUSTED_SUBNETS]
            if not any(client_ip in subnet for subnet in trusted_subnets):
                api_key_header = request.headers.get("Authorization")
                api_key_query = request.query_params.get("api_key")
                if api_key_header:
                    api_key_header = api_key_header.lower().split("bearer ")[-1]
                if api_key_header not in API.KEYS and api_key_query not in API.KEYS:
                    err(f"Invalid API key provided by a requester.")
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"}
                    )
        response = await call_next(request)
        return response

# Add the middleware to your FastAPI app
app.add_middleware(SimpleAPIKeyMiddleware)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    err(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    err(f"Request: {request.method} {request.url}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.middleware("http")
async def handle_exception_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
    except RuntimeError as exc:
        if str(exc) == "Response content longer than Content-Length":
            # Update the Content-Length header to match the actual response content length
            response.headers["Content-Length"] = str(len(response.body))
        else:
            raise
    return response

@app.middleware("http")
async def sync_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # Check if the request was a database write operation
    if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
        try:
            # Push changes to other databases
            await API.push_changes_to_all()
        except Exception as e:
            err(f"Error pushing changes to other databases: {str(e)}")
    
    return response

def load_router(router_name):
    router_file = ROUTER_DIR / f'{router_name}.py'
    module_logger = L.get_module_logger(router_name)
    module_logger.debug(f"Attempting to load {router_name.capitalize()}...")
    if router_file.exists():
        module_path = f'sijapi.routers.{router_name}'
        try:
            module = importlib.import_module(module_path)
            router = getattr(module, router_name)
            app.include_router(router)
        except (ImportError, AttributeError) as e:
            module_logger.critical(f"Failed to load router {router_name}: {e}")
    else:
        module_logger.error(f"Router file for {router_name} does not exist.")

def main(argv):
    config = HypercornConfig()
    config.bind = [API.BIND]
    config.startup_timeout = 3600  # 1 hour
    asyncio.run(serve(app, config))

if __name__ == "__main__":
    main(sys.argv[1:])
