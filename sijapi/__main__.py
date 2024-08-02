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
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    crit("sijapi launched")
    info(f"Arguments: {args}")

    # Load routers
    if args.test:
        load_router(args.test)
    else:
        for module_name in API.MODULES.__fields__:
            if getattr(API.MODULES, module_name):
                load_router(module_name)

    try:
        # Initialize sync structures on all databases
        await API.initialize_sync()
        
    except Exception as e:
        crit(f"Error during startup: {str(e)}")
        crit(f"Traceback: {traceback.format_exc()}")

    try:
        yield  # This is where the app runs
        
    finally:
        # Shutdown
        crit("Shutting down...")
        try:
            await asyncio.wait_for(API.close_db_pools(), timeout=20)
            crit("Database pools closed.")
        except asyncio.TimeoutError:
            crit("Timeout while closing database pools.")
        except Exception as e:
            crit(f"Error during shutdown: {str(e)}")
            crit(f"Traceback: {traceback.format_exc()}")

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


@app.post("/sync/pull")
async def pull_changes():
    try:
        await API.add_primary_keys_to_local_tables()
        await API.add_primary_keys_to_remote_tables()
        try:
            
            source = await API.get_most_recent_source()
            
            if source:
                # Pull changes from the source
                total_changes = await API.pull_changes(source)
                
                return JSONResponse(content={
                    "status": "success",
                    "message": f"Pull complete. Total changes: {total_changes}",
                    "source": f"{source['ts_id']} ({source['ts_ip']})",
                    "changes": total_changes
                })
            else:
                return JSONResponse(content={
                    "status": "info",
                    "message": "No instances with more recent data found or all instances are offline."
                })
        
        except Exception as e:
            err(f"Error during pull: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error during pull: {str(e)}")
        
    except Exception as e:
            err(f"Error while ensuring primary keys to tables: {str(e)}")
            err(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error during primary key insurance: {str(e)}")


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
