#!/Users/sij/miniforge3/envs/api/bin/python
#__main__.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from hypercorn.asyncio import serve
from hypercorn.config import Config as HypercornConfig
import sys
import os
import traceback
import asyncio
import ipaddress
import importlib
from pathlib import Path
import argparse
from . import Sys, Db, Dir
from .logs import L, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Personal API.')
    parser.add_argument('--log', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set overall log level (e.g., DEBUG, INFO, WARNING)')
    parser.add_argument('--debug', nargs='+', default=[], 
                        help='Set DEBUG log level for specific modules')
    parser.add_argument('--info', nargs='+', default=[],
                        help='Set INFO log level for specific modules')
    parser.add_argument('--test', type=str, help='Load only the specified module.')
    return parser.parse_args()

args = parse_args()

# Setup logging
L.setup_from_args(args)
l = get_logger("main")
l.info(f"Logging initialized. Debug modules: {L.debug_modules}")
l.info(f"Command line arguments: {args}")

l.debug(f"Current working directory: {os.getcwd()}")
l.debug(f"__file__ path: {__file__}")
l.debug(f"Absolute path of __file__: {os.path.abspath(__file__)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    l.critical("sijapi launched")
    l.info(f"Arguments: {args}")
    
    # Log the router directory path
    l.debug(f"Router directory path: {Dir.ROUTER.absolute()}")
    l.debug(f"Router directory exists: {Dir.ROUTER.exists()}")
    l.debug(f"Router directory is a directory: {Dir.ROUTER.is_dir()}")
    l.debug(f"Contents of router directory: {list(Dir.ROUTER.iterdir())}")

    # Load routers
    if args.test:
        load_router(args.test)
    else:
        for module_name in Sys.MODULES.__fields__:
            if getattr(Sys.MODULES, module_name):
                load_router(module_name)

    try:
        await Db.initialize_engines()
    except Exception as e:
        l.critical(f"Error during startup: {str(e)}")
        l.critical(f"Traceback: {traceback.format_exc()}")

    try:
        yield  # This is where the app runs
    finally:
        # Shutdown
        l.critical("Shutting down...")
        try:
            await asyncio.wait_for(Db.close(), timeout=20)
            l.critical("Database pools closed.")
        except asyncio.TimeoutError:
            l.critical("Timeout while closing database pools.")
        except Exception as e:
            l.critical(f"Error during shutdown: {str(e)}")
            l.critical(f"Traceback: {traceback.format_exc()}")

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
        if request.url.path not in Sys.PUBLIC:
            trusted_subnets = [ipaddress.ip_network(subnet) for subnet in Sys.TRUSTED_SUBNETS]
            if not any(client_ip in subnet for subnet in trusted_subnets):
                api_key_header = request.headers.get("Authorization")
                api_key_query = request.query_params.get("api_key")
                
                # Convert Sys.KEYS to lowercase for case-insensitive comparison
                api_keys_lower = [key.lower() for key in Sys.KEYS]
                l.debug(f"Sys.KEYS (lowercase): {api_keys_lower}")
                
                if api_key_header:
                    api_key_header = api_key_header.lower().split("bearer ")[-1]
                    l.debug(f"API key provided in header: {api_key_header}")
                if api_key_query:
                    api_key_query = api_key_query.lower()
                    l.debug(f"API key provided in query: {api_key_query}")
                
                if (api_key_header is None or api_key_header.lower() not in api_keys_lower) and \
                (api_key_query is None or api_key_query.lower() not in api_keys_lower):
                    l.error(f"Invalid API key provided by a requester.")
                    if api_key_header:
                        l.debug(f"Invalid API key in header: {api_key_header}")
                    if api_key_query:
                        l.debug(f"Invalid API key in query: {api_key_query}")
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"}
                    )
                else:
                    if api_key_header and api_key_header.lower() in api_keys_lower:
                        l.debug(f"Valid API key provided in header: {api_key_header}")
                    if api_key_query and api_key_query.lower() in api_keys_lower:
                        l.debug(f"Valid API key provided in query: {api_key_query}")
        
        response = await call_next(request)
        return response

# Add the middleware to your FastAPI app
app.add_middleware(SimpleAPIKeyMiddleware)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    l.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    l.error(f"Request: {request.method} {request.url}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.middleware("http")
async def handle_exception_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        l.error(f"Unhandled exception in request: {request.method} {request.url}")
        l.error(f"Exception: {str(exc)}")
        l.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"}
        )

@app.post("/sync/pull")
async def pull_changes():
    l.info(f"Received request to /sync/pull")
    try:
        await Sys.add_primary_keys_to_local_tables()
        await Sys.add_primary_keys_to_remote_tables()
        try:
            source = await Sys.get_most_recent_source()
            if source:
                # Pull changes from the source
                total_changes = await Sys.pull_changes(source)

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
            l.error(f"Error in /sync/pull: {str(e)}")
            l.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error during pull: {str(e)}")
        finally:
            l.info(f"Finished processing /sync/pull request")
    except Exception as e:
        l.error(f"Error while ensuring primary keys to tables: {str(e)}")
        l.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error during primary key insurance: {str(e)}")

def load_router(router_name):
    router_logger = get_logger(f"router.{router_name}")
    router_logger.debug(f"Attempting to load {router_name.capitalize()}...")
    
    # Log the full path being checked
    router_file = Dir.ROUTER / f'{router_name}.py'
    router_logger.debug(f"Checking for router file at: {router_file.absolute()}")
    
    if router_file.exists():
        router_logger.debug(f"Router file found: {router_file}")
        module_path = f'sijapi.routers.{router_name}'
        router_logger.debug(f"Attempting to import module: {module_path}")
        try:
            module = importlib.import_module(module_path)
            router_logger.debug(f"Module imported successfully: {module}")
            router = getattr(module, router_name)
            router_logger.debug(f"Router object retrieved: {router}")
            app.include_router(router)
            router_logger.info(f"Router {router_name} loaded successfully")
        except (ImportError, AttributeError) as e:
            router_logger.critical(f"Failed to load router {router_name}: {e}")
            router_logger.debug(f"Current working directory: {os.getcwd()}")
            router_logger.debug(f"Python path: {sys.path}")
    else:
        router_logger.error(f"Router file for {router_name} does not exist at {router_file.absolute()}")
        router_logger.debug(f"Contents of router directory: {list(Dir.ROUTER.iterdir())}")


def main(argv):
    config = HypercornConfig()
    config.bind = [Sys.BIND]
    config.startup_timeout = 300 # 5 minutes
    config.shutdown_timeout = 15 # 15 seconds
    asyncio.run(serve(app, config))

if __name__ == "__main__":
    main(sys.argv[1:])
