#!/Users/sij/miniforge3/envs/api/bin/python
#__main__.py
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from hypercorn.asyncio import serve
from hypercorn.config import Config as HypercornConfig
import sys
import asyncio 
import argparse
import ipaddress
import importlib
import argparse

parser = argparse.ArgumentParser(description='Personal API.')
parser.add_argument('--debug', action='store_true', help='Set log level to L.INFO')
parser.add_argument('--test', type=str, help='Load only the specified module.')
args = parser.parse_args()

from . import L, API, Dir
L.setup_from_args(args)
app = FastAPI()

# CORSMiddleware
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
                    L.ERR(f"Invalid API key provided by a requester.")
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"}
                    )
        response = await call_next(request)
        return response

app.add_middleware(SimpleAPIKeyMiddleware)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    L.ERR(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    L.ERR(f"Request: {request.method} {request.url}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.middleware("http")
async def handle_exception_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
    except RuntimeError as exc:
        if str(exc) == "Response content longer than Content-Length":
            response.headers["Content-Length"] = str(len(response.body))
        else:
            raise
    return response


def load_router(router_name):
    router_file = Dir.ROUTERS / f'{router_name}.py'
    L.DEBUG(f"Attempting to load {router_name.capitalize()}...")
    if router_file.exists():
        module_path = f'sijapi.routers.{router_name}'
        try:
            module = importlib.import_module(module_path)
            router = getattr(module, router_name)
            app.include_router(router)
            L.INFO(f"{router_name.capitalize()} router loaded.")
        except (ImportError, AttributeError) as e:
            L.CRIT(f"Failed to load router {router_name}: {e}")
    else:
        L.ERR(f"Router file for {router_name} does not exist.")

def main(argv):
    if args.test:
        load_router(args.test)
    else:
        L.CRIT(f"sijapi launched")
        L.CRIT(f"{args._get_args}")
        for module_name in API.MODULES.__fields__:
            if getattr(API.MODULES, module_name):
                load_router(module_name)
    
    config = HypercornConfig()
    config.bind = [API.BIND]  # Use the resolved BIND value
    asyncio.run(serve(app, config))

if __name__ == "__main__":
    main(sys.argv[1:])