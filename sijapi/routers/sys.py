# routers/sys.py

import socket
import httpx
from fastapi import APIRouter
from sijapi import Sys
from sijapi.logs import get_logger

l = get_logger(__name__)

sys = APIRouter()

@sys.get("/health")
def get_health():
    return {"status": "ok"}

@sys.get("/id")
def get_id() -> str:
    """Get the server's hostname."""
    return socket.gethostname()

@sys.get("/routers")
def get_routers() -> str:
    active_modules = [module for module, is_active in Sys.MODULES.__dict__.items() if is_active]
    return active_modules

@sys.get("/ip")
def get_local_ip():
    """Get the server's local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

@sys.get("/wan_ip")
async def get_wan_ip():
    """Get the WAN IP address using Mullvad's API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get('https://am.i.mullvad.net/json')
            response.raise_for_status()
            wan_info = response.json()
            return wan_info.get('ip', 'Unavailable')
        except Exception as e:
            l.error(f"Error fetching WAN IP: {e}")
            return "Unavailable"
