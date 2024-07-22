'''
Health check module. /health returns `'status': 'ok'`, /id returns TS_ID, /routers responds with a list of the active routers, /ip responds with the device's local IP, /ts_ip responds with its tailnet IP, and /wan_ip responds with WAN IP.
Depends on:
  TS_ID, LOGGER, SUBNET_BROADCAST
'''
import os
import httpx
import socket
from fastapi import APIRouter
from tailscale import Tailscale
from sijapi import L, API, TS_ID, SUBNET_BROADCAST

health = APIRouter(tags=["public", "trusted", "private"])
logger = L.get_module_logger("health")
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

@health.get("/health")
def get_health():
    return {"status": "ok"}

@health.get("/id")
def get_health() -> str:
    return TS_ID

@health.get("/routers")
def get_routers() -> str:
    active_modules = [module for module, is_active in API.MODULES.__dict__.items() if is_active]
    return active_modules

@health.get("/ip")
def get_local_ip():
    """Get the server's local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((f'{SUBNET_BROADCAST}', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

@health.get("/wan_ip")
async def get_wan_ip():
    """Get the WAN IP address using Mullvad's API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get('https://am.i.mullvad.net/json')
            response.raise_for_status()
            wan_info = response.json()
            return wan_info.get('ip', 'Unavailable')
        except Exception as e:
            err(f"Error fetching WAN IP: {e}")
            return "Unavailable"

@health.get("/ts_ip")
async def get_tailscale_ip():
    """Get the Tailscale IP address."""
    tailnet = os.getenv("TAILNET")
    api_key = os.getenv("TAILSCALE_API_KEY")
    async with Tailscale(tailnet=tailnet, api_key=api_key) as tailscale:
        devices = await tailscale.devices()
        if devices:
            # Assuming you want the IP of the first device in the list
            return devices[0]['addresses'][0]
        else:
            return "No devices found"