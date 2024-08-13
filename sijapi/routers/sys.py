# routers/sys.py

import os
import httpx
import socket
from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import text, select
from tailscale import Tailscale
from sijapi import Sys, Db, TS_ID
from sijapi.logs import get_logger
from sijapi.serialization import json_loads
from sijapi.database import QueryTracking

l = get_logger(__name__)

sys = APIRouter()

@sys.get("/health")
def get_health():
    return {"status": "ok"}

@sys.get("/id")
def get_id() -> str:
    return TS_ID

@sys.get("/routers")
def get_routers() -> str:
    active_modules = [module for module, is_active in Sys.MODULES.__dict__.items() if is_active]
    return active_modules

@sys.get("/ip")
def get_local_ip():
    """Get the server's local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((f'{Sys.SUBNET_BROADCAST}', 1))
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

@sys.get("/ts_ip")
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

async def sync_process():
    async with Db.sessions[TS_ID]() as session:
        # Find unexecuted queries
        unexecuted_queries = await session.execute(
            select(QueryTracking).where(~QueryTracking.completed_by.has_key(TS_ID)).order_by(QueryTracking.id)
        )

        for query in unexecuted_queries:
            try:
                params = json_loads(query.args)
                await session.execute(text(query.query), params)
                actual_checksum = await Db._local_compute_checksum(query.query, params)
                if actual_checksum != query.result_checksum:
                    l.error(f"Checksum mismatch for query ID {query.id}")
                    continue
                
                # Update the completed_by field
                query.completed_by[TS_ID] = True
                await session.commit()
                
                l.info(f"Successfully executed and verified query ID {query.id}")
            except Exception as e:
                l.error(f"Failed to execute query ID {query.id} during sync: {str(e)}")
                await session.rollback()

        l.info(f"Sync process completed. Executed {unexecuted_queries.rowcount} queries.")

    # After executing all queries, perform combinatorial sync
    await Db.sync_query_tracking()

@sys.post("/db/sync")
async def db_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_process)
    return {"message": "Sync process initiated"}
