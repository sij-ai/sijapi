'''
Used for port-forwarding and reverse proxy configurations.
'''
#routers/forward.py

import asyncio
from fastapi import APIRouter
from sijapi import Sys, Serve
from sijapi.logs import get_logger

l = get_logger(__name__)

forward = APIRouter()


async def forward_traffic(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, destination: str):
    try:
        dest_host, dest_port = destination.split(':')
        dest_port = int(dest_port)
    except ValueError:
        l.warning(f"Invalid destination format: {destination}. Expected 'host:port'.")
        writer.close()
        await writer.wait_closed()
        return
    
    try:
        dest_reader, dest_writer = await asyncio.open_connection(dest_host, dest_port)
    except Exception as e:
        l.warning(f"Failed to connect to destination {destination}: {str(e)}")
        writer.close()
        await writer.wait_closed()
        return

    async def forward(src, dst):
        try:
            while True:
                data = await src.read(8192)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except Exception as e:
            l.warning(f"Error in forwarding: {str(e)}")
        finally:
            dst.close()
            await dst.wait_closed()

    await asyncio.gather(
        forward(reader, dest_writer),
        forward(dest_reader, writer)
    )

async def start_server(source: str, destination: str):
    if ':' in source:
        host, port = source.split(':')
        port = int(port)
    else:
        host = source
        port = 80

    server = await asyncio.start_server(
        lambda r, w: forward_traffic(r, w, destination),
        host,
        port
    )

    async with server:
        await server.serve_forever()


async def start_port_forwarding():
    if hasattr(Serve, 'forwarding_rules'):
        for rule in Serve.forwarding_rules:
            asyncio.create_task(start_server(rule.source, rule.destination))
    else:
        l.warning("No forwarding rules found in the configuration.")


@forward.get("/forward_status")
async def get_forward_status():
    if hasattr(Serve, 'forwarding_rules'):
        return {"status": "active", "rules": Serve.forwarding_rules}
    else:
        return {"status": "inactive", "message": "No forwarding rules configured"}


asyncio.create_task(start_port_forwarding())
