'''
WORK IN PROGRESS: This module will handle tasks related to multi-server environments.
'''
from fastapi import APIRouter, HTTPException
import asyncio
import logging
from sijapi.utilities import run_ssh_command
from sijapi import L, REBOOT_SCRIPT_PATH, HOST_CONFIG, API_CONFIG

dist = APIRouter()
logger = L.get_module_logger("dist")

@dist.get("/update-restart-others")
async def update_and_restart_others():
    results = []
    for server in API_CONFIG.servers:
        if HOST_CONFIG.id != server.id:
            try:
                output, error = await run_ssh_command(server, f"bash {server.scripts.update_and_restart}")
                results.append({"server": server.id, "status": "success", "output": output, "error": error})
            except Exception as e:
                results.append({"server": server.id, "status": "failed", "error": str(e)})
    return {"message": "Update and restart process initiated for other servers.", "results": results}

@dist.get("/update-restart-self")
async def update_and_restart_self(safe: bool = True):
    if safe and not await ensure_redundancy():
        raise HTTPException(status_code=400, detail="Cannot safely restart: no redundancy available")
    try:
        process = await asyncio.create_subprocess_exec(
            "bash", API_CONFIG.servers[HOST_CONFIG.id].scripts.update_and_restart,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        logger.info(f"Update and restart initiated for self. Stdout: {stdout.decode()}. Stderr: {stderr.decode()}")
        return {"message": "Update and restart process initiated for this server."}
    except Exception as e:
        logger.error(f"Failed to initiate update and restart for self: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate update and restart: {str(e)}")

@dist.get("/update-and-restart-all")
async def update_and_restart_all():
    others_result = await update_and_restart_others()
    self_result = await update_and_restart_self(safe=False)
    return {"others": others_result, "self": self_result}

async def ensure_redundancy():
    import aiohttp
    redundancy = False
    async with aiohttp.ClientSession() as session:
        for server in API_CONFIG.servers:
            if server.id != HOST_CONFIG.id:
                try:
                    async with session.get(f"{server.protocol}://{server.ip}:{server.port}/health") as response:
                        if response.status // 100 == 2:
                            redundancy = True
                            break
                except aiohttp.ClientError:
                    logger.warning(f"Failed to check health of server {server.id}")
    return redundancy
