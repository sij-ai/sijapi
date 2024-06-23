'''
IN DEVELOPMENT - Cloudflare + Caddy module. Based on a bash script that's able to rapidly deploy new Cloudflare subdomains on new Caddy reverse proxy configurations, managing everything including restarting Caddy. The Python version needs more testing before actual use.
'''
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse, JSONResponse
from typing import Optional
from sijapi import DEBUG, INFO, WARN, ERR, CRITICAL
from sijapi import CF_TOKEN, CADDYFILE_PATH, CF_API_BASE_URL, CF_IP
import httpx
import asyncio
from asyncio import sleep
import os

cf = APIRouter()

class DNSRecordRequest(BaseModel):
    full_domain: str
    ip: Optional[str] = None
    port: str


# Update to make get_zone_id async
async def get_zone_id(domain: str) -> str:
    url = f"{CF_API_BASE_URL}/zones"
    headers = {
        "Authorization": f"Bearer {CF_TOKEN}",
        "Content-Type": "application/json"
    }
    params = {"name": domain}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    if data['success']:
        if len(data['result']) > 0:
            return data['result'][0]['id']
        else:
            raise ValueError(f"No Zone ID found for domain '{domain}'")
    else:
        errors = ', '.join(err['message'] for err in data['errors'])
        raise ValueError(f"Cloudflare API returned errors: {errors}")



async def update_caddyfile(full_domain, caddy_ip, port):
    caddy_config = f"""
{full_domain} {{
    reverse_proxy {caddy_ip}:{port}
    tls {{
        dns cloudflare {{"$CLOUDFLARE_API_TOKEN"}}
    }}
}}
"""
    with open(CADDYFILE_PATH, 'a') as file:
        file.write(caddy_config)

    # Using asyncio to create subprocess
    proc = await asyncio.create_subprocess_exec("sudo", "systemctl", "restart", "caddy")
    await proc.communicate()


# Retry mechanism for API calls
async def retry_request(url, headers, max_retries=5, backoff_factor=1):
    for retry in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, httpx.ConnectTimeout) as e:
            ERR(f"Request failed: {e}. Retrying {retry + 1}/{max_retries}...")
            await sleep(backoff_factor * (2 ** retry))
    raise HTTPException(status_code=500, detail="Max retries exceeded for Cloudflare API request")

# Helper function to load Caddyfile domains
def load_caddyfile_domains():
    with open(CADDYFILE_PATH, 'r') as file:
        caddyfile_content = file.read()
    domains = []
    for line in caddyfile_content.splitlines():
        if line.strip() and not line.startswith('#'):
            if "{" in line:
                domain = line.split("{")[0].strip()
                domains.append(domain)
    return domains

# Endpoint to add new configuration to Cloudflare, Caddyfile, and cf_domains.json
@cf.post("/cf/add_config")
async def add_config(record: DNSRecordRequest):
    full_domain = record.full_domain
    caddy_ip = record.ip or "localhost"
    port = record.port

    # Extract subdomain and domain
    parts = full_domain.split(".")
    if len(parts) == 2:
        domain = full_domain
        subdomain = "@"
    else:
        subdomain = parts[0]
        domain = ".".join(parts[1:])

    zone_id = await get_zone_id(domain)
    if not zone_id:
        raise HTTPException(status_code=400, detail=f"Zone ID for {domain} could not be found")

    # API call setup for Cloudflare A record
    endpoint = f"{CF_API_BASE_URL}/zones/{zone_id}/dns_records"
    headers = {
        "Authorization": f"Bearer {CF_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "type": "A",
        "name": subdomain,
        "content": CF_IP,
        "ttl": 120,
        "proxied": True
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(endpoint, headers=headers, json=data)
    
    result = response.json()

    if not result.get("success", False):
        error_message = result.get("errors", [{}])[0].get("message", "Unknown error")
        error_code = result.get("errors", [{}])[0].get("code", "Unknown code")
        raise HTTPException(status_code=400, detail=f"Failed to create A record: {error_message} (Code: {error_code})")

    # Update Caddyfile
    await update_caddyfile(full_domain, caddy_ip, port)
    
    return {"message": "Configuration added successfully"}



@cf.get("/cf/list_zones")
async def list_zones_endpoint():
    domains = await list_zones()
    return JSONResponse(domains)

async def list_zones():
    endpoint = f"{CF_API_BASE_URL}/zones"
    headers = {
        "Authorization": f"Bearer {CF_TOKEN}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:  # async http call
        response = await client.get(endpoint, headers=headers)
    response.raise_for_status()
    
    result = response.json()

    if not result.get("success"):
        raise HTTPException(status_code=400, detail="Failed to retrieve zones from Cloudflare")

    zones = result.get("result", [])
    domains = {}

    for zone in zones:
        zone_id = zone.get("id")
        zone_name = zone.get("name")
        domains[zone_name] = {"zone_id": zone_id}

        records_endpoint = f"{CF_API_BASE_URL}/zones/{zone_id}/dns_records"
        async with httpx.AsyncClient() as client:  # async http call
            records_response = await client.get(records_endpoint, headers=headers)
        records_result = records_response.json()

        if not records_result.get("success"):
            raise HTTPException(status_code=400, detail=f"Failed to retrieve DNS records for zone {zone_name}")

        records = records_result.get("result", [])
        for record in records:
            record_id = record.get("id")
            domain_name = record.get("name").replace(f".{zone_name}", "")
            domains[zone_name].setdefault(domain_name, {})["dns_id"] = record_id

    return domains

@cf.get("/cf/compare_caddy", response_class=PlainTextResponse)
async def crossreference_caddyfile():
    cf_domains_data = await list_zones()
    caddyfile_domains = load_caddyfile_domains()

    cf_domains_list = [
        f"{sub}.{domain}" if sub != "@" else domain 
        for domain, data in cf_domains_data.items() 
        for sub in data.get("subdomains", {}).keys()
    ]
    caddyfile_domains_set = set(caddyfile_domains)
    cf_domains_set = set(cf_domains_list)

    only_in_caddyfile = caddyfile_domains_set - cf_domains_set
    only_in_cf_domains = cf_domains_set - caddyfile_domains_set

    markdown_output = "# Cross-reference cf_domains.json and Caddyfile\n\n"
    markdown_output += "## Domains only in Caddyfile:\n\n"
    for domain in only_in_caddyfile:
        markdown_output += f"- **{domain}**\n"

    markdown_output += "\n## Domains only in cf_domains.json:\n\n"
    for domain in only_in_cf_domains:
        markdown_output += f"- **{domain}**\n"
    
    return markdown_output
