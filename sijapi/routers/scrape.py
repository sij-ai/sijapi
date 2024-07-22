import asyncio
import json
import re
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
import aiohttp
import PyPDF2
import io
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pathlib import Path
from sijapi import Scrape, L, Dir

logger = L.get_module_logger('scrape')
def debug(text: str): logger.debug(text)
def info(text: str): logger.info(text)
def warn(text: str): logger.warning(text)
def err(text: str): logger.error(text)
def crit(text: str): logger.critical(text)

scrape = APIRouter()

# Ensure Dir.DATA is a Path object
Dir.DATA = Path(Dir.DATA).expanduser()

def save_to_json(data: List[Dict], output_file: str):
    output_path = Dir.DATA / output_file
    info(f"Saving data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    info(f"Data saved successfully to {output_path}")

def load_from_json(output_file: str) -> List[Dict]:
    output_path = Dir.DATA / output_file
    info(f"Loading data from {output_path}")
    try:
        with open(output_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        warn(f"File {output_path} not found")
        return []

async def fetch_content(config: Any) -> str:
    info(f"Fetching content from {config.url}")
    if config.content.js_render:
        return await fetch_with_selenium(config.url)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(config.url) as response:
            if config.content.type == 'pdf':
                return await handle_pdf(response)
            elif config.content.type in ['html', 'xml']:
                return await handle_html_xml(response, config.content.selector)
            elif config.content.type == 'json':
                return await handle_json(response)
            elif config.content.type == 'txt':
                return await response.text()
            else:
                warn(f"Unsupported content type: {config.content.type}")
                return await response.text()

async def fetch_with_selenium(url: str) -> str:
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    content = driver.page_source
    driver.quit()
    return content

async def handle_pdf(response):
    pdf_content = await response.read()
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return "\n".join(page.extract_text() for page in pdf_reader.pages)

async def handle_html_xml(response, selector):
    content = await response.text()
    soup = BeautifulSoup(content, 'html.parser')
    if selector:
        return soup.select_one(selector).get_text()
    return soup.get_text()

async def handle_json(response):
    return await response.json()

def apply_processing_step(data: Any, step: Any) -> Any:
    info(f"Applying processing step: {step.type}")
    if step.type == 'regex_split':
        return re.split(step.pattern, data)[1:]
    elif step.type == 'keyword_filter':
        return [item for item in data if any(keyword.lower() in str(item).lower() for keyword in step.keywords)]
    elif step.type == 'regex_extract':
        if isinstance(data, list):
            return [apply_regex_extract(item, step.extractions) for item in data]
        return apply_regex_extract(data, step.extractions)
    debug(f"Unknown processing step type: {step.type}")
    return data

def apply_regex_extract(text: str, extractions: List[Any]) -> Dict:
    debug(f"Applying regex extraction on text of length {len(text)}")
    result = {}
    for extraction in extractions:
        extraction_dict = extraction.dict() if hasattr(extraction, 'dict') else extraction
        flags = sum(getattr(re, flag.upper()) for flag in extraction_dict.get('flags', []))
        
        pattern = extraction_dict['pattern']
        matches = re.findall(pattern, text, flags=flags)
        if matches:
            if extraction_dict.get('all_matches', False):
                if extraction_dict.get('group_names'):
                    result[extraction_dict['name']] = [dict(zip(extraction_dict['group_names'], match)) for match in matches]
                else:
                    result[extraction_dict['name']] = matches
            else:
                result[extraction_dict['name']] = matches[-1].strip()  # Take the last match
    
    debug(f"Extracted {len(result)} items")
    return result

def apply_post_processing(data: List[Dict], post_processing: List[Any]) -> List[Dict]:
    info("Applying post-processing steps")
    for step in post_processing:
        if step.type == 'custom':
            data = globals()[step.function](data)
    return data

def data_has_changed(new_data: List[Dict], old_data: List[Dict]) -> bool:
    return new_data != old_data

@scrape.get("/scrape/{config_name}")
async def scrape_site(config_name: str):
    info(f"Starting scrape operation for {config_name}")
    
    if not hasattr(Scrape, 'configurations'):
        # If 'configurations' doesn't exist, assume the entire Scrape object is the configuration
        config = Scrape if Scrape.name == config_name else None
    else:
        config = next((c for c in Scrape.configurations if c.name == config_name), None)
    
    if not config:
        raise HTTPException(status_code=404, detail=f"Configuration '{config_name}' not found")
    
    raw_data = await fetch_content(config)
    processed_data = raw_data
    
    for step in config.processing:
        processed_data = apply_processing_step(processed_data, step)
    
    processed_data = apply_post_processing(processed_data, config.post_processing)
    
    # Resolve Dir.DATA in the output file path
    output_file = config.output_file.replace('{{ Dir.DATA }}', str(Dir.DATA))
    previous_data = load_from_json(output_file)
    
    if data_has_changed(processed_data, previous_data):
        save_to_json(processed_data, output_file)
        info("Scrape completed with updates")
        return {"message": "Site updated", "data": processed_data}
    else:
        info("Scrape completed with no updates")
        return {"message": "No updates", "data": processed_data}

def apply_post_processing(data: List[Dict], post_processing: List[Any]) -> List[Dict]:
    info("Applying post-processing steps")
    for step in post_processing:
        if step.type == 'regex_extract':
            for entry in data:
                if step.field in entry:
                    matches = re.findall(step.pattern, entry[step.field])
                    if step.all_matches:
                        entry[step.output_field] = [step.format.format(*match) for match in matches]
                    elif matches:
                        entry[step.output_field] = step.format.format(*matches[0])
    return data
