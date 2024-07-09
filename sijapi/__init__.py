# __init__.py
import os
from pathlib import Path
import ipaddress
from dotenv import load_dotenv
from dateutil import tz
from pathlib import Path
from .logs import Logger
from .classes import Database, Geocoder, APIConfig, Configuration, EmailConfiguration, Dir

### Initial initialization
API = APIConfig.load('api', 'secrets')
Dir = Dir()
ENV_PATH = Dir.CONFIG / ".env"
LOGS_DIR = Dir.LOGS
L = Logger("Central", LOGS_DIR)
os.makedirs(LOGS_DIR, exist_ok=True)
load_dotenv(ENV_PATH)

### API essentials
DB = Database.from_yaml('db.yaml')

ASR = Configuration.load('asr')
IMG = Configuration.load('img')
Cal = Configuration.load('cal', 'secrets')
print(f"Cal configuration: {Cal.__dict__}")
Email = EmailConfiguration.load('email', 'secrets')
LLM = Configuration.load('llm', 'secrets')
News = Configuration.load('news', 'secrets')
Obsidian = Configuration.load('obsidian')
TTS = Configuration.load('tts', 'secrets')
CourtListener = Configuration.load('courtlistener', 'secrets')
Tailscale = Configuration.load('tailscale', 'secrets')
Cloudflare = Configuration.load('cloudflare', 'secrets')


### Directories & general paths
REQUESTS_DIR = LOGS_DIR / "requests"
os.makedirs(REQUESTS_DIR, exist_ok=True)
REQUESTS_LOG_PATH = LOGS_DIR / "requests.log"

### LOCATE AND WEATHER LOCALIZATIONS
# DB = DATA_DIR / "weatherlocate.db" # deprecated
VISUALCROSSING_BASE_URL = os.getenv("VISUALCROSSING_BASE_URL", "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline")
VISUALCROSSING_API_KEY = os.getenv("VISUALCROSSING_API_KEY")
TZ = tz.gettz(os.getenv("TZ", "America/Los_Angeles"))
TZ_CACHE = Dir.DATA / "tzcache.json"
GEO = Geocoder(Dir.config.locations, TZ_CACHE)

### Obsidian & notes
ALLOWED_FILENAME_CHARS = r'[^\w \.-]'
MAX_PATH_LENGTH = 254
OBSIDIAN_VAULT_DIR = Path(os.getenv("OBSIDIAN_BASE_DIR") or Path(Dir.HOME) / "Nextcloud" / "notes")
OBSIDIAN_JOURNAL_DIR = OBSIDIAN_VAULT_DIR / "journal"
OBSIDIAN_RESOURCES_DIR = "obsidian/resources"
OBSIDIAN_BANNER_DIR = f"{OBSIDIAN_RESOURCES_DIR}/banners"
os.makedirs(Path(OBSIDIAN_VAULT_DIR) / OBSIDIAN_BANNER_DIR, exist_ok=True)
OBSIDIAN_BANNER_SCENE = os.getenv("OBSIDIAN_BANNER_SCENE", "wallpaper")
OBSIDIAN_CHROMADB_COLLECTION = os.getenv("OBSIDIAN_CHROMADB_COLLECTION", "obsidian")
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", OBSIDIAN_VAULT_DIR / "archive"))
os.makedirs(ARCHIVE_DIR, exist_ok=True)

### DATETIME SCHEMA FOR DAILY NOTE FOLDER HIERARCHY FORMATTING ###
YEAR_FMT = os.getenv("YEAR_FMT")
MONTH_FMT = os.getenv("MONTH_FMT")
DAY_FMT = os.getenv("DAY_FMT")
DAY_SHORT_FMT = os.getenv("DAY_SHORT_FMT")

### Keys & passwords
MAC_ID = os.getenv("MAC_ID")
MAC_UN = os.getenv("MAC_UN")
MAC_PW = os.getenv("MAC_PW")
TIMING_API_KEY = os.getenv("TIMING_API_KEY")
TIMING_API_URL = os.getenv("TIMING_API_URL", "https://web.timingapp.com/api/v1")

### Caddy - not fully implemented
API.URL = os.getenv("API.URL")
CADDY_SERVER = os.getenv('CADDY_SERVER', None)
CADDYFILE_PATH = os.getenv("CADDYFILE_PATH", "") if CADDY_SERVER is not None else None
CADDY_API_KEY = os.getenv("CADDY_API_KEY")