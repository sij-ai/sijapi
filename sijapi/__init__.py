import os
import json
import yaml
from pathlib import Path
import ipaddress
import multiprocessing
from dotenv import load_dotenv
from dateutil import tz
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from .logs import Logger
from .classes import AutoResponder, IMAPConfig, SMTPConfig, EmailAccount, EmailContact, IncomingEmail, Database, Geocoder, APIConfig, Configuration

### Initial initialization
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
ENV_PATH = CONFIG_DIR / ".env"
LOGS_DIR = BASE_DIR / "logs"

L = Logger("Central", LOGS_DIR)
os.makedirs(LOGS_DIR, exist_ok=True)
load_dotenv(ENV_PATH)

### API essentials
API = APIConfig.load('api', 'secrets')
Dir = Configuration.load('dirs')
HOST = f"{API.BIND}:{API.PORT}" 
LOCAL_HOSTS = [ipaddress.ip_address(localhost.strip()) for localhost in os.getenv('LOCAL_HOSTS', '127.0.0.1').split(',')] + ['localhost']
SUBNET_BROADCAST = os.getenv("SUBNET_BROADCAST", '10.255.255.255')
MAX_CPU_CORES = min(int(os.getenv("MAX_CPU_CORES", int(multiprocessing.cpu_count()/2))), multiprocessing.cpu_count())
DB = Database.from_env()

News = Configuration.load('news', 'secrets')
IMG = Configuration.load('img', 'secrets')

### Directories & general paths
ROUTER_DIR = BASE_DIR / "routers"
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)
ALERTS_DIR = DATA_DIR / "alerts"
os.makedirs(ALERTS_DIR, exist_ok=True)
REQUESTS_DIR = LOGS_DIR / "requests"
os.makedirs(REQUESTS_DIR, exist_ok=True)
REQUESTS_LOG_PATH = LOGS_DIR / "requests.log"

### LOCATE AND WEATHER LOCALIZATIONS
USER_FULLNAME = os.getenv('USER_FULLNAME')
USER_BIO = os.getenv('USER_BIO')
HOME_ZIP = os.getenv("HOME_ZIP") # unimplemented
NAMED_LOCATIONS = CONFIG_DIR / "named-locations.yaml"
# DB = DATA_DIR / "weatherlocate.db" # deprecated
VISUALCROSSING_BASE_URL = os.getenv("VISUALCROSSING_BASE_URL", "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline")
VISUALCROSSING_API_KEY = os.getenv("VISUALCROSSING_API_KEY")
GEONAMES_TXT = DATA_DIR / "geonames.txt"
LOCATIONS_CSV = DATA_DIR / "US.csv"
TZ = tz.gettz(os.getenv("TZ", "America/Los_Angeles"))
TZ_CACHE = DATA_DIR / "tzcache.json"
GEO = Geocoder(NAMED_LOCATIONS, TZ_CACHE)

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
DOC_DIR = DATA_DIR / "docs"
os.makedirs(DOC_DIR, exist_ok=True)

### DATETIME SCHEMA FOR DAILY NOTE FOLDER HIERARCHY FORMATTING ###
YEAR_FMT = os.getenv("YEAR_FMT")
MONTH_FMT = os.getenv("MONTH_FMT")
DAY_FMT = os.getenv("DAY_FMT")
DAY_SHORT_FMT = os.getenv("DAY_SHORT_FMT")

### Large language model
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434")
LLM_SYS_MSG = os.getenv("SYSTEM_MSG", "You are a helpful AI assistant.")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "llama3")
DEFAULT_VISION = os.getenv("DEFAULT_VISION", "llava")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "Luna")
DEFAULT_11L_VOICE = os.getenv("DEFAULT_11L_VOICE", "Victoria")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
### Summarization
SUMMARY_CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", 16384))  # measured in tokens
SUMMARY_CHUNK_OVERLAP = int(os.getenv("SUMMARY_CHUNK_OVERLAP", 256))  # measured in tokens
SUMMARY_TPW = float(os.getenv("SUMMARY_TPW", 1.3))  # measured in tokens
SUMMARY_LENGTH_RATIO = int(os.getenv("SUMMARY_LENGTH_RATIO", 4))  # measured as original to length ratio
SUMMARY_MIN_LENGTH = int(os.getenv("SUMMARY_MIN_LENGTH", 150))  # measured in tokens
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "dolphin-llama3:8b-256k")
SUMMARY_TOKEN_LIMIT = int(os.getenv("SUMMARY_TOKEN_LIMIT", 16384))
SUMMARY_INSTRUCT = os.getenv('SUMMARY_INSTRUCT', "You are an AI assistant that provides accurate summaries of text -- nothing more and nothing less. You must not include ANY extraneous text other than the sumary. Do not include comments apart from the summary, do not preface the summary, and do not provide any form of postscript. Do not add paragraph breaks. Do not add any kind of formatting. Your response should begin with, consist of, and end with an accurate plaintext summary.")
SUMMARY_INSTRUCT_TTS = os.getenv('SUMMARY_INSTRUCT_TTS', "You are an AI assistant that provides email summaries for Sanjay. Your response will undergo Text-To-Speech conversion and added to Sanjay's private podcast. Providing adequate context (Sanjay did not send this question to you, he will only hear your response) but aiming for conciseness and precision, and bearing in mind the Text-To-Speech conversion (avoiding acronyms and formalities), summarize the following email.")


### Stable diffusion
IMG_DIR = DATA_DIR / "img" / "images"
os.makedirs(IMG_DIR, exist_ok=True)
IMG_WORKFLOWS_DIR = DATA_DIR / "img" / "workflows"
os.makedirs(IMG_WORKFLOWS_DIR, exist_ok=True)
COMFYUI_URL = os.getenv('COMFYUI_URL', "http://localhost:8188")
COMFYUI_DIR = Path(os.getenv('COMFYUI_DIR'))
COMFYUI_OUTPUT_DIR = COMFYUI_DIR / 'output'
COMFYUI_LAUNCH_CMD = os.getenv('COMFYUI_LAUNCH_CMD', 'mamba activate comfyui && python main.py')
IMG_CONFIG_PATH = CONFIG_DIR / 'img.yaml'

### ASR
ASR_DIR = DATA_DIR / "asr"
os.makedirs(ASR_DIR, exist_ok=True)
WHISPER_CPP_DIR = Path(Dir.HOME) / str(os.getenv("WHISPER_CPP_DIR"))
WHISPER_CPP_MODELS = os.getenv('WHISPER_CPP_MODELS', 'NULL,VOID').split(',')

### TTS
PREFERRED_TTS = os.getenv("PREFERRED_TTS", "None")
TTS_DIR = DATA_DIR / "tts"
os.makedirs(TTS_DIR, exist_ok=True)
VOICE_DIR = TTS_DIR / 'voices'
os.makedirs(VOICE_DIR, exist_ok=True)
PODCAST_DIR = os.getenv("PODCAST_DIR", TTS_DIR / "sideloads")
os.makedirs(PODCAST_DIR, exist_ok=True)
TTS_OUTPUT_DIR = TTS_DIR / 'outputs'
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
TTS_SEGMENTS_DIR = TTS_DIR / 'segments'
os.makedirs(TTS_SEGMENTS_DIR, exist_ok=True)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


### Calendar & email account
MS365_TOGGLE = True if os.getenv("MS365_TOGGLE") == "True" else False
ICAL_TOGGLE = True if os.getenv("ICAL_TOGGLE") == "True" else False
ICS_PATH = DATA_DIR / 'calendar.ics' # deprecated now, but maybe revive?
ICALENDARS = os.getenv('ICALENDARS', 'NULL,VOID').split(',')

EMAIL_CONFIG = CONFIG_DIR / "email.yaml"
EMAIL_LOGS = LOGS_DIR / "email"
os.makedirs(EMAIL_LOGS, exist_ok = True)

### Courtlistener & other webhooks
COURTLISTENER_DOCKETS_DIR = DATA_DIR / "courtlistener" / "dockets"
os.makedirs(COURTLISTENER_DOCKETS_DIR, exist_ok=True)
COURTLISTENER_SEARCH_DIR = DATA_DIR / "courtlistener" / "cases"
os.makedirs(COURTLISTENER_SEARCH_DIR, exist_ok=True)
CASETABLE_PATH = DATA_DIR / "courtlistener" / "cases.json"
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY")
COURTLISTENER_BASE_URL = os.getenv("COURTLISTENER_BASE_URL", "https://www.courtlistener.com")
COURTLISTENER_DOCKETS_URL = "https://www.courtlistener.com/api/rest/v3/dockets/"

### Keys & passwords
PUBLIC_KEY_FILE = os.getenv("PUBLIC_KEY_FILE", 'you_public_key.asc')
PUBLIC_KEY = (BASE_DIR.parent / PUBLIC_KEY_FILE).read_text()
MAC_ID = os.getenv("MAC_ID")
MAC_UN = os.getenv("MAC_UN")
MAC_PW = os.getenv("MAC_PW")
TIMING_API_KEY = os.getenv("TIMING_API_KEY")
TIMING_API_URL = os.getenv("TIMING_API_URL", "https://web.timingapp.com/api/v1")
PHOTOPRISM_URL = os.getenv("PHOTOPRISM_URL")
PHOTOPRISM_USER = os.getenv("PHOTOPRISM_USER")
PHOTOPRISM_PASS = os.getenv("PHOTOPRISM_PASS")

### Tailscale
TS_IP = ipaddress.ip_address(os.getenv("TS_IP", "NULL"))
TS_SUBNET = ipaddress.ip_network(os.getenv("TS_SUBNET")) if os.getenv("TS_SUBNET") else None
TS_ID = os.getenv("TS_ID", "NULL")
TS_TAILNET = os.getenv("TS_TAILNET", "NULL")
TS_ADDRESS = f"http://{TS_ID}.{TS_TAILNET}.ts.net"

### Cloudflare
CF_API_BASE_URL = os.getenv("CF_API_BASE_URL")
CF_TOKEN = os.getenv("CF_TOKEN")
CF_IP = DATA_DIR / "cf_ip.txt" # to be deprecated soon
CF_DOMAINS_PATH = DATA_DIR / "cf_domains.json" # to be deprecated soon

### Caddy - not fully implemented
API.URL = os.getenv("API.URL")
CADDY_SERVER = os.getenv('CADDY_SERVER', None)
CADDYFILE_PATH = os.getenv("CADDYFILE_PATH", "") if CADDY_SERVER is not None else None
CADDY_API_KEY = os.getenv("CADDY_API_KEY")


### Microsoft Graph
MS365_CLIENT_ID = os.getenv('MS365_CLIENT_ID')
MS365_SECRET = os.getenv('MS365_SECRET')
MS365_TENANT_ID = os.getenv('MS365_TENANT_ID') 
MS365_CERT_PATH = CONFIG_DIR / 'MS365' / '.cert.pem' # deprecated
MS365_KEY_PATH = CONFIG_DIR / 'MS365' / '.cert.key' # deprecated  
MS365_KEY = MS365_KEY_PATH.read_text()
MS365_TOKEN_PATH = CONFIG_DIR / 'MS365' / '.token.txt'
MS365_THUMBPRINT = os.getenv('MS365_THUMBPRINT')

MS365_LOGIN_URL = os.getenv("MS365_LOGIN_URL", "https://login.microsoftonline.com")
MS365_AUTHORITY_URL = f"{MS365_LOGIN_URL}/{MS365_TENANT_ID}"
MS365_REDIRECT_PATH = os.getenv("MS365_REDIRECT_PATH", "https://api.sij.ai/o365/oauth_redirect")
MS365_SCOPE = os.getenv("MS365_SCOPE", 'Calendars.Read,Calendars.ReadWrite,offline_access').split(',')

### Maintenance
GARBAGE_COLLECTION_INTERVAL = 60 * 60  # Run cleanup every hour
GARBAGE_TTL = 60 * 60 * 24  # Delete files older than 24 hours