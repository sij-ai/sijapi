import os
import yaml  
from time import sleep
from pathlib import Path
import ipaddress

import yaml

class Config:
    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as file:
            self.data = yaml.safe_load(file)

    def __getattr__(self, name):
        if name in self.data:
            value = self.data[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")

class ConfigSection:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        if name in self.data:
            value = self.data[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        raise AttributeError(f"ConfigSection has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value

# Load the YAML configuration file
CFG = Config('.config.yaml')

# Access existing attributes
print(CFG.API.PORT)  # Output: localhost

def load_config():
    yaml_file = os.path.join(os.path.dirname(__file__), ".config.yaml")

    HOME_DIR = Path.home()
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_DIR = BASE_DIR / "config"
    ROUTER_DIR = BASE_DIR / "routers"

    DATA_DIR = BASE_DIR / "data"
    os.makedirs(DATA_DIR, exist_ok=True)

    ALERTS_DIR = DATA_DIR / "alerts"
    os.makedirs(ALERTS_DIR, exist_ok=True)

    LOGS_DIR = BASE_DIR / "logs"
    os.makedirs(LOGS_DIR, exist_ok=True)
    REQUESTS_DIR = LOGS_DIR / "requests"
    os.makedirs(REQUESTS_DIR, exist_ok=True)
    REQUESTS_LOG_PATH = LOGS_DIR / "requests.log"
    DOC_DIR = DATA_DIR / "docs"
    os.makedirs(DOC_DIR, exist_ok=True)
    SD_IMAGE_DIR = DATA_DIR / "sd" / "images"
    os.makedirs(SD_IMAGE_DIR, exist_ok=True)
    SD_WORKFLOWS_DIR = DATA_DIR / "sd" / "workflows"



    try:
        with open(yaml_file, 'r') as file:
            config_data = yaml.safe_load(file)

        vars = {
            

            "API": {

            }
        }


        config = Config(config_data)
        return config
    except Exception as e:
        print(f"Error while loading configuration: {e}")
        return None  

def reload_config():
    while True:
        global config
        with open('config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)
        config = Config(config_data)
        sleep(300)  # reload every 5 minutes