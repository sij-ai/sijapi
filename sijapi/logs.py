# logs.py
import os
import sys
from pathlib import Path
import yaml
from loguru import logger as loguru_logger
from typing import Union, Optional

class LogLevels:
    def __init__(self):
        self.levels = {}
        self.default_level = "INFO"
        self.HOME = Path.home()

    def init(self, yaml_path: Union[str, Path]):
        yaml_path = self._resolve_path(yaml_path, 'config')
        
        try:
            with yaml_path.open('r') as file:
                config_data = yaml.safe_load(file)
            
            logs_config = config_data.get('LOGS', {})
            self.default_level = logs_config.get('default', "INFO")
            self.levels = {k: v for k, v in logs_config.items() if k != 'default'}
            
            loguru_logger.info(f"Loaded log levels configuration from {yaml_path}")
        except Exception as e:
            loguru_logger.error(f"Error loading log levels configuration: {str(e)}")
            raise

    def _resolve_path(self, path: Union[str, Path], default_dir: str) -> Path:
        base_path = Path(__file__).parent.parent
        path = Path(path)
        if not path.suffix:
            path = base_path / 'sijapi' / default_dir / f"{path.name}.yaml"
        elif not path.is_absolute():
            path = base_path / path
        return path

    def set_level(self, module, level):
        self.levels[module] = level

    def set_default_level(self, level):
        self.default_level = level

    def get_level(self, module):
        return self.levels.get(module, self.default_level)


class Logger:
    def __init__(self, name):
        self.name = name
        self.logger = loguru_logger
        self.debug_modules = set()
        self.log_levels = LogLevels()
        self.logs_dir = None

    def init(self, yaml_path: Union[str, Path], logs_dir: Path):
        self.log_levels.init(yaml_path)
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set up initial logging configuration
        self.logger.remove()
        log_format = "{time:YYYY-MM-DD HH:mm:ss} - {name} - <level>{level: <8}</level> - <level>{message}</level>"
        self.logger.add(self.logs_dir / 'app.log', rotation="2 MB", level="DEBUG", format=log_format)
        self.logger.add(sys.stdout, level="DEBUG", format=log_format, colorize=True,
                        filter=self._level_filter)

    def setup_from_args(self, args):
        if not self.logs_dir:
            raise ValueError("Logger not initialized. Call init() before setup_from_args().")
        
        # Update log levels based on command line arguments
        for module in args.debug:
            self.log_levels.set_level(module, "DEBUG")
        if hasattr(args, 'info'):
            for module in args.info:
                self.log_levels.set_level(module, "INFO")
        if args.log:
            self.log_levels.set_default_level(args.log.upper())
        
        # Set debug modules
        self.debug_modules = set(args.debug)
        
        # Custom color and style mappings
        self.logger.level("CRITICAL", color="<yellow><bold><MAGENTA>")
        self.logger.level("ERROR", color="<red><bold>")
        self.logger.level("WARNING", color="<yellow><bold>")
        self.logger.level("DEBUG", color="<green><bold>")
        
        self.logger.info(f"Debug modules: {self.debug_modules}")
        self.logger.info(f"Log levels: {self.log_levels.levels}")
        self.logger.info(f"Default log level: {self.log_levels.default_level}")

    def _level_filter(self, record):
        module_level = self.log_levels.get_level(record["name"])
        return record["level"].no >= self.logger.level(module_level).no

    def get_logger(self, module_name):
        level = self.log_levels.get_level(module_name)
        self.logger.debug(f"Creating logger for {module_name} with level {level}")
        return self.logger.bind(name=module_name)

# Global logger instance
L = Logger("Central")

# Function to get module-specific logger
def get_logger(module_name):
    return L.get_logger(module_name)
