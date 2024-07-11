import os
import sys
from loguru import logger

class Logger:
    def __init__(self, name, logs_dir):
        self.logs_dir = logs_dir
        self.name = name
        self.logger = logger
        self.debug_modules = set()

    def setup_from_args(self, args):
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        self.logger.remove()
        
        log_format = "{time:YYYY-MM-DD HH:mm:ss} - {name} - <level>{level: <8}</level> - <level>{message}</level>"
        
        # File handler
        self.logger.add(os.path.join(self.logs_dir, 'app.log'), rotation="2 MB", level="DEBUG", format=log_format)
        
        # Set debug modules
        self.debug_modules = set(args.debug)
        
        # Console handler with custom filter
        def module_filter(record):
            return (record["level"].no >= logger.level(args.log.upper()).no or
                    record["name"] in self.debug_modules)
        
        self.logger.add(sys.stdout, level="DEBUG", format=log_format, filter=module_filter, colorize=True)
        
        # Custom color and style mappings
        self.logger.level("CRITICAL", color="<yellow><bold><MAGENTA>")
        self.logger.level("ERROR", color="<red><bold>")
        self.logger.level("WARNING", color="<yellow><bold>")
        self.logger.level("DEBUG", color="<green><bold>")
        
        self.logger.info(f"Debug modules: {self.debug_modules}")

    def get_module_logger(self, module_name):
        return self.logger.bind(name=module_name)
