import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from colorama import Fore, Back, Style, init as colorama_init
import traceback

# Force colorama to initialize for the current platform
colorama_init(autoreset=True, strip=False, convert=True)

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""
    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Back.WHITE,
    }

    def format(self, record):
        log_message = super().format(record)
        color = self.COLOR_MAP.get(record.levelno, '')
        return f"{color}{log_message}{Style.RESET_ALL}"

class Logger:
    def __init__(self, name, logs_dir):
        self.logs_dir = logs_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

    def setup_from_args(self, args):
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # File handler
        handler_path = os.path.join(self.logs_dir, 'app.log')
        file_handler = RotatingFileHandler(handler_path, maxBytes=2000000, backupCount=10)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)  # Explicitly use sys.stdout
        console_formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Set console handler level based on args
        if args.debug:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Test color output
        self.logger.debug("Debug message (should be Cyan)")
        self.logger.info("Info message (should be Green)")
        self.logger.warning("Warning message (should be Yellow)")
        self.logger.error("Error message (should be Red)")
        self.logger.critical("Critical message (should be Magenta on White)")

    def get_logger(self):
        return self.logger

# Add this at the end of the file for testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logger = Logger("test", "logs")
    logger.setup_from_args(args)
    test_logger = logger.get_logger()

    print("FORCE_COLOR:", os.environ.get('FORCE_COLOR'))
    print("NO_COLOR:", os.environ.get('NO_COLOR'))
    print("TERM:", os.environ.get('TERM'))
    print("PYCHARM_HOSTED:", os.environ.get('PYCHARM_HOSTED'))
    print("PYTHONIOENCODING:", os.environ.get('PYTHONIOENCODING'))

    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")