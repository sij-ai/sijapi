import os
import sys
from loguru import logger
import traceback

class Logger:
    def __init__(self, name, logs_dir):
        self.logs_dir = logs_dir
        self.name = name
        self.logger = logger.bind(name=name)

    def setup_from_args(self, args):
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # Remove default logger
        logger.remove()

        # File handler
        handler_path = os.path.join(self.logs_dir, 'app.log')
        logger.add(handler_path, rotation="2 MB", compression="zip", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}")
        
        # Console handler
        log_format = "<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> - <cyan>{name}</cyan> - <level>{level: <8}</level> - <level>{message}</level>"
        console_level = "DEBUG" if args.debug else "INFO"
        logger.add(sys.stdout, format=log_format, level=console_level, colorize=True)

        # Test color output
        self.logger.debug("Debug message (should be Cyan)")
        self.logger.info("Info message (should be Green)")
        self.logger.warning("Warning message (should be Yellow)")
        self.logger.error("Error message (should be Red)")
        self.logger.critical("Critical message (should be Magenta)")

    def get_logger(self):
        return self.logger

# Add this at the end of the file for testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logger_instance = Logger("test", "logs")
    logger_instance.setup_from_args(args)
    test_logger = logger_instance.get_logger()

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
