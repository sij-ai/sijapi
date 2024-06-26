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
        log_format = (
            "{time:YYYY-MM-DD HH:mm:ss} - "
            "{name} - "
            "<level>{level: <8}</level> - "
            "<level>"
            "{message}"
            "</level>"
        )
        console_level = "DEBUG" if args.debug else "INFO"
        logger.add(
            sys.stdout,
            format=log_format,
            level=console_level,
            colorize=True,
            filter=lambda record: record["level"].name != "INFO",  # Apply colors to all levels except INFO
        )
        
        # Add a separate handler for INFO level without colors
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level: <8} - {message}",
            level="INFO",
            filter=lambda record: record["level"].name == "INFO",
        )


        # Custom color and style mappings
        logger.level("CRITICAL", color="<yellow><bold><MAGENTA>")
        logger.level("ERROR", color="<red><bold>")
        logger.level("WARNING", color="<yellow><bold>") 
        logger.level("DEBUG", color="<green><bold>")

        # Test color output
        self.logger.debug("Debug message (should be italic green)")
        self.logger.info("Info message (should be uncolored)")
        self.logger.warning("Warning message (should be bold orange/yellow)")
        self.logger.error("Error message (should be bold red)")
        self.logger.critical("Critical message (should be bold yellow on magenta)")

    
    def DEBUG(self, log_message): self.logger.debug(log_message)
    def INFO(self, log_message): self.logger.info(log_message)
    def WARN(self, log_message): self.logger.warning(log_message)
    def ERR(self, log_message):
        self.logger.error(log_message)
        self.logger.error(traceback.format_exc())
    def CRIT(self, log_message):
        self.logger.critical(log_message)
        self.logger.critical(traceback.format_exc())

    def get_logger(self):
        return self


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
