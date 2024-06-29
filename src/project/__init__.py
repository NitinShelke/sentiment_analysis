import logging
import os

class CustomLogger:
    def __init__(self, log_file="RUNNING_LOGGS.log"):
        self.logger = logging.getLogger("CustomLogger")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create a console handler and set level to debug
        log_folder="Loggings"
        if log_folder:
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)

        log_file=os.path.join(log_folder,log_file)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message,exc_info=True)

    def critical(self, message):
        self.logger.critical(message)
        
logger=CustomLogger()

