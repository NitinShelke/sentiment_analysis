from project.pipeline.cleaning import CleaningPipeline
from project import logger

processor = CleaningPipeline("config/config.yaml")
logger.info(">>>>>>>>>> Data preprocessing started <<<<<<<<<<<<<<<")
processor.load_data()
logger.info(">>>>>>>>>> Data Loaded from storage   <<<<<<<<<<<<<<<")
processor.clean_data()
logger.info(">>>>>>>>>> Data Cleaning Completed <<<<<<<<<<<<<<<")
processor.save_data()
logger.info(">>>>>>>>>> Data Saved to Storage <<<<<<<<<<<<<<<")
logger.info(">>>>>>>>>> Data preprocessing Completed <<<<<<<<<<<<<<<")