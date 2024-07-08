from project.pipeline.cleaning import CleaningPipeline
from project.pipeline.training import TrainingPipeline
from  project.pipeline.prediction import PredictPipeline
from project import logger

"""processor = CleaningPipeline("config/config.yaml")
logger.info(">>>>>>>>>> Data preprocessing started <<<<<<<<<<<<<<<")
processor.load_data()
logger.info(">>>>>>>>>> Data Loaded from storage   <<<<<<<<<<<<<<<")
processor.clean_data()
logger.info(">>>>>>>>>> Data Cleaning Completed <<<<<<<<<<<<<<<")
processor.save_data()
logger.info(">>>>>>>>>> Data Saved to Storage <<<<<<<<<<<<<<<")
logger.info(">>>>>>>>>> Data preprocessing Completed <<<<<<<<<<<<<<<")

processor = TrainingPipeline()
logger.info(">>>>>>>>>> Training pipeline started <<<<<<<<<<<<<<<")
processor.train()
logger.info(">>>>>>>>>> Training completed <<<<<<<<<<<<<<<")
processor.register_model()
logger.info(">>>>>>>>>> Model registry process: Complete <<<<<<<<<<<<<<<")"""

text='i am feeling realy good today'
pipeline=PredictPipeline()
output=pipeline.predict(text)
print("sentence={}\nsentiment={}".format(text,output))
