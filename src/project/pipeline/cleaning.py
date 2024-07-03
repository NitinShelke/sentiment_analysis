from project.utils.common import read_yaml_file, read_data_from_s3, upload_dataframe_to_s3, clean_data
from project import logger

class CleaningPipeline:
    def __init__(self, config_path):
        self.config = read_yaml_file(config_path)

    def load_data(self):
        self.train = read_data_from_s3(self.config.s3.bucket, self.config.s3.train_object)
        return self.train

    def clean_data(self):
        self.train.dropna(inplace=True, ignore_index=True)
        self.train.drop_duplicates(keep='first', inplace=True, ignore_index=True)
        self.train = clean_data(self.train, self.config.data.feature)
        return self.train

    def save_data(self):
        upload_dataframe_to_s3(self.train, self.config.s3.bucket, self.config.s3.save_object)

def main():
    processor = CleaningPipeline("config/config.yaml")
    logger.info(">>>>>>>>>> Data preprocessing started <<<<<<<<<<<<<<<")
    processor.load_data()
    logger.info(">>>>>>>>>> Data Loaded from storage   <<<<<<<<<<<<<<<")
    processor.clean_data()
    logger.info(">>>>>>>>>> Data Cleaning Completed <<<<<<<<<<<<<<<")
    processor.save_data()
    logger.info(">>>>>>>>>> Data Saved to Storage <<<<<<<<<<<<<<<")
    logger.info(">>>>>>>>>> Data preprocessing Completed <<<<<<<<<<<<<<<")

if __name__ == "__main__":
    main()