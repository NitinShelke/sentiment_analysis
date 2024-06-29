from project import logger
from project.utils.common import read_yaml_file,read_data_from_s3,upload_dataframe_to_s3,clean_data

config=read_yaml_file("config/config.yaml")

train=read_data_from_s3(config.s3.bucket,config.s3.train_object)

train.dropna(inplace=True,ignore_index=True)
train.drop_duplicates(keep='first',inplace=True,ignore_index=True)

train=clean_data(train,config.data.feature)

upload_dataframe_to_s3(train,config.s3.bucket,config.s3.save_object)