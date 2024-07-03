import pandas as pd
import numpy as np
from io import StringIO

import warnings
warnings.filterwarnings("ignore")

import json
from botocore.exceptions import ClientError
import os
import boto3
from box import ConfigBox

from project import logger

s3 = boto3.client('s3')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import string

stopwords=stopwords.words("english")
stopwords.remove("no")

from nltk.stem import PorterStemmer
stem=PorterStemmer()

def read_data_from_s3(bucket, key):
    
    """
    Read a file from an S3 bucket and return as a pandas DataFrame.

    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The key (path) to the file in the bucket.

    Returns:
        DataFrame or None: A pandas DataFrame containing the file data,
                           or None if the file does not exist or is of an unsupported type.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        file_extension = os.path.splitext(key)[1][1:].lower()  # Extract file extension from the key
        if file_extension == 'csv':
            return pd.read_csv(obj['Body'])
        elif file_extension == 'xlsx' or file_extension == 'xls':
            return pd.read_excel(obj['Body'])
        elif file_extension == 'json':
            return pd.read_json(obj['Body'])
        else:
            logger.error(f"Unsupported file type for key '{key}'. Supported types are: 'csv', 'xlsx', 'xls', 'json'.")
            return None
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.error(f"File '{key}' does not exist in bucket '{bucket}'.")
        else:
            logger.error(f"An error occurred: {e}")
        return None

import yaml

# Function to read YAML file
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = ConfigBox(yaml.safe_load(file))
    return data


def create_directory(directory_path):
    """
    Create directory/directories at the given path, including any missing parent directories.

    Args:
        directory_path (str): The path of the directory to be created.
        
    Returns:
        bool: True if the directory/directories were successfully created or already exist, False otherwise.
    """
    try:
        # Create directory and any missing parent directories
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Directory '{directory_path}' created successfully.")
        return True
    except OSError as e:
        logger.error(f"Error creating directory '{directory_path}': {e}")
        return False



def upload_dataframe_to_s3(df, bucket_name, object_key):
    """
    Uploads a DataFrame to an S3 bucket as a CSV file.

    :param df: Pandas DataFrame to upload
    :param bucket_name: Name of the S3 bucket
    :param object_key: S3 object key
    :return: None
    """
    
    # Convert DataFrame to CSV
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Upload the CSV to S3
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=csv_buffer.getvalue())

# Example usage
# df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['A', 'B', 'C']})
# upload_dataframe_to_s3(df, 'your-bucket-name', 'path/to/your/file.csv')


def clean_data(df,col):
    try:
        df.dropna(inplace=True,ignore_index=True)
        df.drop_duplicates(keep='first',inplace=True,ignore_index=True)
        df[col]=df[col].apply(lambda x:word_tokenize(x))
        df[col]=df[col].apply(lambda x:[a.lower().strip() for a in x])
        df[col]=df[col].apply(lambda x:[a for a in x if a not in string.punctuation])
        df[col]=df[col].apply(lambda x:[a for a in x if a not in stopwords ])
        df[col]=df[col].apply(lambda x:[stem.stem(a) for a in x])
        df[col]=df[col].apply(lambda x:' '.join(x))
        df=df[df[col]!='']
    except Exception as e:
        logger.error(e)
        return None
    logger.info("Data cleaning performed")
    return df