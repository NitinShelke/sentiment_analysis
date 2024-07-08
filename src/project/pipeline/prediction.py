from project import logger
from project.utils.common import read_yaml_file, clean_data

import mlflow
import boto3
import pandas as pd
import joblib
import os

# Load configuration
config = read_yaml_file("config/config.yaml")
model_name = config.mlflow.model_name
tracking_uri = config.mlflow.tracking_uri
mlflow.set_tracking_uri(tracking_uri)

class PredictPipeline:
    def __init__(self):
        pass

    def load_latest_model_and_artifacts(self):
        client = mlflow.tracking.MlflowClient()
        
        try:
            # Get the latest version of the model
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            
            if not latest_versions:
                logger.error(f"No registered versions found for model {model_name}.")
            
            latest_version = latest_versions[-1]
            run_id = latest_version.run_id

            # Load the latest model
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            
            local_artifact_path = os.path.join('artifacts', run_id)
            os.makedirs(local_artifact_path, exist_ok=True)
            client.download_artifacts(run_id, "vectorizer.pkl", local_artifact_path)
            
            logger.info(f"Loaded model from {model_uri}")
            logger.info(f"Downloaded artifacts to {local_artifact_path}")
            
            return model, local_artifact_path
        
        except mlflow.exceptions.RestException as e:
            logger.error(f"Error during loading model and artifacts: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise e

    def predict(self, text):
        model, artifact_path = self.load_latest_model_and_artifacts()
        df = pd.DataFrame({'text': pd.Series(text)})
        df = clean_data(df, 'text')
        cleaned_text = df['text']
        vectorizer = joblib.load(os.path.join(artifact_path, "vectorizer.pkl"))
        cleaned_text = vectorizer.transform(df['text'])
        cleaned_text = cleaned_text.toarray()
        return model.predict(cleaned_text)[0]

def main(txt):
    pipeline = PredictPipeline()
    prediction = pipeline.predict(txt)
    return prediction

if __name__ == "__main__":
    # Example usage: replace 'your_text_here' with the actual text you want to predict
    txt = "your_text_here"
    prediction = main(txt)
    print(f"Prediction: {prediction}")
