import os
import joblib
import boto3
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from project import logger
from project.utils.common import clean_data, read_yaml_file, read_data_from_s3


class TrainingPipeline:
    def __init__(self):
        self.config = read_yaml_file("config/config.yaml")
        self.s3 = boto3.client("s3")
        self._update_experiment_name()
        self._load_config()

        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)

        self.data = self._load_data()
        self.vectorizer = self._fit_vectorizer(self.data['Sentiment'])
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()

        self.models = {
            'naive_bayes': MultinomialNB(),
            'support_vector_classifier': SVC()
        }

    def _update_experiment_name(self):
        experiment_name_parts = self.config.mlflow.experiment_name.split("_")
        experiment_name_parts[1] = str(int(experiment_name_parts[1]) + 1)
        self.config.mlflow.experiment_name = "_".join(experiment_name_parts)
        self.config.to_yaml("config/config.yaml")

    def _load_config(self):
        self.config = read_yaml_file("config/config.yaml")

    def _load_data(self):
        return read_data_from_s3(self.config.s3.bucket, self.config.s3.save_object)

    def _fit_vectorizer(self, data):
        vectorizer = TfidfVectorizer()
        vectorizer.fit(data)
        joblib.dump(vectorizer, 'vectorizer.pkl')
        return vectorizer

    def _prepare_data(self):
        tfidf_matrix = self.vectorizer.transform(self.data['Sentiment'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
        return train_test_split(tfidf_df, self.data['Sentiment'], test_size=0.3)

    def train_and_log_model(self, model, model_name):
        with mlflow.start_run(run_name=model_name):
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_artifact("vectorizer.pkl")
            mlflow.sklearn.log_model(model, "model")

    def train(self):
        logger.info(">>>>>>>>>> Experiment started >>>>>>>>>>")
        for model_name, model in self.models.items():
            try:
                self.train_and_log_model(model, model_name)
            except Exception as e:
                logger.error(f"Something went wrong with model {model_name}: {e}")
        logger.info(">>>>>>>>>>> Experiment completed >>>>>>>>>>")

    def register_model(self):
        client = mlflow.tracking.MlflowClient()
        try:
            experiment = client.get_experiment_by_name(self.config.mlflow.experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment {self.config.mlflow.experiment_name} not found.")

            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs:
                raise ValueError(f"No runs found for experiment {self.config.mlflow.experiment_name}.")

            best_run = max(runs, key=lambda run: run.data.metrics.get('accuracy', 0))
            best_run_id = best_run.info.run_id
            latest_accuracy = best_run.data.metrics.get('accuracy', 0)
            best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

            try:
                latest_versions = client.get_latest_versions(self.config.mlflow.model_name, stages=["None"])
            except mlflow.exceptions.RestException as e:
                if e.error_code == 'RESOURCE_DOES_NOT_EXIST':
                    latest_versions = []
                else:
                    raise e

            if not latest_versions:
                mlflow.register_model(f"runs:/{best_run_id}/model", self.config.mlflow.model_name)
                logger.info("Model has been registered.")
            else:
                latest_version = latest_versions[-1]
                run_id = latest_version.run_id
                run = client.get_run(run_id)
                last_accuracy = run.data.metrics.get("accuracy", 0)

                if latest_accuracy > last_accuracy:
                    mlflow.register_model(f"runs:/{best_run_id}/model", self.config.mlflow.model_name)
                    logger.info("Latest version of the model has been registered.")
                else:
                    logger.info("Last version accuracy is greater than or equal to the latest version.")

        except Exception as e:
            logger.error(f"Error during model registration: {e}")
            raise e


def main():
    processor = TrainingPipeline()
    logger.info(">>>>>>>>>> Training pipeline started <<<<<<<<<<<<<<<")
    processor.train()
    logger.info(">>>>>>>>>> Training completed <<<<<<<<<<<<<<<")
    processor.register_model()
    logger.info(">>>>>>>>>> Model registry process: Complete <<<<<<<<<<<<<<<")


if __name__ == '__main__':
    main()
