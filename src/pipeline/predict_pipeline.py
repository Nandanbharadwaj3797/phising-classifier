import os
import sys
import pandas as pd
from dataclasses import dataclass
from flask import Request
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException
from src.constant import TARGET_COLUMN
from src.utils.main_utils import MainUtils

load_dotenv()


@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "predicted_file.csv"
    prediction_file_path: str = os.path.join(
        prediction_output_dirname,
        prediction_file_name
    )


class PredictionPipeline:

    def __init__(self, request: Request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_file_detail = PredictionFileDetail()

    # Save Uploaded File
    def save_input_files(self) -> str:

        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(
                pred_file_input_dir,
                input_csv_file.filename
            )

            input_csv_file.save(pred_file_path)

            return pred_file_path

        except Exception as e:
            raise CustomException(e, sys)

    # Load Model Locally

    def predict(self, features: pd.DataFrame):

        try:
            local_model_path = os.path.join("artifacts", "model", "model.pkl")
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

            bucket_name = os.getenv("AWS_BUCKET_NAME")

            #  Try downloading from S3
            try:
                model_path = self.utils.download_model(
                    bucket_name=bucket_name,
                    bucket_file_name="model.pkl",
                    dest_file_name=local_model_path
                )
                print("Model downloaded from S3.")

            except Exception:
                print("S3 download failed. Using local model if available.")
                model_path = local_model_path

            # Ensure model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model not found locally or in S3.")

            model = self.utils.load_object(model_path)

            predictions = model.predict(features)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


    # Generate Prediction File

    def get_predicted_dataframe(self, input_dataframe_path: str):

        try:
            prediction_column_name = TARGET_COLUMN

            input_dataframe = pd.read_csv(input_dataframe_path)

            predictions = self.predict(input_dataframe)

            input_dataframe[prediction_column_name] = predictions

            #  Correct mapping for phishing dataset
            target_column_mapping = {
                -1: "phising",
                1: "safe"
            }

            input_dataframe[prediction_column_name] = \
                input_dataframe[prediction_column_name].map(
                    target_column_mapping
                )

            os.makedirs(
                self.prediction_file_detail.prediction_output_dirname,
                exist_ok=True
            )

            input_dataframe.to_csv(
                self.prediction_file_detail.prediction_file_path,
                index=False
            )

            logging.info("Predictions completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):

        try:
            input_csv_path = self.save_input_files()

            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_file_detail

        except Exception as e:
            raise CustomException(e, sys)