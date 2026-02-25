import sys
from typing import Dict, Tuple
import os
import numpy as np
import pandas as pd
import pickle
import yaml
import boto3

from src.constant import *
from src.exception import CustomException
from src.logger import logging


class MainUtils:
    def __init__(self) -> None:
        pass

    # ==============================
    # YAML FILE READING
    # ==============================

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
            raise CustomException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(
                os.path.join("config", "schema.yaml")
            )
            return schema_config
        except Exception as e:
            raise CustomException(e, sys) from e

    # ==============================
    # SAVE OBJECT
    # ==============================

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info("Entered the save_object method of MainUtils class")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

            logging.info("Exited the save_object method of MainUtils class")
        except Exception as e:
            raise CustomException(e, sys) from e

    # ==============================
    # LOAD OBJECT
    # ==============================

    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("Entered the load_object method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)

            logging.info("Exited the load_object method of MainUtils class")
            return obj

        except Exception as e:
            raise CustomException(e, sys) from e

    # ==============================
    # S3 UPLOAD (FIXED VERSION)
    # ==============================

    @staticmethod
    def upload_file(from_filename: str, to_filename: str, bucket_name: str):
        try:
            if not bucket_name:
                raise ValueError("AWS_BUCKET_NAME is not set.")

            if not os.path.exists(from_filename):
                raise FileNotFoundError(f"{from_filename} not found.")

            region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

            s3_client = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )

            s3_client.upload_file(
                Filename=from_filename,
                Bucket=bucket_name,
                Key=to_filename
            )

            logging.info(
                f"File uploaded successfully to S3: s3://{bucket_name}/{to_filename}"
            )

        except Exception as e:
            logging.error(f"S3 upload failed: {str(e)}")
            raise CustomException(e, sys)

    # ==============================
    # S3 DOWNLOAD (FIXED VERSION)
    # ==============================

    @staticmethod
    def download_model(bucket_name: str, bucket_file_name: str, dest_file_name: str):
        try:
            if not bucket_name:
                raise ValueError("AWS_BUCKET_NAME is not set.")

            os.makedirs(os.path.dirname(dest_file_name), exist_ok=True)

            region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

            s3_client = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )

            s3_client.download_file(
                Bucket=bucket_name,
                Key=bucket_file_name,
                Filename=dest_file_name
            )

            logging.info(
                f"Model downloaded successfully from S3: s3://{bucket_name}/{bucket_file_name}"
            )

            return dest_file_name

        except Exception as e:
            logging.error(f"S3 download failed: {str(e)}")
            raise CustomException(e, sys)

    # ==============================
    # REMOVE UNWANTED SPACES
    # ==============================

    @staticmethod
    def remove_unwanted_spaces(data: pd.DataFrame) -> pd.DataFrame:
        try:
            df_without_spaces = data.apply(
                lambda x: x.str.strip() if x.dtype == "object" else x
            )

            logging.info(
                "Unwanted spaces removal successful. Exited remove_unwanted_spaces."
            )

            return df_without_spaces

        except Exception as e:
            raise CustomException(e, sys)

    # ==============================
    # IDENTIFY FEATURE TYPES
    # ==============================

    @staticmethod
    def identify_feature_types(dataframe: pd.DataFrame):
        data_types = dataframe.dtypes

        categorical_features = []
        continuous_features = []
        discrete_features = []

        for column, dtype in dict(data_types).items():
            unique_values = dataframe[column].nunique()

            if dtype == "object" or unique_values < 10:
                categorical_features.append(column)

            elif dtype in [np.int64, np.float64]:
                if unique_values > 20:
                    continuous_features.append(column)
                else:
                    discrete_features.append(column)

        return categorical_features, continuous_features, discrete_features