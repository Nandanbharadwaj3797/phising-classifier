import pathlib
import sys
import numpy as np
from src.components.data_ingenstion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion()
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, raw_data_dir):
        try:
            logging.info("Starting Data Validation")
            data_validation = DataValidation(raw_data_store_dir=raw_data_dir)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, valid_data_dir):
        try:
            logging.info("Starting Data Transformation")
            data_transformation = DataTransformation(valid_data_dir=valid_data_dir)
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self,
                             x_train: np.array,
                             y_train: np.array,
                             x_test: np.array,
                             y_test: np.array,
                             preprocessor_path: pathlib.Path):
        try:
            logging.info("Starting Model Training")
            model_trainer = ModelTrainer()
            return model_trainer.initiate_model_trainer(
                x_train,
                y_train,
                x_test,
                y_test,
                preprocessor_path
            )
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            raw_data_dir = self.start_data_ingestion()
            valid_data_dir = self.start_data_validation(raw_data_dir)
            x_train, y_train, x_test, y_test, preprocessor_path = \
                self.start_data_transformation(valid_data_dir)

            model_accuracy = self.start_model_training(
                x_train, y_train, x_test, y_test, preprocessor_path
            )

            logging.info(f"Training completed successfully. Accuracy: {model_accuracy}")
            print("Training completed. Model accuracy:", model_accuracy)

        except Exception as e:
            raise CustomException(e, sys)
