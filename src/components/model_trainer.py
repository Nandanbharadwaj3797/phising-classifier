import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class ModelTrainerConfig:
    model_trainer_dir = os.path.join("artifacts", "model")
    trained_model_path = os.path.join(model_trainer_dir, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')


class VisibilityModel:
    def __init__(self, preprocessing_object: ColumnTransformer, trained_model_object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X: pd.DataFrame):
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            "GaussianNB": GaussianNB(),
            "XGBClassifier": XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss'),
            "LogisticRegression": LogisticRegression(max_iter=1000)
        }


    def evaluate_models(self, X_train, y_train, X_test, y_test):

        report = {}

        for model_name, model in self.models.items():

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report


    def finetune_best_model(self, model_name, model, X_train, y_train):

        model_param_grid = self.utils.read_yaml_file(
            self.model_trainer_config.model_config_file_path
        )["model_selection"]["model"][model_name]["search_param_grid"]

        grid_search = GridSearchCV(
            model,
            param_grid=model_param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print("Best params:", grid_search.best_params_)

        return grid_search.best_estimator_


    def initiate_model_trainer(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        preprocessor_path
    ):

        try:
            logging.info("Loading preprocessor")
            preprocessor = self.utils.load_object(preprocessor_path)

            # 1 Evaluate Base Models
            model_report = self.evaluate_models(
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test
            )

            print("Base Model Scores:", model_report)

            best_model_name = max(model_report, key=model_report.get)
            best_model = self.models[best_model_name]

            logging.info(f"Best base model: {best_model_name}")

            # 2 Finetune Best Model
            best_model = self.finetune_best_model(
                model_name=best_model_name,
                model=best_model,
                X_train=x_train,
                y_train=y_train
            )

            # 3 Final Evaluation
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            final_score = accuracy_score(y_test, y_pred)

            logging.info(f"Final tuned model accuracy: {final_score}")

            if final_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No model found with accuracy greater than "
                    f"{self.model_trainer_config.expected_accuracy}"
                )

            # 4  Save Model Locally
            custom_model = VisibilityModel(
                preprocessing_object=preprocessor,
                trained_model_object=best_model
            )

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path),
                exist_ok=True
            )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=custom_model
            )

            # Optional S3 Upload
            bucket_name = os.getenv("AWS_BUCKET_NAME")

            if bucket_name:
                try:
                    self.utils.upload_file(
                        from_filename=self.model_trainer_config.trained_model_path,
                        to_filename="model.pkl",
                        bucket_name=bucket_name
                    )
                except Exception as e:
                    logging.error(f"S3 upload failed: {str(e)}")

            return final_score

        except Exception as e:
            raise CustomException(e, sys)