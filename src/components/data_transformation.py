import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    data_transformation_dir = os.path.join(artifact_folder, 'data_transformation')
    transformed_train_file_path = os.path.join(data_transformation_dir, 'train.npy')
    transformed_test_file_path = os.path.join(data_transformation_dir, 'test.npy')
    transformed_object_file_path = os.path.join(data_transformation_dir, 'preprocessing.pkl')


class DataTransformation:
    def __init__(self,
                 valid_data_dir):

        self.valid_data_dir = valid_data_dir

        self.data_transformation_config = DataTransformationConfig()

        self.utils = MainUtils()

    @staticmethod
    def get_merged_batch_data(valid_data_dir: str) -> pd.DataFrame:
        """
        Method Name :   get_merged_batch_data
        Description :   This method reads all the validated raw data from the valid_data_dir and returns a pandas DataFrame containing the merged data. 
        
        Output      :   a pandas DataFrame containing the merged data 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            raw_files = os.listdir(valid_data_dir)
            csv_data = []
            for filename in raw_files:
                data = pd.read_csv(os.path.join(valid_data_dir, filename))
                csv_data.append(data)

            merged_data = pd.concat(csv_data)


            return merged_data
        except Exception as e:
            raise CustomException(e, sys)

    

    def initiate_data_transformation(self):

        logging.info("Starting data transformation")

        try:
            dataframe = self.get_merged_batch_data(self.valid_data_dir)
            dataframe = self.utils.remove_unwanted_spaces(dataframe)
            dataframe.replace('?', np.nan, inplace=True)

            X = dataframe.drop(columns=TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN].map({-1: 0, 1: 1})

            categorical_features, continuous_features, _ = \
                self.utils.identify_feature_types(X)

            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('num', numeric_pipeline, continuous_features),
                ('cat', categorical_pipeline, categorical_features)
            ])

            sampler = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled,
                test_size=0.2,
                random_state=42
            )

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)

            self.utils.save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            return X_train_transformed, y_train, X_test_transformed, y_test, \
                self.data_transformation_config.transformed_object_file_path

        except Exception as e:
            raise CustomException(e, sys)
