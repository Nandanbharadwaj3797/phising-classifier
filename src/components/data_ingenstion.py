import sys
import os
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from src.constant import MONGO_DATABASE_NAME, artifact_folder
from src.exception import CustomException
from src.logger import logging
from src.data_access.phising_data import PhisingData


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(artifact_folder, "data_ingestion")


class DataIngestion:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def export_data_into_raw_data_dir(self) -> None:
        """
        Reads data from MongoDB and saves it as CSV inside artifacts folder.
        """
        try:
            logging.info("Exporting data from MongoDB")

            raw_batch_files_path = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(raw_batch_files_path, exist_ok=True)

            phising_data = PhisingData(
                database_name=MONGO_DATABASE_NAME
            )

            logging.info(f"Saving exported data into: {raw_batch_files_path}")

            exported_any = False

            for collection_name, dataset in phising_data.export_collections_as_dataframe():

                if dataset.empty:
                    logging.warning(f"Collection {collection_name} is empty. Skipping.")
                    continue

                logging.info(f"Shape of {collection_name}: {dataset.shape}")

                file_path = os.path.join(
                    raw_batch_files_path,
                    f"{collection_name}.csv"
                )

                dataset.to_csv(file_path, index=False)

                print(f"Saved: {file_path}")
                exported_any = True

            if not exported_any:
                raise ValueError("No data was exported from MongoDB.")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> Path:
        """
        Initiates the data ingestion process.
        """
        try:
            logging.info("Entered initiate_data_ingestion")

            self.export_data_into_raw_data_dir()

            logging.info("Data ingestion completed successfully")

            return Path(self.data_ingestion_config.data_ingestion_dir)

        except Exception as e:
            raise CustomException(e, sys)