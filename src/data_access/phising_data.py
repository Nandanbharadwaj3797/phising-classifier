import sys
import pandas as pd
from pymongo import MongoClient
from typing import Generator, Tuple

from src.constant import MONGODB_URI
from src.exception import CustomException


class PhisingData:
    """
    This class exports MongoDB collections as pandas DataFrames.
    """

    def __init__(self, database_name: str):
        try:
            if not MONGODB_URI:
                raise ValueError("MONGODB_URI is not set in environment variables")

            self.database_name = database_name
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[self.database_name]

        except Exception as e:
            raise CustomException(e, sys)

    def export_collections_as_dataframe(
        self
    ) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        """
        Exports each collection in the database as a pandas DataFrame.

        Yields:
            (collection_name, dataframe)
        """
        try:
            collections = self.db.list_collection_names()

            print("Collections found:", collections)

            if not collections:
                print("⚠ No collections found in database.")
                return

            for collection_name in collections:
                collection = self.db[collection_name]
                data = list(collection.find())

                print(f"Collection: {collection_name}")
                print(f"Document count: {len(data)}")

                if len(data) > 0:
                    df = pd.DataFrame(data)

                    # Drop MongoDB internal ID
                    if "_id" in df.columns:
                        df.drop(columns=["_id"], inplace=True)

                    yield collection_name, df
                else:
                    print(f"⚠ Collection '{collection_name}' is empty.")

        except Exception as e:
            raise CustomException(e, sys)