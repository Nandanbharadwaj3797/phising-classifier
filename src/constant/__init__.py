from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

AWS_S3_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

MONGO_DATABASE_NAME = os.getenv("DATABASE_NAME")
MONGODB_URI = os.getenv("MONGODB_URI")

TARGET_COLUMN = "Result"

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder_name = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
artifact_folder = os.path.join("artifacts", artifact_folder_name)