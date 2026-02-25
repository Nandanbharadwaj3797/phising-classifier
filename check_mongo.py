from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGODB_URI")
db_name = os.getenv("DATABASE_NAME")

client = MongoClient(uri)

print("All Databases:")
print(client.list_database_names())

db = client[db_name]

print("\nCollections inside:", db_name)
print(db.list_collection_names())

for col in db.list_collection_names():
    print("Collection:", col)
    print("Document count:", db[col].count_documents({}))