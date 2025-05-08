from datetime import datetime
import os
import pickle

import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

print(vec.settings.database.service_url)

# Read the CSV file
df = pd.read_csv("../data/faq_dataset.csv", sep=";")


# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.

    This function creates a record with a UUID version 1 as the ID, which captures
    the current time or a specified time.

    Note:
        - By default, this function uses the current time for the UUID.
        - To use a specific time:
          1. Import the datetime module.
          2. Create a datetime object for your desired time.
          3. Use uuid_from_time(your_datetime) instead of uuid_from_time(datetime.now()).

        Example:
            from datetime import datetime
            specific_time = datetime(2023, 1, 1, 12, 0, 0)
            id = str(uuid_from_time(specific_time))

        This is useful when your content already has an associated datetime.
    """
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "category": row["category"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


# File to save/load records
RECORDS_CACHE_FILE = "../data/records_df.pickle"

# Check if cached records exist
if os.path.exists(RECORDS_CACHE_FILE):
    # Load from pickle file
    print(f"Loading records from cache: {RECORDS_CACHE_FILE}")
    with open(RECORDS_CACHE_FILE, "rb") as f:
        records_df = pickle.load(f)
else:
    # Generate records with embeddings
    print("Generating embeddings for records...")
    records_df = df.apply(prepare_record, axis=1)
    
    # Save to pickle for future use
    print(f"Saving records to cache: {RECORDS_CACHE_FILE}")
    with open(RECORDS_CACHE_FILE, "wb") as f:
        pickle.dump(records_df, f)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)
