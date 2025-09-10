import os
from pinecone import Pinecone

# Initialize Pinecone using environment variables
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# Test: list indexes
try:
    indexes = pc.list_indexes()
    print("Pinecone indexes:", indexes)
except Exception as e:
    print("Error:", e)
