import os
from dotenv import load_dotenv
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent.parent

# Load environment variables from .env file
load_dotenv(ROOT_DIR / ".env")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MOVIES_HOST = os.getenv("MOVIES_HOST")
INDEX_NAME = os.getenv("INDEX_NAME")
RECORD_NAMESPACE = os.getenv("RECORD_NAMESPACE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))

# Make sure required environment variables are set
def validate_env_vars():
    required_vars = ["PINECONE_API_KEY", "MOVIES_HOST", "INDEX_NAME", "RECORD_NAMESPACE", "EMBEDDING_MODEL", "EMBEDDING_DIM"]
    missing_vars = [var for var in required_vars if not globals().get(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate environment variables at import time
validate_env_vars()