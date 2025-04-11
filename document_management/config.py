"""
Configuration module to load AWS credentials and other settings from .env file.
"""
import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

# S3 Configuration
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "fddsearch-fdd-pipeline")

# S3 Path Prefixes
S3_PATHS: Dict[str, str] = {
    "pdfs": "pdfs/",
    "extracted_headers_huridoc": "extracted_headers/original/huridoc_extract/",
    "extracted_headers_llm": "extracted_headers/original/llm_extract/",
    "document_layout_analysis": "document_layout_analysis/",
    "qc_results": "qc_results/"
}

# Database Configuration
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./fdd_metadata.db")

# Upload Processing Configuration
MAX_UPLOAD_ATTEMPTS: int = int(os.getenv("MAX_UPLOAD_ATTEMPTS", "3"))

def get_s3_path(path_type: str) -> Optional[str]:
    """
    Get the S3 path for a specific file type.
    
    Args:
        path_type: Type of file path to retrieve (e.g., 'pdfs', 'extracted_headers_huridoc')
        
    Returns:
        The S3 path prefix as a string, or None if path_type is not recognized
    """
    return S3_PATHS.get(path_type)

def validate_config() -> bool:
    """
    Validate that all required configuration values are set.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    required_vars = [
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_REGION,
        S3_BUCKET_NAME,
        SQLITE_DB_PATH
    ]
    
    return all(required_vars) 