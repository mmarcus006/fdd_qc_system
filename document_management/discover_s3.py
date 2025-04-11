"""
Script to discover existing files in S3 and register them in the database.
"""
import os
import logging
import argparse
from typing import List, Dict, Any, Optional

from config import S3_BUCKET_NAME, S3_PATHS
import s3_utils
import naming_utils
import db_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_output_file_exists() -> bool:
    """
    Check if the SQLite database file exists.
    
    Returns:
        True if the file exists, False otherwise
    """
    return db_utils.check_output_file_exists()

def process_s3_object(bucket: str, s3_key: str) -> bool:
    """
    Process a single S3 object, attempting to parse its key and register it in the database.
    
    Args:
        bucket: S3 bucket name
        s3_key: S3 key to process
        
    Returns:
        True if processing succeeds, False otherwise
    """
    # Generate the full S3 URI
    storage_uri = naming_utils.generate_s3_uri(bucket, s3_key)
    
    # Check if this URI is already registered
    if db_utils.storage_uri_exists(storage_uri):
        logger.debug(f"Object already registered: {storage_uri}")
        return True
    
    # Attempt to parse the key
    parsed = naming_utils.parse_s3_key(s3_key)
    if not parsed:
        logger.warning(f"Unable to parse S3 key: {s3_key}")
        return False
    
    # Extract metadata
    fdd_id = parsed['fdd_id']
    file_type = parsed['file_type']
    file_subtype = parsed['file_subtype']  # May be None
    version = parsed['version']
    
    # Ensure the FDD record exists
    if not db_utils.ensure_fdd_exists(fdd_id):
        logger.error(f"Failed to ensure FDD exists for ID: {fdd_id}")
        return False
    
    # Register the file
    success = db_utils.insert_fdd_file(
        fdd_id=fdd_id,
        file_type=file_type,
        file_subtype=file_subtype,
        storage_uri=storage_uri,
        version=version
    )
    
    if success:
        logger.info(f"Registered new file: {storage_uri}")
    else:
        logger.warning(f"Failed to register file: {storage_uri}")
    
    return success

def discover_in_prefix(bucket: str, prefix: str) -> Dict[str, int]:
    """
    Discover and process all objects within a specific S3 prefix.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix to scan
        
    Returns:
        Dictionary with counts of total, processed, and skipped objects
    """
    logger.info(f"Discovering objects in s3://{bucket}/{prefix}")
    
    # List objects in the prefix
    objects = s3_utils.list_s3_objects(bucket, prefix)
    
    # Initialize counts
    counts = {
        'total': len(objects),
        'processed': 0,
        'skipped': 0,
        'failed': 0
    }
    
    # Process each object
    for obj in objects:
        s3_key = obj['key']
        if process_s3_object(bucket, s3_key):
            counts['processed'] += 1
        else:
            counts['failed'] += 1
    
    logger.info(f"Discovery in {prefix} complete. "
                f"Total: {counts['total']}, "
                f"Processed: {counts['processed']}, "
                f"Failed: {counts['failed']}")
    
    return counts

def discover_all(bucket: str) -> Dict[str, Dict[str, int]]:
    """
    Discover and process objects in all defined S3 prefixes.
    
    Args:
        bucket: S3 bucket name
        
    Returns:
        Dictionary with counts for each prefix
    """
    results = {}
    
    # Initialize the database if needed
    if not db_utils.check_output_file_exists():
        logger.info("Initializing database...")
        db_utils.initialize_db()
    
    # Process each prefix
    for path_type, prefix in S3_PATHS.items():
        results[path_type] = discover_in_prefix(bucket, prefix)
    
    return results

def main():
    """Main function to run the S3 discovery process."""
    parser = argparse.ArgumentParser(description="Discover and register S3 objects in the database")
    parser.add_argument("--prefix", help="Specific S3 prefix to scan", default=None)
    parser.add_argument("--bucket", help="S3 bucket name", default=S3_BUCKET_NAME)
    args = parser.parse_args()
    
    bucket = args.bucket
    
    if args.prefix:
        # Discover in a specific prefix
        if args.prefix in S3_PATHS:
            # If prefix is a key in S3_PATHS, use the corresponding value
            prefix = S3_PATHS[args.prefix]
        else:
            # Otherwise, use the input directly
            prefix = args.prefix
        
        discover_in_prefix(bucket, prefix)
    else:
        # Discover in all prefixes
        discover_all(bucket)

if __name__ == "__main__":
    main() 