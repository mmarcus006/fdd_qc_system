"""
Main script demonstrating usage of the FDD management system.
"""
import os
import uuid
import logging
import argparse
from typing import Dict, List, Any, Optional

from config import S3_BUCKET_NAME, validate_config
import s3_utils
import naming_utils
import db_utils
import discover_s3
import process_uploads

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

def initialize_system():
    """Initialize the system by validating config and setting up the database."""
    # Validate configuration
    if not validate_config():
        logger.error("Invalid configuration. Please check your .env file.")
        return False
    
    # Initialize database if it doesn't exist
    if not db_utils.check_output_file_exists():
        logger.info("Initializing database...")
        db_utils.initialize_db()
        return True
    
    logger.info("System already initialized.")
    return True

def example_create_fdd(franchise_name: str, doc_year: int) -> Optional[str]:
    """
    Example: Create a new FDD record with real metadata.
    
    Args:
        franchise_name: Name of the franchise
        doc_year: Year of the document
        
    Returns:
        FDD ID if creation succeeds, None otherwise
    """
    # Generate a new UUID for the FDD
    fdd_id = str(uuid.uuid4())
    
    # Create the FDD record
    if db_utils.create_fdd(fdd_id, franchise_name, doc_year):
        logger.info(f"Created new FDD record: {fdd_id} - {franchise_name} ({doc_year})")
        return fdd_id
    
    logger.error(f"Failed to create FDD record: {franchise_name} ({doc_year})")
    return None

def example_queue_upload(
    fdd_id: str, 
    local_path: str, 
    file_type: str,
    file_subtype: Optional[str] = None,
    version: int = 1
) -> Optional[int]:
    """
    Example: Add a file to the upload queue.
    
    Args:
        fdd_id: UUID of the FDD
        local_path: Path to the local file
        file_type: Type of file (e.g., 'pdf', 'json')
        file_subtype: Subtype of file (optional)
        version: Version number (default: 1)
        
    Returns:
        Queue item ID if successful, None otherwise
    """
    # Check if local file exists
    if not os.path.exists(local_path):
        logger.error(f"Local file does not exist: {local_path}")
        return None
    
    # Get file extension
    extension = naming_utils.get_extension_from_path(local_path)
    
    # Generate S3 key
    if file_type == 'pdf':
        path_type = 'pdfs'
    elif file_type == 'json' and file_subtype == 'headers':
        path_type = 'extracted_headers_llm'  # Example: choose LLM extraction
    elif file_type == 'json' and file_subtype == 'quality_check':
        path_type = 'qc_results'
    else:
        # Default to pdfs for other file types
        path_type = 'pdfs'
    
    # Generate the target S3 key with appropriate path
    target_s3_key = naming_utils.get_s3_key_for_path_type(
        path_type, fdd_id, file_type, version, extension, file_subtype
    )
    
    # Add to upload queue
    queue_id = db_utils.add_to_upload_queue(
        fdd_id=fdd_id,
        file_type=file_type,
        file_subtype=file_subtype,
        local_path=local_path,
        target_s3_key=target_s3_key
    )
    
    if queue_id:
        logger.info(f"Added file to upload queue: {local_path} -> {target_s3_key}")
        return queue_id
    
    logger.error(f"Failed to add file to upload queue: {local_path}")
    return None

def example_query_files(fdd_id: str, latest_only: bool = True) -> List[Dict[str, Any]]:
    """
    Example: Query files for a specific FDD.
    
    Args:
        fdd_id: UUID of the FDD
        latest_only: If True, return only the latest version of each file
        
    Returns:
        List of file records
    """
    if latest_only:
        logger.info(f"Querying latest files for FDD: {fdd_id}")
        files = db_utils.get_latest_files_by_fdd_id(fdd_id)
    else:
        logger.info(f"Querying all files for FDD: {fdd_id}")
        files = db_utils.get_fdd_files_by_fdd_id(fdd_id)
    
    logger.info(f"Found {len(files)} files for FDD: {fdd_id}")
    return files

def example_get_presigned_url(fdd_id: str, file_type: str, file_subtype: Optional[str] = None) -> Optional[str]:
    """
    Example: Get a presigned URL for the latest version of a file.
    
    Args:
        fdd_id: UUID of the FDD
        file_type: Type of file
        file_subtype: Subtype of file (optional)
        
    Returns:
        Presigned URL if successful, None otherwise
    """
    # Get the latest file record
    files = db_utils.get_latest_files_by_fdd_id(fdd_id)
    
    # Find the matching file
    matching_file = None
    for file in files:
        if file['file_type'] == file_type:
            if (file_subtype is None and file['file_subtype'] is None) or file['file_subtype'] == file_subtype:
                matching_file = file
                break
    
    if not matching_file:
        logger.error(f"No matching file found for FDD {fdd_id}, type {file_type}, subtype {file_subtype}")
        return None
    
    # Parse the S3 URI to get bucket and key
    storage_uri = matching_file['storage_uri']
    bucket, s3_key = naming_utils.parse_s3_uri(storage_uri)
    
    # Generate presigned URL
    url = s3_utils.generate_presigned_url(bucket, s3_key)
    if url:
        logger.info(f"Generated presigned URL for {storage_uri}")
        return url
    
    logger.error(f"Failed to generate presigned URL for {storage_uri}")
    return None

def example_run_discovery():
    """Example: Run the S3 discovery process."""
    logger.info("Running S3 discovery process...")
    results = discover_s3.discover_all(S3_BUCKET_NAME)
    
    for path_type, counts in results.items():
        logger.info(f"Discovery results for {path_type}: "
                   f"Total: {counts['total']}, "
                   f"Processed: {counts['processed']}, "
                   f"Failed: {counts['failed']}")

def example_process_uploads():
    """Example: Process pending uploads."""
    logger.info("Processing pending uploads...")
    counts = process_uploads.process_queue()
    
    logger.info(f"Upload processing results: "
               f"Total: {counts['total']}, "
               f"Successful: {counts['success']}, "
               f"Failed: {counts['failure']}")

def main():
    """Main function demonstrating usage examples."""
    parser = argparse.ArgumentParser(description="FDD Management System Examples")
    parser.add_argument("--init", action="store_true", help="Initialize the system")
    parser.add_argument("--create-fdd", action="store_true", help="Create a new FDD record")
    parser.add_argument("--franchise-name", help="Franchise name for new FDD")
    parser.add_argument("--doc-year", type=int, help="Document year for new FDD")
    parser.add_argument("--queue-upload", action="store_true", help="Queue a file for upload")
    parser.add_argument("--fdd-id", help="FDD ID for operations")
    parser.add_argument("--local-path", help="Local file path for upload")
    parser.add_argument("--file-type", help="File type (e.g., 'pdf', 'json')")
    parser.add_argument("--file-subtype", help="File subtype")
    parser.add_argument("--version", type=int, default=1, help="File version")
    parser.add_argument("--query-files", action="store_true", help="Query files for a specific FDD")
    parser.add_argument("--all-versions", action="store_true", help="Include all versions in query")
    parser.add_argument("--get-url", action="store_true", help="Get a presigned URL for a file")
    parser.add_argument("--run-discovery", action="store_true", help="Run the S3 discovery process")
    parser.add_argument("--process-uploads", action="store_true", help="Process pending uploads")
    
    args = parser.parse_args()
    
    # Initialize if requested
    if args.init:
        initialize_system()
    
    # Create FDD if requested
    if args.create_fdd:
        if not args.franchise_name or not args.doc_year:
            logger.error("Franchise name and document year are required to create an FDD")
        else:
            fdd_id = example_create_fdd(args.franchise_name, args.doc_year)
            if fdd_id:
                print(f"Created FDD: {fdd_id}")
    
    # Queue upload if requested
    if args.queue_upload:
        if not args.fdd_id or not args.local_path or not args.file_type:
            logger.error("FDD ID, local path, and file type are required to queue an upload")
        else:
            queue_id = example_queue_upload(
                args.fdd_id, 
                args.local_path, 
                args.file_type,
                args.file_subtype,
                args.version
            )
            if queue_id:
                print(f"Queued upload: {queue_id}")
    
    # Query files if requested
    if args.query_files:
        if not args.fdd_id:
            logger.error("FDD ID is required to query files")
        else:
            files = example_query_files(args.fdd_id, not args.all_versions)
            print(f"Found {len(files)} files:")
            for file in files:
                print(f"  {file['file_type']}/{file['file_subtype']} v{file['version']}: {file['storage_uri']}")
    
    # Get presigned URL if requested
    if args.get_url:
        if not args.fdd_id or not args.file_type:
            logger.error("FDD ID and file type are required to get a presigned URL")
        else:
            url = example_get_presigned_url(args.fdd_id, args.file_type, args.file_subtype)
            if url:
                print(f"Presigned URL: {url}")
    
    # Run discovery if requested
    if args.run_discovery:
        example_run_discovery()
    
    # Process uploads if requested
    if args.process_uploads:
        example_process_uploads()
    
    # If no actions specified, show help
    if not any([
        args.init, args.create_fdd, args.queue_upload, 
        args.query_files, args.get_url, 
        args.run_discovery, args.process_uploads
    ]):
        parser.print_help()

if __name__ == "__main__":
    main() 