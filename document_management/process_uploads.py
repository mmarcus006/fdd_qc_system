"""
Script to process pending uploads from the upload queue.
"""
import os
import logging
import argparse
from typing import Dict, List, Any, Optional

from config import S3_BUCKET_NAME, MAX_UPLOAD_ATTEMPTS
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

def process_upload_task(task: Dict[str, Any]) -> bool:
    """
    Process a single upload task.
    
    Args:
        task: Dictionary containing upload task data
        
    Returns:
        True if processing succeeds, False otherwise
    """
    task_id = task['id']
    fdd_id = task['fdd_id']
    file_type = task['file_type']
    file_subtype = task['file_subtype']
    local_path = task['local_path']
    target_s3_key = task['target_s3_key']
    
    logger.info(f"Processing upload task {task_id}: {local_path} -> {target_s3_key}")
    
    # Update task status to 'uploading'
    if not db_utils.update_upload_status(task_id, 'uploading'):
        logger.error(f"Failed to update task status: {task_id}")
        return False
    
    # Check if local file exists
    if not os.path.exists(local_path):
        error_message = f"Local file does not exist: {local_path}"
        logger.error(error_message)
        db_utils.update_upload_status(task_id, 'failed', error_message)
        return False
    
    # Attempt to upload the file
    if not s3_utils.upload_to_s3(local_path, S3_BUCKET_NAME, target_s3_key):
        error_message = f"Failed to upload file to S3: {local_path} -> {target_s3_key}"
        logger.error(error_message)
        db_utils.update_upload_status(task_id, 'failed', error_message)
        return False
    
    # Construct the storage URI for database
    storage_uri = naming_utils.generate_s3_uri(S3_BUCKET_NAME, target_s3_key)
    
    # Ensure FDD record exists
    if not db_utils.ensure_fdd_exists(fdd_id):
        error_message = f"Failed to ensure FDD exists: {fdd_id}"
        logger.error(error_message)
        db_utils.update_upload_status(task_id, 'failed', error_message)
        return False
    
    # Parse the key to get version
    parsed = naming_utils.parse_s3_key(target_s3_key)
    if not parsed:
        error_message = f"Failed to parse S3 key: {target_s3_key}"
        logger.error(error_message)
        db_utils.update_upload_status(task_id, 'failed', error_message)
        return False
    
    version = parsed['version']
    
    # Register the file in the database
    if not db_utils.insert_fdd_file(
        fdd_id=fdd_id,
        file_type=file_type,
        file_subtype=file_subtype,
        storage_uri=storage_uri,
        version=version
    ):
        error_message = f"Failed to register file in database: {storage_uri}"
        logger.error(error_message)
        db_utils.update_upload_status(task_id, 'failed', error_message)
        return False
    
    # Update task status to 'complete'
    if not db_utils.update_upload_status(task_id, 'complete'):
        logger.error(f"Failed to update task status to complete: {task_id}")
        return False
    
    logger.info(f"Successfully processed upload task {task_id}")
    return True

def process_queue() -> Dict[str, int]:
    """
    Process all pending upload tasks in the queue.
    
    Returns:
        Dictionary with counts of total, successful, and failed tasks
    """
    # Initialize counts
    counts = {
        'total': 0,
        'success': 0,
        'failure': 0
    }
    
    # Get pending uploads
    tasks = db_utils.get_pending_uploads(MAX_UPLOAD_ATTEMPTS)
    counts['total'] = len(tasks)
    
    logger.info(f"Found {len(tasks)} pending upload tasks")
    
    # Process each task
    for task in tasks:
        if process_upload_task(task):
            counts['success'] += 1
        else:
            counts['failure'] += 1
    
    logger.info(f"Upload processing complete. "
                f"Total: {counts['total']}, "
                f"Successful: {counts['success']}, "
                f"Failed: {counts['failure']}")
    
    return counts

def main():
    """Main function to run the upload processing."""
    parser = argparse.ArgumentParser(description="Process pending uploads in the queue")
    parser.add_argument("--max-attempts", type=int, default=MAX_UPLOAD_ATTEMPTS,
                       help="Maximum number of upload attempts before giving up")
    args = parser.parse_args()
    
    # Initialize the database if needed
    if not db_utils.check_output_file_exists():
        logger.info("Initializing database...")
        db_utils.initialize_db()
    
    process_queue()

if __name__ == "__main__":
    main() 