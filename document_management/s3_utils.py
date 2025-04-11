"""
Utility module for interacting with AWS S3.
Provides functions for listing, uploading, and checking objects in S3.
"""
import os
import logging
import boto3
from botocore.exceptions import ClientError
from typing import List, Dict, Optional, Any, Union

from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_s3_client():
    """
    Initialize and return a boto3 S3 client using credentials from config.
    
    Returns:
        boto3 S3 client
    """
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

def list_s3_objects(bucket: str, prefix: str) -> List[Dict[str, Any]]:
    """
    List objects within a specific S3 prefix.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (path)
        
    Returns:
        List of dictionaries containing object metadata
    """
    client = get_s3_client()
    objects = []
    paginator = client.get_paginator('list_objects_v2')
    
    try:
        # Handle pagination for large buckets
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'etag': obj['ETag']
                    })
        
        logger.info(f"Listed {len(objects)} objects from s3://{bucket}/{prefix}")
        return objects
    except ClientError as e:
        logger.error(f"Error listing S3 objects in s3://{bucket}/{prefix}: {e}")
        return []

def upload_to_s3(local_path: str, bucket: str, s3_key: str) -> bool:
    """
    Upload a file from a local path to S3.
    
    Args:
        local_path: Path to local file
        bucket: S3 bucket name
        s3_key: S3 key (path/filename)
        
    Returns:
        True if upload succeeds, False otherwise
    """
    client = get_s3_client()
    
    try:
        # Check if local file exists
        if not os.path.exists(local_path):
            logger.error(f"Local file does not exist: {local_path}")
            return False
        
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        client.upload_file(local_path, bucket, s3_key)
        logger.info(f"Successfully uploaded file to s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False

def s3_object_exists(bucket: str, s3_key: str) -> bool:
    """
    Check if an S3 object exists.
    
    Args:
        bucket: S3 bucket name
        s3_key: S3 key (path/filename)
        
    Returns:
        True if object exists, False otherwise
    """
    client = get_s3_client()
    
    try:
        client.head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError as e:
        # If error code is 404, object does not exist
        if e.response['Error']['Code'] == '404':
            return False
        # For other errors, log and return False
        logger.error(f"Error checking if S3 object exists: {e}")
        return False

def generate_presigned_url(bucket: str, s3_key: str, expiration: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for downloading an S3 object.
    
    Args:
        bucket: S3 bucket name
        s3_key: S3 key (path/filename)
        expiration: URL expiration time in seconds (default: 1 hour)
        
    Returns:
        Presigned URL as string, or None if generation fails
    """
    client = get_s3_client()
    
    try:
        # Check if object exists first
        if not s3_object_exists(bucket, s3_key):
            logger.warning(f"Object does not exist: s3://{bucket}/{s3_key}")
            return None
        
        url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': s3_key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None

def download_from_s3(bucket: str, s3_key: str, local_path: str) -> bool:
    """
    Download an S3 object to a local file.
    
    Args:
        bucket: S3 bucket name
        s3_key: S3 key (path/filename)
        local_path: Local path to save the file
        
    Returns:
        True if download succeeds, False otherwise
    """
    client = get_s3_client()
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
        client.download_file(bucket, s3_key, local_path)
        logger.info(f"Successfully downloaded file to {local_path}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading file from S3: {e}")
        return False 