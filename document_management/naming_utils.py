"""
Utility module for handling file naming conventions in the S3 bucket.
Provides functions to generate and parse S3 keys according to the defined convention.
"""
import re
import os
import logging
from typing import Dict, Optional, Tuple, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define file types that can be stored in the system
FILE_TYPES = {
    "pdf": ["original"],
    "json": ["headers", "metadata", "quality_check"],
    "md": ["report"],
    "csv": ["metrics", "results"],
    "txt": ["logs", "notes"]
}

# Regex pattern for parsing S3 keys based on the naming convention
# Format: {fdd_id}_{file_type}_{file_subtype}_v{version}.{extension}
S3_KEY_PATTERN = re.compile(
    r"(?P<fdd_id>[a-f0-9\-]{36})_"  # UUID format (36 chars with hyphens)
    r"(?P<file_type>[a-z]+)_"       # file type (e.g., pdf, json)
    r"(?P<file_subtype>[a-z_]+)_"   # file subtype (may be 'na' if not applicable)
    r"v(?P<version>\d+)"            # version number
    r"\.(?P<extension>[a-z]+)$"     # file extension
)

def generate_s3_key(
    fdd_id: str,
    file_type: str,
    version: int,
    extension: str,
    file_subtype: Optional[str] = None
) -> str:
    """
    Generate an S3 key following the naming convention.
    
    Args:
        fdd_id: UUID of the FDD
        file_type: Type of file (e.g., 'pdf', 'json')
        version: Version number
        extension: File extension without the dot (e.g., 'pdf', 'json')
        file_subtype: Subtype of file (optional, defaults to 'na')
        
    Returns:
        S3 key string following the naming convention
    """
    # Use 'na' (not applicable) as a placeholder when subtype is None
    subtype = file_subtype if file_subtype else "na"
    
    # Generate the key following the convention
    return f"{fdd_id}_{file_type}_{subtype}_v{version}.{extension}"

def parse_s3_key(s3_key: str) -> Optional[Dict[str, Any]]:
    """
    Parse an S3 key to extract metadata according to the naming convention.
    
    Args:
        s3_key: S3 key to parse
        
    Returns:
        Dictionary containing extracted metadata (fdd_id, file_type, file_subtype, version, extension),
        or None if the key doesn't match the convention
    """
    # Extract just the filename from the full path if necessary
    filename = os.path.basename(s3_key)
    
    # Try to match the filename against the pattern
    match = S3_KEY_PATTERN.match(filename)
    if not match:
        logger.warning(f"S3 key doesn't match naming convention: {s3_key}")
        return None
    
    # Extract metadata from the match
    result = match.groupdict()
    
    # Convert version to integer
    result['version'] = int(result['version'])
    
    # Convert 'na' subtype to None for database consistency
    if result['file_subtype'] == 'na':
        result['file_subtype'] = None
    
    return result

def generate_s3_uri(bucket_name: str, s3_key: str) -> str:
    """
    Generate a full S3 URI from a bucket name and key.
    
    Args:
        bucket_name: Name of the S3 bucket
        s3_key: S3 key (path/filename)
        
    Returns:
        Full S3 URI (s3://bucket/path/filename)
    """
    return f"s3://{bucket_name}/{s3_key}"

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parse an S3 URI to extract bucket name and key.
    
    Args:
        s3_uri: Full S3 URI (s3://bucket/path/filename)
        
    Returns:
        Tuple containing (bucket_name, s3_key)
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    # Remove 's3://' prefix
    path = s3_uri[5:]
    
    # Split into bucket and key parts
    parts = path.split("/", 1)
    if len(parts) < 2:
        # No key part (just bucket)
        return parts[0], ""
    
    return parts[0], parts[1]

def get_extension_from_path(path: str) -> str:
    """
    Extract the file extension from a path.
    
    Args:
        path: File path or name
        
    Returns:
        File extension without the dot
    """
    _, ext = os.path.splitext(path)
    if ext.startswith('.'):
        ext = ext[1:]
    return ext

def get_s3_key_for_path_type(
    path_type: str,
    fdd_id: str,
    file_type: str,
    version: int,
    extension: str,
    file_subtype: Optional[str] = None
) -> str:
    """
    Generate a complete S3 key including the appropriate path prefix for a specific path type.
    
    Args:
        path_type: Type of path (e.g., 'pdfs', 'extracted_headers_huridoc')
        fdd_id: UUID of the FDD
        file_type: Type of file (e.g., 'pdf', 'json')
        version: Version number
        extension: File extension without the dot
        file_subtype: Subtype of file (optional)
        
    Returns:
        Complete S3 key including path prefix
    """
    from config import get_s3_path
    
    # Get the base path for the specified path type
    base_path = get_s3_path(path_type)
    if not base_path:
        raise ValueError(f"Unknown path type: {path_type}")
    
    # Generate the filename part
    filename = generate_s3_key(fdd_id, file_type, version, extension, file_subtype)
    
    # Combine path and filename
    return os.path.join(base_path, filename).replace('\\', '/')  # Ensure forward slashes for S3 