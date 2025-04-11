"""
Database utility module for interacting with SQLite database.
Provides functions for CRUD operations on FDD-related tables.
"""
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import os

from config import SQLITE_DB_PATH

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
    return os.path.exists(SQLITE_DB_PATH)

def get_db_connection() -> sqlite3.Connection:
    """
    Get a connection to the SQLite database.
    Ensures foreign key constraints are enabled.
    
    Returns:
        A SQLite connection object
    """
    # Create directory for database if it doesn't exist
    db_dir = os.path.dirname(SQLITE_DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
    
    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON;")
    
    return conn

def initialize_db() -> None:
    """
    Initialize the database by creating tables if they don't exist.
    """
    conn = get_db_connection()
    try:
        # FDD table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS fdd (
            id TEXT PRIMARY KEY,
            franchise_name TEXT NOT NULL,
            doc_year INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # FDD files table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS fdd_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fdd_id TEXT NOT NULL REFERENCES fdd(id) ON DELETE CASCADE,
            file_type TEXT NOT NULL,
            file_subtype TEXT,
            storage_uri TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            uploaded_by TEXT,
            environment TEXT DEFAULT 'dev',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(fdd_id, file_type, file_subtype, version),
            UNIQUE(storage_uri)
        );
        """)
        
        # Upload queue table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS upload_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fdd_id TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_subtype TEXT,
            local_path TEXT NOT NULL,
            target_s3_key TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            last_attempt TIMESTAMP,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create triggers
        conn.execute("""
        CREATE TRIGGER IF NOT EXISTS update_fdd_files_updated_at
        AFTER UPDATE ON fdd_files
        FOR EACH ROW
        BEGIN
            UPDATE fdd_files SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
        END;
        """)
        
        conn.execute("""
        CREATE TRIGGER IF NOT EXISTS update_fdd_updated_at
        AFTER UPDATE ON fdd
        FOR EACH ROW
        BEGIN
            UPDATE fdd SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
        END;
        """)
        
        conn.execute("""
        CREATE TRIGGER IF NOT EXISTS update_upload_queue_updated_at
        AFTER UPDATE ON upload_queue
        FOR EACH ROW
        BEGIN
            UPDATE upload_queue SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
        END;
        """)
        
        # Create view for latest files
        conn.execute("""
        CREATE VIEW IF NOT EXISTS fdd_latest_files AS
        SELECT
            f.*
        FROM
            fdd_files f
        INNER JOIN (
            SELECT
                fdd_id,
                file_type,
                file_subtype,
                MAX(version) AS max_version
            FROM
                fdd_files
            GROUP BY
                fdd_id, file_type, file_subtype
        ) latest ON f.fdd_id = latest.fdd_id
                AND f.file_type = latest.file_type
                AND (f.file_subtype = latest.file_subtype OR (f.file_subtype IS NULL AND latest.file_subtype IS NULL))
                AND f.version = latest.max_version;
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

# FDD Table Operations
def create_fdd(fdd_id: str, franchise_name: str, doc_year: int) -> bool:
    """
    Create a new FDD record in the database.
    
    Args:
        fdd_id: UUID of the FDD as TEXT
        franchise_name: Name of the franchise
        doc_year: Year of the document
        
    Returns:
        True if successful, False if an error occurs
    """
    conn = get_db_connection()
    try:
        # Use INSERT OR IGNORE to handle duplicate IDs gracefully
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO fdd (id, franchise_name, doc_year)
            VALUES (?, ?, ?)
            """,
            (fdd_id, franchise_name, doc_year)
        )
        conn.commit()
        # Return True if a row was inserted, False if it was ignored (already exists)
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Error creating FDD record: {e}")
        return False
    finally:
        conn.close()

def get_fdd(fdd_id: str) -> Optional[Dict[str, Any]]:
    """
    Get an FDD record by ID.
    
    Args:
        fdd_id: UUID of the FDD as TEXT
        
    Returns:
        Dictionary with FDD data or None if not found
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM fdd WHERE id = ?",
            (fdd_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    except sqlite3.Error as e:
        logger.error(f"Error retrieving FDD: {e}")
        return None
    finally:
        conn.close()

def fdd_exists(fdd_id: str) -> bool:
    """
    Check if an FDD record exists.
    
    Args:
        fdd_id: UUID of the FDD as TEXT
        
    Returns:
        True if exists, False otherwise
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT 1 FROM fdd WHERE id = ?",
            (fdd_id,)
        )
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Error checking FDD existence: {e}")
        return False
    finally:
        conn.close()

def ensure_fdd_exists(fdd_id: str) -> bool:
    """
    Ensure an FDD record exists, creating it with placeholder values if not.
    
    Args:
        fdd_id: UUID of the FDD as TEXT
        
    Returns:
        True if exists or was created, False if creation failed
    """
    if fdd_exists(fdd_id):
        return True
    
    # Create with placeholder values
    # TODO: Implement a more robust solution for obtaining franchise_name and doc_year
    return create_fdd(fdd_id, "Unknown", -1)

# FDD Files Table Operations
def insert_fdd_file(
    fdd_id: str,
    file_type: str,
    storage_uri: str,
    file_subtype: Optional[str] = None,
    version: int = 1,
    uploaded_by: Optional[str] = None,
    environment: str = "dev",
    notes: Optional[str] = None
) -> bool:
    """
    Insert a new FDD file record.
    
    Args:
        fdd_id: UUID of the FDD as TEXT
        file_type: Type of file (e.g., 'pdf', 'json')
        storage_uri: Full S3 URI (e.g., 's3://bucket/path/key')
        file_subtype: Subtype of the file (optional)
        version: Version number (default: 1)
        uploaded_by: User who uploaded the file (optional)
        environment: Environment (default: 'dev')
        notes: Additional notes (optional)
        
    Returns:
        True if successful, False otherwise
    """
    conn = get_db_connection()
    try:
        # Ensure FDD exists before attempting to insert a file record
        if not ensure_fdd_exists(fdd_id):
            logger.error(f"Failed to ensure FDD exists: {fdd_id}")
            return False
        
        # Use INSERT OR IGNORE to handle unique constraint violations
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO fdd_files 
            (fdd_id, file_type, file_subtype, storage_uri, version, uploaded_by, environment, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fdd_id, file_type, file_subtype, storage_uri, version, uploaded_by, environment, notes)
        )
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Error inserting FDD file: {e}")
        return False
    finally:
        conn.close()

def get_fdd_files_by_fdd_id(fdd_id: str) -> List[Dict[str, Any]]:
    """
    Get all file records for a specific FDD.
    
    Args:
        fdd_id: UUID of the FDD as TEXT
        
    Returns:
        List of dictionaries containing file records
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM fdd_files WHERE fdd_id = ? ORDER BY file_type, file_subtype, version",
            (fdd_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Error retrieving FDD files: {e}")
        return []
    finally:
        conn.close()

def get_file_by_storage_uri(storage_uri: str) -> Optional[Dict[str, Any]]:
    """
    Get a file record by its S3 URI.
    
    Args:
        storage_uri: Full S3 URI (e.g., 's3://bucket/path/key')
        
    Returns:
        Dictionary with file data or None if not found
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM fdd_files WHERE storage_uri = ?",
            (storage_uri,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    except sqlite3.Error as e:
        logger.error(f"Error retrieving file by URI: {e}")
        return None
    finally:
        conn.close()

def storage_uri_exists(storage_uri: str) -> bool:
    """
    Check if a file with the given storage URI exists.
    
    Args:
        storage_uri: Full S3 URI to check
        
    Returns:
        True if exists, False otherwise
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT 1 FROM fdd_files WHERE storage_uri = ?",
            (storage_uri,)
        )
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Error checking storage URI existence: {e}")
        return False
    finally:
        conn.close()

def get_next_file_version(fdd_id: str, file_type: str, file_subtype: Optional[str] = None) -> int:
    """
    Get the next version number for a file type.
    
    Args:
        fdd_id: UUID of the FDD
        file_type: Type of file
        file_subtype: Subtype of file (optional)
        
    Returns:
        Next version number (1 if no previous versions exist)
    """
    conn = get_db_connection()
    try:
        if file_subtype is None:
            cursor = conn.execute(
                """
                SELECT MAX(version) FROM fdd_files 
                WHERE fdd_id = ? AND file_type = ? AND file_subtype IS NULL
                """,
                (fdd_id, file_type)
            )
        else:
            cursor = conn.execute(
                """
                SELECT MAX(version) FROM fdd_files 
                WHERE fdd_id = ? AND file_type = ? AND file_subtype = ?
                """,
                (fdd_id, file_type, file_subtype)
            )
        
        max_version = cursor.fetchone()[0]
        return (max_version or 0) + 1
    except sqlite3.Error as e:
        logger.error(f"Error getting next file version: {e}")
        return 1
    finally:
        conn.close()

def get_latest_files() -> List[Dict[str, Any]]:
    """
    Get the latest version of each file from the view.
    
    Returns:
        List of dictionaries containing the latest file records
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM fdd_latest_files")
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Error retrieving latest files: {e}")
        return []
    finally:
        conn.close()

def get_latest_files_by_fdd_id(fdd_id: str) -> List[Dict[str, Any]]:
    """
    Get the latest version of each file for a specific FDD.
    
    Args:
        fdd_id: UUID of the FDD
        
    Returns:
        List of dictionaries containing the latest file records
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM fdd_latest_files WHERE fdd_id = ?",
            (fdd_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Error retrieving latest files for FDD: {e}")
        return []
    finally:
        conn.close()

# Upload Queue Operations
def add_to_upload_queue(
    fdd_id: str,
    file_type: str,
    local_path: str,
    target_s3_key: str,
    file_subtype: Optional[str] = None
) -> Optional[int]:
    """
    Add a new entry to the upload queue.
    
    Args:
        fdd_id: UUID of the FDD
        file_type: Type of file
        local_path: Path to the local file
        target_s3_key: Target S3 key
        file_subtype: Subtype of file (optional)
        
    Returns:
        ID of the new queue item, or None if insert failed
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO upload_queue
            (fdd_id, file_type, file_subtype, local_path, target_s3_key)
            VALUES (?, ?, ?, ?, ?)
            """,
            (fdd_id, file_type, file_subtype, local_path, target_s3_key)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Error adding to upload queue: {e}")
        return None
    finally:
        conn.close()

def get_pending_uploads(max_attempts: int) -> List[Dict[str, Any]]:
    """
    Get pending upload tasks from the queue.
    
    Args:
        max_attempts: Maximum number of attempts allowed
        
    Returns:
        List of dictionaries containing upload tasks
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            SELECT * FROM upload_queue 
            WHERE (status = 'pending' OR (status = 'failed' AND attempts < ?))
            ORDER BY created_at
            """,
            (max_attempts,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Error retrieving pending uploads: {e}")
        return []
    finally:
        conn.close()

def update_upload_status(
    queue_id: int,
    status: str,
    error_message: Optional[str] = None
) -> bool:
    """
    Update the status of an upload task.
    
    Args:
        queue_id: ID of the upload queue item
        status: New status ('uploading', 'complete', 'failed')
        error_message: Error message if status is 'failed'
        
    Returns:
        True if successful, False otherwise
    """
    conn = get_db_connection()
    try:
        if status == 'uploading':
            cursor = conn.execute(
                """
                UPDATE upload_queue
                SET status = ?, attempts = attempts + 1, last_attempt = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, queue_id)
            )
        else:
            cursor = conn.execute(
                """
                UPDATE upload_queue
                SET status = ?, error_message = ?
                WHERE id = ?
                """,
                (status, error_message, queue_id)
            )
        
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Error updating upload status: {e}")
        return False
    finally:
        conn.close() 