# FDD Document Management System

A Python-based system to manage files related to Franchise Disclosure Documents (FDDs) using AWS S3 for storage and a SQLite database for tracking file metadata.

## System Overview

This system provides utilities for:
- Discovering existing files in an S3 bucket
- Registering files in a SQLite database according to a defined naming convention
- Managing file uploads to S3
- Querying the database for file metadata

## Directory Structure

- `config.py`: Configuration module for loading AWS credentials and settings
- `db_utils.py`: Database utilities for interacting with SQLite
- `naming_utils.py`: Utilities for generating and parsing S3 keys
- `s3_utils.py`: Utilities for interacting with AWS S3
- `discover_s3.py`: Script for discovering and registering S3 objects
- `process_uploads.py`: Script for processing pending uploads
- `main.py`: Example usage of the system

## Prerequisites

- Python 3.8+
- AWS credentials with access to the S3 bucket
- SQLite database (will be created automatically if it doesn't exist)

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following contents:
```
# AWS Credentials
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
AWS_REGION=us-east-1  # Or your desired region

# S3 Configuration
S3_BUCKET_NAME=fddsearch-fdd-pipeline
S3_PREFIX=pdfs/  # Include trailing slash if it's a 'folder'

# Database Configuration
SQLITE_DB_PATH=./fdd_metadata.db  # Path to your SQLite database file

# Upload Processing Configuration (Optional)
MAX_UPLOAD_ATTEMPTS=3
```

## Usage

### Initialize the System

```
python main.py --init
```

### Create a New FDD Record

```
python main.py --create-fdd --franchise-name "Example Franchise" --doc-year 2023
```

### Add a File to the Upload Queue

```
python main.py --queue-upload --fdd-id <FDD_ID> --local-path /path/to/file.pdf --file-type pdf
```

### Process Pending Uploads

```
python main.py --process-uploads
```

### Run S3 Discovery

```
python main.py --run-discovery
```

### Query Files for an FDD

```
python main.py --query-files --fdd-id <FDD_ID>
```

To include all versions (not just the latest):

```
python main.py --query-files --fdd-id <FDD_ID> --all-versions
```

### Get a Presigned URL for a File

```
python main.py --get-url --fdd-id <FDD_ID> --file-type pdf
```

## S3 Path Structure

The system is configured to work with the following S3 paths:

- `pdfs/`: Original PDF formatted FDDs
- `extracted_headers/original/huridoc_extract/`: Document layout analysis header extraction
- `extracted_headers/original/llm_extract/`: LLM-based header extraction
- `document_layout_analysis/`: Huridoc document layout analysis
- `qc_results/`: Quality control results with versioning based on corrected headers

## File Naming Convention

Files are named following this convention:
```
{fdd_id}_{file_type}_{file_subtype}_v{version}.{extension}
```

For example:
```
550e8400-e29b-41d4-a716-446655440000_pdf_original_v1.pdf
```

## Database Schema

The system uses a SQLite database with the following tables:

- `fdd`: Stores metadata about FDDs
- `fdd_files`: Stores metadata about files related to FDDs
- `upload_queue`: Manages pending file uploads
- `fdd_latest_files` (view): Gets the latest version of each file

## Requirements.txt

Create a `requirements.txt` file with the following contents:

```
boto3>=1.28.0
python-dotenv>=1.0.0
``` 