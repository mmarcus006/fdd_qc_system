# FDD Header Quality Control System

A system for automating the quality control of extracted FDD (Franchise Disclosure Document) section headers and page numbers.

## Overview

The FDD Header Quality Control System allows users to:

1. Automatically verify the accuracy of extracted page numbers for the standard 23 sections of FDDs
2. View PDFs alongside JSON data in a side-by-side interface
3. Approve or reject verification results
4. Edit incorrect page numbers
5. Save the corrected data back to JSON format

The system combines automated batch verification with a comprehensive review interface that shows all 23 headers at once.

## Features

- **PDF Viewing**: View PDF documents with navigation controls
- **Automated Verification**: Uses regex and text similarity to verify header page numbers
- **Confidence Scoring**: Calculates confidence scores for verification results
- **Color-Coded Status**: Visual indicators for verification status (verified, likely correct, needs review, etc.)
- **Side-by-Side Comparison**: View PDF content alongside header information
- **Edit Functionality**: Easily update incorrect page numbers
- **Approval System**: Approve or reject verification results
- **Result Export**: Save corrected data back to JSON format

## Installation

### Prerequisites

- Python 3.8 or higher
- Tkinter (usually included with Python)

### Required Python Packages

```
pip install PyPDF2 pdfplumber nltk spacy pandas PyMuPDF
```

## Usage

1. Run the application:

```
python main.py
```

2. Load a PDF file using File > Load PDF
3. Load a JSON file using File > Load JSON
4. Click 'Verify All' to check header page numbers
5. Review the results in the table
6. Double-click a header to go to its expected page
7. Use the detail panel to edit page numbers
8. Click 'Approve' or 'Reject' to update status
9. Save your changes using File > Save Results

## File Structure

- `main.py`: Entry point for the application
- `fdd_qc_app.py`: Main application GUI and functionality
- `pdf_processor.py`: Classes for processing PDFs, JSON data, and verification
- `test_verification.py`: Script for testing verification functionality

## Verification Process

The system uses multiple methods to verify header page numbers:

1. **Regex Pattern Matching**: Identifies section headers in PDF text
2. **Text Similarity Comparison**: Calculates similarity between extracted headers and PDF content
3. **Confidence Scoring**: Determines verification status based on match confidence

## Status Indicators

- **Green**: Verified (high confidence match on expected page)
- **Light Green**: Likely correct (high confidence match on different page)
- **Yellow**: Needs review (medium confidence match)
- **Red**: Likely incorrect (low confidence match)
- **Gray**: Not found (no match found)

## Testing Results

The verification engine was tested with three sample FDD documents:

1. Christian Brothers Automotive Corporation FDD 2024:
   - 20 verified, 1 likely correct, 2 not found

2. Dryer Vent Superheroes Franchising LLC FDD 2024:
   - 1 verified, 19 likely correct, 3 not found

3. Ruth's Chris Steak House Franchise LLC FDD 2025:
   - 16 verified, 4 likely correct, 3 not found

These results demonstrate the system's effectiveness in automatically verifying most headers while flagging those that need manual review.
