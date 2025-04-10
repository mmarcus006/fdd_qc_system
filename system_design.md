# FDD Header Quality Control System - Design Document

## System Overview
The FDD Header Quality Control System is designed to automate the verification of page numbers for the standard 23 sections of Franchise Disclosure Documents (FDDs). The system combines automated batch verification with a comprehensive review interface that allows users to view all 23 headers side by side with the PDF document.

## Architecture Components

### 1. Data Processing Module
- **PDF Text Extraction**: Extract text from PDF files using PyPDF2 or pdfplumber
- **JSON Data Processing**: Parse and process the JSON files containing extracted headers
- **Data Mapping**: Map JSON data to PDF content for verification

### 2. Verification Engine
- **Regex Pattern Matching**: Use regular expressions to identify section headers in PDF text
- **NLP-based Verification**: Implement text similarity comparison between extracted headers and PDF content
- **Page Number Validation**: Verify if the extracted page numbers match the actual location in the PDF
- **Confidence Scoring**: Calculate confidence scores for each verification result

### 3. User Interface
- **PDF Viewer**: Display PDF document with navigation controls
- **Header Dashboard**: Show all 23 headers with their extracted page numbers and verification status
- **Side-by-Side Comparison**: Display PDF content alongside header information
- **Edit Interface**: Allow users to modify incorrect page numbers
- **Approval System**: Enable users to approve or reject verification results

### 4. Data Persistence
- **Result Storage**: Save verification results and user edits
- **Export Functionality**: Export corrected data back to JSON format

## Verification Methods

### Regex-based Verification
- Create patterns to match standard FDD section headers (e.g., "ITEM 1. THE FRANCHISOR")
- Search for these patterns in the PDF text
- Compare the page where the pattern is found with the extracted page number

### NLP-based Verification
- Calculate text similarity between extracted headers and PDF content
- Use techniques like cosine similarity or Levenshtein distance
- Set thresholds for automatic approval/flagging

### Flagging System
- **Auto-approve**: High confidence matches (>90%)
- **Flag for Review**: Medium confidence matches (70-90%)
- **Auto-flag**: Low confidence matches (<70%)

## User Interface Design

### Main Window Layout
- Left panel: PDF viewer with navigation controls
- Right panel: Header dashboard showing all 23 items
- Bottom panel: Detailed view of selected header with edit controls

### Header Dashboard
- Display item number, header text, extracted page number, and verification status
- Color-coding for status (green: verified, yellow: needs review, red: likely incorrect)
- Sort and filter options

### Interaction Flow
1. System loads PDF and JSON data
2. Batch verification runs automatically
3. Results are displayed in the dashboard
4. User reviews flagged items
5. User can navigate to specific pages in the PDF
6. User can edit page numbers and approve/reject verification results
7. Final results can be exported

## Technology Stack
- **Backend**: Python 3.x
- **PDF Processing**: PyPDF2, pdfplumber
- **UI Framework**: Tkinter or PyQt for desktop application
- **NLP Components**: NLTK or spaCy for text processing
- **Data Handling**: Pandas for data manipulation
