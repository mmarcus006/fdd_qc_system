import os
import sys
import json
import PyPDF2
import pdfplumber
import re
from difflib import SequenceMatcher
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def test_verification(pdf_path, json_path):
    """
    Test the verification functionality with a PDF and JSON file
    
    Args:
        pdf_path (str): Path to the PDF file
        json_path (str): Path to the JSON file
        
    Returns:
        dict: Verification results
    """
    print(f"Testing verification with PDF: {os.path.basename(pdf_path)}")
    print(f"JSON: {os.path.basename(json_path)}")
    
    # Load JSON data
    with open(json_path, 'r') as f:
        headers = json.load(f)
    
    # Extract text from PDF
    pdf_text = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Page numbers in PDFs are 0-indexed, but we want 1-indexed for user display
            page_num = i + 1
            text = page.extract_text()
            if text:
                pdf_text[page_num] = text
    
    # Verify each header
    results = []
    for header in headers:
        item_number = header.get('item_number')
        header_text = header.get('text')
        expected_page = header.get('page_number')
        
        # Create regex pattern for the header
        # This handles variations in formatting (e.g., "ITEM 1" vs "ITEM 1.")
        try:
            header_parts = header_text.split("ITEM")
            if len(header_parts) > 1:
                header_pattern = re.compile(
                    r'ITEM\s+\d+\.?\s+' + re.escape(header_parts[1].strip().split(".", 1)[-1].strip()),
                    re.IGNORECASE
                )
            else:
                # Handle cases where "ITEM" might not be in the header text
                header_pattern = re.compile(re.escape(header_text), re.IGNORECASE)
        except Exception as e:
            print(f"Error creating pattern for header {item_number}: {str(e)}")
            header_pattern = re.compile(re.escape(header_text), re.IGNORECASE)
        
        # Search for the header in the PDF
        found_pages = {}
        
        # First, check the expected page and nearby pages
        window_size = 5
        start = max(1, expected_page - window_size)
        end = min(len(pdf_text), expected_page + window_size)
        
        for page_num in range(start, end + 1):
            page_text = pdf_text.get(page_num, "")
            
            # Check for exact match
            if header_text in page_text:
                similarity = 1.0
            else:
                # Check for pattern match
                pattern_match = header_pattern.search(page_text)
                if pattern_match:
                    # Calculate similarity between found text and header text
                    found_text = pattern_match.group(0)
                    similarity = SequenceMatcher(None, header_text, found_text).ratio()
                else:
                    similarity = 0
            
            if similarity > 0:
                found_pages[page_num] = {
                    'confidence': similarity,
                    'distance_from_expected': abs(page_num - expected_page)
                }
        
        # If no pages found in the window, search the entire PDF
        if not found_pages:
            for page_num, page_text in pdf_text.items():
                # Check for exact match
                if header_text in page_text:
                    similarity = 1.0
                else:
                    # Check for pattern match
                    pattern_match = header_pattern.search(page_text)
                    if pattern_match:
                        # Calculate similarity between found text and header text
                        found_text = pattern_match.group(0)
                        similarity = SequenceMatcher(None, header_text, found_text).ratio()
                    else:
                        similarity = 0
                
                if similarity > 0:
                    found_pages[page_num] = {
                        'confidence': similarity,
                        'distance_from_expected': abs(page_num - expected_page)
                    }
        
        # Determine verification status
        if not found_pages:
            status = "not_found"
            confidence = 0
            best_page = None
        else:
            # Find the page with the highest confidence
            best_page = max(found_pages.items(), key=lambda x: x[1]['confidence'])
            page_num = best_page[0]
            confidence = best_page[1]['confidence']
            
            # Determine status based on confidence and match with expected page
            if page_num == expected_page and confidence > 0.9:
                status = "verified"
            elif confidence > 0.8:
                status = "likely_correct"
            elif confidence > 0.6:
                status = "needs_review"
            else:
                status = "likely_incorrect"
        
        result = {
            'item_number': item_number,
            'header_text': header_text,
            'expected_page': expected_page,
            'found_pages': found_pages,
            'best_match_page': best_page[0] if best_page else None,
            'confidence': confidence,
            'status': status
        }
        
        results.append(result)
    
    # Print summary
    print("\nVerification Summary:")
    print(f"Total headers: {len(results)}")
    
    status_counts = {
        'verified': 0,
        'likely_correct': 0,
        'needs_review': 0,
        'likely_incorrect': 0,
        'not_found': 0
    }
    
    for result in results:
        status = result.get('status')
        if status in status_counts:
            status_counts[status] += 1
    
    print(f"Verified: {status_counts['verified']}")
    print(f"Likely correct: {status_counts['likely_correct']}")
    print(f"Needs review: {status_counts['needs_review']}")
    print(f"Likely incorrect: {status_counts['likely_incorrect']}")
    print(f"Not found: {status_counts['not_found']}")
    
    # Print headers that need review
    print("\nHeaders that need review:")
    for result in results:
        if result.get('status') in ['needs_review', 'likely_incorrect', 'not_found']:
            print(f"Item {result['item_number']}: {result['header_text']}")
            print(f"  Expected page: {result['expected_page']}")
            print(f"  Best match page: {result['best_match_page']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print()
    
    return results

if __name__ == "__main__":
    # Test with the provided sample data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(os.path.dirname(base_dir), "upload")
    
    # Test files
    pdf_files = [
        os.path.join(upload_dir, "Christian_Brothers_Automotive_Corporation_FDD_2024_ID636152.pdf"),
        os.path.join(upload_dir, "Dryer_Vent_Superheroes_Franchising_LLC_FDD_2024_ID637010.pdf"),
        os.path.join(upload_dir, "RUTHS_CHRIS_STEAK_HOUSE_FRANCHISE_LLC_FDD_2025_ID637618.pdf")
    ]
    
    json_files = [
        os.path.join(upload_dir, "00a8fab2-e570-4818-a60e-47fbd351ef0d_origin_huridocs_analysis_extracted_headers.json"),
        os.path.join(upload_dir, "00a72862-7472-473d-87c4-863010fa4835_origin_huridocs_analysis_extracted_headers.json"),
        os.path.join(upload_dir, "00cccd30-3f35-497c-801e-2047cd516b35_origin_huridocs_analysis_extracted_headers.json")
    ]
    
    # Test each PDF with its corresponding JSON
    for i in range(len(pdf_files)):
        if os.path.exists(pdf_files[i]) and os.path.exists(json_files[i]):
            print(f"\n{'='*50}")
            print(f"Testing file pair {i+1}:")
            test_verification(pdf_files[i], json_files[i])
            print(f"{'='*50}\n")
        else:
            print(f"File pair {i+1} not found. Skipping.")
