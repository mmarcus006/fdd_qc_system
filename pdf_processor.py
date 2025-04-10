import os
import json
import PyPDF2
import pdfplumber
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import pandas as pd
from difflib import SequenceMatcher

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class PDFProcessor:
    """
    Class for processing PDF files and extracting text for FDD header verification.
    """
    
    def __init__(self, pdf_path):
        """
        Initialize the PDF processor with a PDF file path.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.pdf_name = os.path.basename(pdf_path)
        self.text_by_page = {}
        self.total_pages = 0
        self.toc_page = None  # Will store the detected Table of Contents page number
        self.load_pdf()
        self.detect_toc_page()  # Detect TOC page on initialization
    
    def load_pdf(self):
        """
        Load the PDF file and extract text from each page.
        """
        try:
            # Get total number of pages
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                self.total_pages = len(pdf_reader.pages)
            
            # Extract text from each page using pdfplumber for better text extraction
            with pdfplumber.open(self.pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Page numbers in PDFs are 0-indexed, but we want 1-indexed for user display
                    page_num = i + 1
                    text = page.extract_text()
                    if text:
                        self.text_by_page[page_num] = text
        
        except Exception as e:
            print(f"Error loading PDF {self.pdf_path}: {str(e)}")
            raise
    
    def detect_toc_page(self):
        """
        Detects the Table of Contents page in the PDF.
        Uses two criteria:
        1. Page contains phrases like "Table of Contents", "Contents", etc.
        2. Page contains multiple headers (3 or more ITEM X entries)
        
        The first matching page is considered the TOC.
        """
        # Define TOC regex patterns
        toc_patterns = [
            re.compile(r'table\s+of\s+contents', re.IGNORECASE),
            re.compile(r'^contents$', re.IGNORECASE),
            re.compile(r'^table\s+of\s+contents$', re.IGNORECASE),
            re.compile(r'table\s+of\s+contents|t\.?o\.?c\.?', re.IGNORECASE)
        ]
        
        # Define pattern to count headers in a page
        header_pattern = re.compile(r'ITEM\s+\d+', re.IGNORECASE)
        
        # Check each page - prioritize pages near the beginning of the document
        # Start from the first page and check the first ~20 pages (typical TOC location)
        search_range = min(20, self.total_pages)
        
        for page_num in range(1, search_range + 1):
            page_text = self.get_page_text(page_num)
            
            # Method 1: Check for TOC keywords
            if any(pattern.search(page_text) for pattern in toc_patterns):
                self.toc_page = page_num
                print(f"TOC detected on page {page_num} using keyword detection")
                return
            
            # Method 2: Count ITEM X occurrences
            headers_found = len(header_pattern.findall(page_text))
            if headers_found >= 3:  # If 3 or more headers are found, likely a TOC
                self.toc_page = page_num
                print(f"TOC detected on page {page_num} with {headers_found} header matches")
                return
        
        # If no TOC found, log it
        print("No Table of Contents page detected")
    
    def get_page_text(self, page_num):
        """
        Get the text from a specific page.
        
        Args:
            page_num (int): Page number (1-indexed)
            
        Returns:
            str: Text content of the page
        """
        return self.text_by_page.get(page_num, "")
    
    def search_text_in_page(self, text, page_num):
        """
        Search for text in a specific page.
        
        Args:
            text (str): Text to search for
            page_num (int): Page number to search in
            
        Returns:
            bool: True if text is found, False otherwise
        """
        page_text = self.get_page_text(page_num)
        return text in page_text
    
    def find_text_in_pdf(self, text, start_page=1, end_page=None):
        """
        Find text in the PDF within a range of pages.
        
        Args:
            text (str): Text to search for
            start_page (int): Starting page number
            end_page (int): Ending page number
            
        Returns:
            list: List of page numbers where the text was found
        """
        if end_page is None:
            end_page = self.total_pages
        
        found_pages = []
        for page_num in range(start_page, end_page + 1):
            if self.search_text_in_page(text, page_num):
                found_pages.append(page_num)
        
        return found_pages
    
    def find_header_in_pdf(self, header_text, expected_page=None, window_size=5):
        """
        Find a header in the PDF, focusing around the expected page if provided.
        Excludes matches on the detected Table of Contents page.
        
        Args:
            header_text (str): Header text to search for
            expected_page (int): Expected page number
            window_size (int): Number of pages to search before and after expected page
            
        Returns:
            dict: Dictionary with found page numbers and confidence scores
        """
        results = {}
        
        # If expected page is provided, search around that page first
        if expected_page:
            start = max(1, expected_page - window_size)
            end = min(self.total_pages, expected_page + window_size)
        else:
            start = 1
            end = self.total_pages
        
        # Create regex pattern for the header
        # This handles variations in formatting (e.g., "ITEM 1" vs "ITEM 1.")
        try:
            header_parts = header_text.split("ITEM")
            if len(header_parts) > 1:
                escaped_text = re.escape(header_parts[1].strip().split(".", 1)[-1].strip())
                pattern_text = r'ITEM\s+\d+\.?\s+' + escaped_text
                header_pattern = re.compile(pattern_text, re.IGNORECASE)
            else:
                # Handle cases where "ITEM" might not be in the header text
                header_pattern = re.compile(re.escape(header_text), re.IGNORECASE)
        except Exception as e:
            print(f"Error creating pattern for header text: {str(e)}")
            header_pattern = re.compile(re.escape(header_text), re.IGNORECASE)
        
        # Ensure the pattern isn't empty or too short
        if len(str(header_pattern.pattern)) < 3:
            print(f"Warning: Pattern too short for reliable matching: {header_pattern.pattern}")
        
        # Search for exact and pattern matches
        for page_num in range(start, end + 1):
            # Skip the TOC page if we've identified one
            if self.toc_page and page_num == self.toc_page:
                print(f"Skipping TOC page {page_num} for header '{header_text[:30]}...'")
                continue
                
            page_text = self.get_page_text(page_num)
            
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
                results[page_num] = {
                    'confidence': similarity,
                    'distance_from_expected': abs(page_num - expected_page) if expected_page else None
                }
        
        # If no matches found and we skipped the TOC, check if it's really ONLY on the TOC
        if not results and self.toc_page:
            page_num = self.toc_page
            page_text = self.get_page_text(page_num)
            
            # Same match logic as above
            if header_text in page_text:
                # Lower confidence for TOC matches
                similarity = 0.8  # Reduced confidence for TOC matches
                print(f"Found header on TOC page only: '{header_text[:30]}...'")
                results[page_num] = {
                    'confidence': similarity,
                    'distance_from_expected': abs(page_num - expected_page) if expected_page else None,
                    'is_toc_match': True  # Flag this as a TOC match
                }
            else:
                pattern_match = header_pattern.search(page_text)
                if pattern_match:
                    found_text = pattern_match.group(0)
                    similarity = SequenceMatcher(None, header_text, found_text).ratio() * 0.8  # Reduced
                    results[page_num] = {
                        'confidence': similarity,
                        'distance_from_expected': abs(page_num - expected_page) if expected_page else None,
                        'is_toc_match': True  # Flag this as a TOC match
                    }
        
        return results


class JSONProcessor:
    """
    Class for processing JSON files containing FDD header information.
    """
    
    def __init__(self, json_path):
        """
        Initialize the JSON processor with a JSON file path.
        
        Args:
            json_path (str): Path to the JSON file
        """
        self.json_path = json_path
        self.json_name = os.path.basename(json_path)
        self.headers = []
        self.load_json()
    
    def load_json(self):
        """
        Load the JSON file containing FDD header information.
        """
        try:
            with open(self.json_path, 'r') as file:
                self.headers = json.load(file)
        except Exception as e:
            print(f"Error loading JSON {self.json_path}: {str(e)}")
            raise
    
    def get_header_by_item_number(self, item_number):
        """
        Get header information by item number.
        
        Args:
            item_number (int): Item number (1-23)
            
        Returns:
            dict: Header information
        """
        for header in self.headers:
            if header.get('item_number') == item_number:
                return header
        return None
    
    def get_all_headers(self):
        """
        Get all headers.
        
        Returns:
            list: List of all headers
        """
        return self.headers
    
    def update_header_page_number(self, item_number, new_page_number):
        """
        Update the page number for a header.
        
        Args:
            item_number (int): Item number
            new_page_number (int): New page number
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        for i, header in enumerate(self.headers):
            if header.get('item_number') == item_number:
                self.headers[i]['page_number'] = new_page_number
                return True
        return False
    
    def save_json(self, output_path=None):
        """
        Save the updated JSON data.
        
        Args:
            output_path (str): Path to save the JSON file
            
        Returns:
            str: Path where the JSON was saved
        """
        if output_path is None:
            # Create a new filename with _verified suffix
            base_name = os.path.splitext(self.json_path)[0]
            output_path = f"{base_name}_verified.json"
        
        try:
            with open(output_path, 'w') as file:
                json.dump(self.headers, file, indent=2)
            return output_path
        except Exception as e:
            print(f"Error saving JSON to {output_path}: {str(e)}")
            raise


class VerificationEngine:
    """
    Engine for verifying FDD headers against PDF content.
    """
    
    def __init__(self, pdf_processor, json_processor):
        """
        Initialize the verification engine.
        
        Args:
            pdf_processor (PDFProcessor): PDF processor instance
            json_processor (JSONProcessor): JSON processor instance
        """
        self.pdf_processor = pdf_processor
        self.json_processor = json_processor
        self.verification_results = {}
    
    def verify_all_headers(self):
        """
        Verify all headers in the JSON against the PDF.
        
        Returns:
            dict: Verification results for all headers
        """
        headers = self.json_processor.get_all_headers()
        
        for header in headers:
            item_number = header.get('item_number')
            header_text = header.get('text')
            expected_page = header.get('page_number')
            
            result = self.verify_header(item_number, header_text, expected_page)
            self.verification_results[item_number] = result
        
        return self.verification_results
    
    def verify_header(self, item_number, header_text, expected_page):
        """
        Verify a single header against the PDF.
        
        Args:
            item_number (int): Item number
            header_text (str): Header text
            expected_page (int): Expected page number
            
        Returns:
            dict: Verification result
        """
        # Find the header in the PDF
        found_pages = self.pdf_processor.find_header_in_pdf(header_text, expected_page)
        
        # If no pages found, search in the entire PDF
        if not found_pages and expected_page:
            found_pages = self.pdf_processor.find_header_in_pdf(header_text)
        
        # Determine verification status
        if not found_pages:
            status = "not_found"
            confidence = 0
            best_page = None
        else:
            # Filter out TOC matches if we have other options
            non_toc_matches = {page: data for page, data in found_pages.items() 
                              if not data.get('is_toc_match', False)}
            
            # Use non-TOC matches if available, otherwise use all matches
            pages_to_consider = non_toc_matches if non_toc_matches else found_pages
            
            # Find the page with the highest confidence
            best_page = max(pages_to_consider.items(), key=lambda x: x[1]['confidence'])
            page_num = best_page[0]
            confidence = best_page[1]['confidence']
            
            # If this is a TOC match, log it and adjust confidence
            if best_page[1].get('is_toc_match', False):
                print(f"Warning: Best match for Item {item_number} is on TOC page")
                # Reduce confidence for TOC matches to ensure they're more likely to be flagged
                confidence = confidence * 0.7  # Further reduce confidence
            
            # Determine status based on confidence and match with expected page
            if page_num == expected_page and confidence > 0.9:
                status = "verified"
            elif confidence > 0.8:
                status = "likely_correct"
            elif confidence > 0.6:
                status = "needs_review"
            else:
                status = "likely_incorrect"
            
            # If it's a TOC match and not already flagged, force to "needs_review"
            if best_page[1].get('is_toc_match', False) and status != "verified":
                status = "needs_review"
        
        return {
            'item_number': item_number,
            'header_text': header_text,
            'expected_page': expected_page,
            'found_pages': found_pages,
            'best_match_page': best_page[0] if best_page else None,
            'confidence': confidence,
            'status': status,
            'is_toc_match': best_page[1].get('is_toc_match', False) if best_page else False
        }
    
    def get_verification_summary(self):
        """
        Get a summary of the verification results.
        
        Returns:
            dict: Summary of verification results
        """
        if not self.verification_results:
            self.verify_all_headers()
        
        summary = {
            'total': len(self.verification_results),
            'verified': 0,
            'likely_correct': 0,
            'needs_review': 0,
            'likely_incorrect': 0,
            'not_found': 0
        }
        
        for result in self.verification_results.values():
            status = result.get('status')
            if status in summary:
                summary[status] += 1
        
        return summary
    
    def get_headers_by_status(self, status):
        """
        Get headers with a specific verification status.
        
        Args:
            status (str): Verification status
            
        Returns:
            list: Headers with the specified status
        """
        if not self.verification_results:
            self.verify_all_headers()
        
        return [result for result in self.verification_results.values() if result.get('status') == status]
    
    def get_all_results(self):
        """
        Get all verification results.
        
        Returns:
            dict: All verification results
        """
        if not self.verification_results:
            self.verify_all_headers()
        
        return self.verification_results


# Example usage
if __name__ == "__main__":
    # This is just for testing the module directly
    pdf_path = "/path/to/pdf"
    json_path = "/path/to/json"
    
    if os.path.exists(pdf_path) and os.path.exists(json_path):
        pdf_processor = PDFProcessor(pdf_path)
        json_processor = JSONProcessor(json_path)
        
        engine = VerificationEngine(pdf_processor, json_processor)
        results = engine.verify_all_headers()
        
        print("Verification Summary:")
        print(engine.get_verification_summary())
        
        print("\nHeaders that need review:")
        for header in engine.get_headers_by_status("needs_review"):
            print(f"Item {header['item_number']}: {header['header_text']}")
            print(f"  Expected page: {header['expected_page']}")
            print(f"  Best match page: {header['best_match_page']}")
            print(f"  Confidence: {header['confidence']:.2f}")
