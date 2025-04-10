import os
import sys
import json
import re
import random
from typing import Dict, List, Any, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_processor import PDFProcessor, JSONProcessor
from enhanced_verification import EnhancedVerificationEngine, HeaderDatabase
from advanced_nlp import AdvancedNLPProcessor

class EdgeCaseTestSuite:
    """
    Test suite for evaluating the verification system against edge cases
    """
    
    def __init__(self, pdf_path, json_path):
        """
        Initialize the edge case test suite
        
        Args:
            pdf_path (str): Path to the PDF file
            json_path (str): Path to the JSON file
        """
        self.pdf_path = pdf_path
        self.json_path = json_path
        self.pdf_processor = PDFProcessor(pdf_path)
        self.json_processor = JSONProcessor(json_path)
        self.engine = EnhancedVerificationEngine(self.pdf_processor, self.json_processor)
        self.nlp_processor = AdvancedNLPProcessor()
        
        # Load the original headers
        with open(json_path, 'r') as f:
            self.original_headers = json.load(f)
    
    def run_all_tests(self):
        """
        Run all edge case tests
        
        Returns:
            dict: Test results
        """
        results = {
            "inconsistent_formatting": self.test_inconsistent_formatting(),
            "multi_line_headers": self.test_multi_line_headers(),
            "similar_headers": self.test_similar_headers(),
            "missing_headers": self.test_missing_headers(),
            "page_boundary_cases": self.test_page_boundary_cases(),
            "unusual_page_numbering": self.test_unusual_page_numbering(),
            "low_quality_text": self.test_low_quality_text(),
            "special_characters": self.test_special_characters()
        }
        
        return results
    
    def test_inconsistent_formatting(self):
        """
        Test headers with inconsistent formatting
        
        Returns:
            dict: Test results
        """
        print("\n=== Testing Inconsistent Formatting ===")
        
        # Create variations of headers with inconsistent formatting
        variations = []
        
        for header in self.original_headers[:5]:  # Test with first 5 headers
            item_number = header.get('item_number')
            original_text = header.get('text')
            expected_page = header.get('page_number')
            
            # Create variations
            variations.extend([
                {
                    'item_number': item_number,
                    'text': original_text.replace('.', ''),  # Remove periods
                    'page_number': expected_page,
                    'variation': 'no_periods'
                },
                {
                    'item_number': item_number,
                    'text': original_text.replace(' ', '   '),  # Extra spaces
                    'page_number': expected_page,
                    'variation': 'extra_spaces'
                },
                {
                    'item_number': item_number,
                    'text': original_text.lower(),  # Lowercase
                    'page_number': expected_page,
                    'variation': 'lowercase'
                },
                {
                    'item_number': item_number,
                    'text': f"ITEM{item_number}{original_text.split(f'ITEM {item_number}')[1] if len(original_text.split(f'ITEM {item_number}')) > 1 else ''}",  # No space after ITEM
                    'page_number': expected_page,
                    'variation': 'no_space_after_item'
                }
            ])
        
        # Test each variation
        results = []
        
        for variation in variations:
            print(f"\nTesting variation: {variation['variation']}")
            print(f"Original: {self.original_headers[variation['item_number']-1]['text']}")
            print(f"Modified: {variation['text']}")
            
            # Verify with pattern matching
            pattern_result = self._verify_with_patterns(variation)
            
            # Verify with NLP
            nlp_result = self._verify_with_nlp(variation)
            
            # Determine if the test passed
            passed = (pattern_result['verified'] or nlp_result['verified'])
            
            result = {
                'item_number': variation['item_number'],
                'variation': variation['variation'],
                'original_text': self.original_headers[variation['item_number']-1]['text'],
                'modified_text': variation['text'],
                'pattern_verified': pattern_result['verified'],
                'pattern_confidence': pattern_result['confidence'],
                'nlp_verified': nlp_result['verified'],
                'nlp_confidence': nlp_result['confidence'],
                'passed': passed
            }
            
            results.append(result)
            
            print(f"Pattern verification: {'✓' if pattern_result['verified'] else '✗'} ({pattern_result['confidence']:.2f})")
            print(f"NLP verification: {'✓' if nlp_result['verified'] else '✗'} ({nlp_result['confidence']:.2f})")
            print(f"Test {'passed' if passed else 'failed'}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['passed']) / len(results) if results else 0
        print(f"\nInconsistent formatting test success rate: {success_rate:.2%}")
        
        return {
            'success_rate': success_rate,
            'results': results
        }
    
    def test_multi_line_headers(self):
        """
        Test headers that span multiple lines
        
        Returns:
            dict: Test results
        """
        print("\n=== Testing Multi-line Headers ===")
        
        # Create multi-line variations of headers
        variations = []
        
        for header in self.original_headers[:5]:  # Test with first 5 headers
            item_number = header.get('item_number')
            original_text = header.get('text')
            expected_page = header.get('page_number')
            
            # Split the header at different points
            words = original_text.split()
            if len(words) >= 4:
                split_point = len(words) // 2
                
                # Create variations
                variations.extend([
                    {
                        'item_number': item_number,
                        'text': f"{' '.join(words[:split_point])}\n{' '.join(words[split_point:])}",
                        'page_number': expected_page,
                        'variation': 'split_middle'
                    },
                    {
                        'item_number': item_number,
                        'text': f"ITEM {item_number}\n{' '.join(words[2:]) if len(words) > 2 else ''}",
                        'page_number': expected_page,
                        'variation': 'split_after_item'
                    }
                ])
        
        # Test each variation
        results = []
        
        for variation in variations:
            print(f"\nTesting variation: {variation['variation']}")
            print(f"Original: {self.original_headers[variation['item_number']-1]['text']}")
            print(f"Modified: {variation['text'].replace('\n', ' <newline> ')}")
            
            # Verify with pattern matching
            pattern_result = self._verify_with_patterns(variation)
            
            # Verify with NLP
            nlp_result = self._verify_with_nlp(variation)
            
            # Determine if the test passed
            passed = (pattern_result['verified'] or nlp_result['verified'])
            
            result = {
                'item_number': variation['item_number'],
                'variation': variation['variation'],
                'original_text': self.original_headers[variation['item_number']-1]['text'],
                'modified_text': variation['text'],
                'pattern_verified': pattern_result['verified'],
                'pattern_confidence': pattern_result['confidence'],
                'nlp_verified': nlp_result['verified'],
                'nlp_confidence': nlp_result['confidence'],
                'passed': passed
            }
            
            results.append(result)
            
            print(f"Pattern verification: {'✓' if pattern_result['verified'] else '✗'} ({pattern_result['confidence']:.2f})")
            print(f"NLP verification: {'✓' if nlp_result['verified'] else '✗'} ({nlp_result['confidence']:.2f})")
            print(f"Test {'passed' if passed else 'failed'}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['passed']) / len(results) if results else 0
        print(f"\nMulti-line headers test success rate: {success_rate:.2%}")
        
        return {
            'success_rate': success_rate,
            'results': results
        }
    
    def test_similar_headers(self):
        """
        Test headers with very similar text
        
        Returns:
            dict: Test results
        """
        print("\n=== Testing Similar Headers ===")
        
        # Find pairs of headers with similar text
        similar_pairs = []
        
        for i, header1 in enumerate(self.original_headers):
            for j, header2 in enumerate(self.original_headers):
                if i != j:
                    # Calculate similarity
                    similarity = self.nlp_processor.compute_text_similarity(
                        header1.get('text', ''),
                        header2.get('text', ''),
                        method='ensemble'
                    )
                    
                    if similarity > 0.7:  # Consider headers similar if similarity > 0.7
                        similar_pairs.append({
                            'header1': header1,
                            'header2': header2,
                            'similarity': similarity
                        })
        
        # If no naturally similar headers found, create artificial ones
        if not similar_pairs:
            for i, header in enumerate(self.original_headers[:5]):  # Use first 5 headers
                item_number = header.get('item_number')
                original_text = header.get('text')
                expected_page = header.get('page_number')
                
                # Create a similar header by changing a few words
                words = original_text.split()
                if len(words) >= 5:
                    # Change one word in the middle
                    modified_words = words.copy()
                    modified_words[len(words) // 2] = "SIMILAR"
                    
                    similar_pairs.append({
                        'header1': header,
                        'header2': {
                            'item_number': item_number + 10 if item_number + 10 <= 23 else item_number - 10,
                            'text': ' '.join(modified_words),
                            'page_number': expected_page + 5 if expected_page + 5 <= self.pdf_processor.total_pages else expected_page - 5
                        },
                        'similarity': 0.8
                    })
        
        # Test each pair
        results = []
        
        for pair in similar_pairs[:5]:  # Limit to 5 pairs for brevity
            header1 = pair['header1']
            header2 = pair['header2']
            
            print(f"\nTesting similar headers (similarity: {pair['similarity']:.2f}):")
            print(f"Header 1 (Item {header1['item_number']}): {header1['text']}")
            print(f"Header 2 (Item {header2['item_number']}): {header2['text']}")
            
            # Verify header1 with pattern matching
            pattern_result1 = self._verify_with_patterns(header1)
            
            # Verify header1 with NLP
            nlp_result1 = self._verify_with_nlp(header1)
            
            # Verify header2 with pattern matching
            pattern_result2 = self._verify_with_patterns(header2)
            
            # Verify header2 with NLP
            nlp_result2 = self._verify_with_nlp(header2)
            
            # Check if the system correctly distinguished between the similar headers
            correct_distinction = (
                (pattern_result1['best_match_page'] == header1['page_number'] or 
                 nlp_result1['best_match_page'] == header1['page_number']) and
                (pattern_result2['best_match_page'] == header2['page_number'] or 
                 nlp_result2['best_match_page'] == header2['page_number'])
            )
            
            result = {
                'header1_item': header1['item_number'],
                'header2_item': header2['item_number'],
                'similarity': pair['similarity'],
                'header1_pattern_verified': pattern_result1['verified'],
                'header1_nlp_verified': nlp_result1['verified'],
                'header2_pattern_verified': pattern_result2['verified'],
                'header2_nlp_verified': nlp_result2['verified'],
                'correct_distinction': correct_distinction
            }
            
            results.append(result)
            
            print(f"Header 1 pattern verification: {'✓' if pattern_result1['verified'] else '✗'} (page: {pattern_result1['best_match_page']})")
            print(f"Header 1 NLP verification: {'✓' if nlp_result1['verified'] else '✗'} (page: {nlp_result1['best_match_page']})")
            print(f"Header 2 pattern verification: {'✓' if pattern_result2['verified'] else '✗'} (page: {pattern_result2['best_match_page']})")
            print(f"Header 2 NLP verification: {'✓' if nlp_result2['verified'] else '✗'} (page: {nlp_result2['best_match_page']})")
            print(f"Correct distinction: {'✓' if correct_distinction else '✗'}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['correct_distinction']) / len(results) if results else 0
        print(f"\nSimilar headers test success rate: {success_rate:.2%}")
        
        return {
            'success_rate': success_rate,
            'results': results
        }
    
    def test_missing_headers(self):
        """
        Test handling of missing headers
        
        Returns:
            dict: Test results
        """
        print("\n=== Testing Missing Headers ===")
        
        # Create a set of headers with some missing
        all_item_numbers = set(range(1, 24))  # Items 1-23
        existing_item_numbers = set(h['item_number'] for h in self.original_headers)
        
        # If all headers exist, randomly remove some for testing
        if len(existing_item_numbers) == 23:
            # Randomly select 3-5 headers to remove
            num_to_remove = random.randint(3, 5)
            to_remove = random.sample(list(existing_item_numbers), num_to_remove)
            modified_headers = [h for h in self.original_headers if h['item_number'] not in to_remove]
            missing_items = to_remove
        else:
            # Use naturally missing headers
            modified_headers = self.original_headers
            missing_items = all_item_numbers - existing_item_numbers
        
        print(f"Missing headers: {sorted(missing_items)}")
        
        # Test the system's ability to handle missing headers
        results = []
        
        # First, verify all existing headers
        verification_results = {}
        for header in modified_headers:
            item_number = header['item_number']
            result = self.engine._verify_with_patterns(item_number, header['text'], header['page_number'])
            verification_results[item_number] = result
        
        # Then, try to predict the missing headers
        for missing_item in missing_items:
            # Find the closest existing headers before and after
            prev_item = max((i for i in verification_results.keys() if i < missing_item), default=None)
            next_item = min((i for i in verification_results.keys() if i > missing_item), default=None)
            
            print(f"\nPredicting location for missing Item {missing_item}:")
            print(f"Previous item: {prev_item}")
            print(f"Next item: {next_item}")
            
            # Create a standard header text for the missing item
            standard_text = f"ITEM {missing_item}. {self._get_standard_header_text(missing_item)}"
            
            # Try to predict the page using document structure
            pdf_text_by_page = {}
            for page_num in range(1, self.pdf_processor.total_pages + 1):
                pdf_text_by_page[page_num] = self.pdf_processor.get_page_text(page_num)
            
            document_structure = self.nlp_processor.analyze_document_structure(pdf_text_by_page)
            predicted_page = self.nlp_processor.predict_header_page(missing_item, document_structure)
            
            print(f"Predicted page: {predicted_page}")
            
            # Determine if the prediction is reasonable
            reasonable = False
            
            if predicted_page:
                # Check if the prediction is between the previous and next items
                if prev_item and next_item:
                    prev_page = verification_results[prev_item]['expected_page']
                    next_page = verification_results[next_item]['expected_page']
                    reasonable = prev_page <= predicted_page <= next_page
                elif prev_item:
                    prev_page = verification_results[prev_item]['expected_page']
                    reasonable = predicted_page >= prev_page
                elif next_item:
                    next_page = verification_results[next_item]['expected_page']
                    reasonable = predicted_page <= next_page
            
            result = {
                'missing_item': missing_item,
                'standard_text': standard_text,
                'predicted_page': predicted_page,
                'reasonable_prediction': reasonable
            }
            
            results.append(result)
            
            print(f"Reasonable prediction: {'✓' if reasonable else '✗'}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['reasonable_prediction']) / len(results) if results else 0
        print(f"\nMissing headers test success rate: {success_rate:.2%}")
        
        return {
            'success_rate': success_rate,
            'results': results
        }
    
    def test_page_boundary_cases(self):
        """
        Test headers at page boundaries
        
        Returns:
            dict: Test results
        """
        print("\n=== Testing Page Boundary Cases ===")
        
        # Find headers that are at the top or bottom of pages
        boundary_headers = []
        
        for header in self.original_headers:
            item_number = header['item_number']
            header_text = header['text']
            page_number = header['page_number']
            
            page_text = self.pdf_processor.get_page_text(page_number)
            if not page_text:
                continue
            
            # Check if header is at the top of the page (first 20% of text)
            top_text = page_text[:int(len(page_text) * 0.2)]
            at_top = header_text in top_text or re.search(f"ITEM\\s+{item_number}", top_text, re.IGNORECASE)
            
            # Check if header is at the bottom of the page (last 20% of text)
            bottom_text = page_text[int(len(page_text) * 0.8):]
            at_bottom = header_text in bottom_text or re.search(f"ITEM\\s+{item_number}", bottom_text, re.IGNORECASE)
            
            if at_top or at_bottom:
                boundary_headers.append({
                    'item_number': item_number,
                    'text': header_text,
                    'page_number': page_number,
                    'position': 'top' if at_top else 'bottom'
                })
        
        # If no boundary headers found, create artificial ones
        if not boundary_headers:
            for header in self.original_headers[:5]:  # Use first 5 headers
                boundary_headers.append({
                    'item_number': header['item_number'],
                    'text': header['text'],
                    'page_number': header['page_number'],
                    'position': 'top'  # Assume top for artificial cases
                })
        
        # Test each boundary header
        results = []
        
        for header in boundary_headers[:5]:  # Limit to 5 for brevity
            print(f"\nTesting boundary header (Item {header['item_number']}, position: {header['position']}):")
            print(f"Header text: {header['text']}")
            print(f"Page number: {header['page_number']}")
            
            # Verify with pattern matching
            pattern_result = self._verify_with_patterns(header)
            
            # Verify with NLP
            nlp_result = self._verify_with_nlp(header)
            
            # Determine if the test passed
            passed = (pattern_result['verified'] or nlp_result['verified'])
            
            result = {
                'item_number': header['item_number'],
                'position': header['position'],
                'pattern_verified': pattern_result['verified'],
                'pattern_confidence': pattern_result['confidence'],
                'nlp_verified': nlp_result['verified'],
                'nlp_confidence': nlp_result['confidence'],
                'passed': passed
            }
            
            results.append(result)
            
            print(f"Pattern verification: {'✓' if pattern_result['verified'] else '✗'} ({pattern_result['confidence']:.2f})")
            print(f"NLP verification: {'✓' if nlp_result['verified'] else '✗'} ({nlp_result['confidence']:.2f})")
            print(f"Test {'passed' if passed else 'failed'}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['passed']) / len(results) if results else 0
        print(f"\nPage boundary test success rate: {success_rate:.2%}")
        
        return {
            'success_rate': success_rate,
            'results': results
        }
    
    def test_unusual_page_numbering(self):
        """
        Test handling of unusual page numbering
        
        Returns:
            dict: Test results
        """
        print("\n=== Testing Unusual Page Numbering ===")
        
        # Create variations with offset page numbers
        variations = []
        
        # Test with different offsets
        offsets = [-10, -5, 5, 10]
        
        for offset in offsets:
            print(f"\nTesting with page offset: {offset}")
            
            # Apply offset to first 5 headers
            for header in self.original_headers[:5]:
                item_number = header['item_number']
                header_text = header['text']
                original_page = header['page_number']
                
                # Apply offset, ensuring page number stays within valid range
                offset_page = max(1, min(self.pdf_processor.total_pages, original_page + offset))
                
                variations.append({
                    'item_number': item_number,
                    'text': header_text,
                    'original_page': original_page,
                    'offset_page': offset_page,
                    'offset': offset
                })
        
        # Test each variation
        results = []
        
        for variation in variations:
            item_number = variation['item_number']
            header_text = variation['text']
            original_page = variation['original_page']
            offset_page = variation['offset_page']
            
            print(f"\nTesting Item {item_number}:")
            print(f"Original page: {original_page}")
            print(f"Offset page: {offset_page}")
            
            # First verify with the offset page
            offset_result = self.engine._verify_header(item_number, header_text, offset_page)
            
            # Then use NLP to try to find the correct page
            pdf_text_by_page = {}
            for page_num in range(1, self.pdf_processor.total_pages + 1):
                pdf_text_by_page[page_num] = self.pdf_processor.get_page_text(page_num)
            
            nlp_result = self.nlp_processor.verify_header_with_nlp(header_text, offset_page, pdf_text_by_page)
            
            # Check if NLP found the correct page or close to it
            found_page = nlp_result['page_number']
            correct_page_found = found_page and abs(found_page - original_page) <= 2
            
            result = {
                'item_number': item_number,
                'original_page': original_page,
                'offset_page': offset_page,
                'offset': variation['offset'],
                'nlp_found_page': found_page,
                'correct_page_found': correct_page_found
            }
            
            results.append(result)
            
            print(f"NLP found page: {found_page}")
            print(f"Correct page found: {'✓' if correct_page_found else '✗'}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['correct_page_found']) / len(results) if results else 0
        print(f"\nUnusual page numbering test success rate: {success_rate:.2%}")
        
        return {
            'success_rate': success_rate,
            'results': results
        }
    
    def test_low_quality_text(self):
        """
        Test handling of low quality text extraction
        
        Returns:
            dict: Test results
        """
        print("\n=== Testing Low Quality Text ===")
        
        # Simulate low quality text by introducing errors
        variations = []
        
        for header in self.original_headers[:5]:  # Test with first 5 headers
            item_number = header['item_number']
            original_text = header['text']
            expected_page = header['page_number']
            
            # Create variations with different types of errors
            variations.extend([
                {
                    'item_number': item_number,
                    'text': self._introduce_ocr_errors(original_text, error_rate=0.1),
                    'page_number': expected_page,
                    'variation': 'minor_errors'
                },
                {
                    'item_number': item_number,
                    'text': self._introduce_ocr_errors(original_text, error_rate=0.3),
                    'page_number': expected_page,
                    'variation': 'moderate_errors'
                },
                {
                    'item_number': item_number,
                    'text': self._introduce_missing_chars(original_text, missing_rate=0.1),
                    'page_number': expected_page,
                    'variation': 'missing_chars'
                }
            ])
        
        # Test each variation
        results = []
        
        for variation in variations:
            print(f"\nTesting variation: {variation['variation']}")
            print(f"Original: {self.original_headers[variation['item_number']-1]['text']}")
            print(f"Modified: {variation['text']}")
            
            # Verify with pattern matching
            pattern_result = self._verify_with_patterns(variation)
            
            # Verify with NLP
            nlp_result = self._verify_with_nlp(variation)
            
            # Determine if the test passed
            passed = (pattern_result['verified'] or nlp_result['verified'])
            
            result = {
                'item_number': variation['item_number'],
                'variation': variation['variation'],
                'original_text': self.original_headers[variation['item_number']-1]['text'],
                'modified_text': variation['text'],
                'pattern_verified': pattern_result['verified'],
                'pattern_confidence': pattern_result['confidence'],
                'nlp_verified': nlp_result['verified'],
                'nlp_confidence': nlp_result['confidence'],
                'passed': passed
            }
            
            results.append(result)
            
            print(f"Pattern verification: {'✓' if pattern_result['verified'] else '✗'} ({pattern_result['confidence']:.2f})")
            print(f"NLP verification: {'✓' if nlp_result['verified'] else '✗'} ({nlp_result['confidence']:.2f})")
            print(f"Test {'passed' if passed else 'failed'}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['passed']) / len(results) if results else 0
        print(f"\nLow quality text test success rate: {success_rate:.2%}")
        
        return {
            'success_rate': success_rate,
            'results': results
        }
    
    def test_special_characters(self):
        """
        Test headers with special characters
        
        Returns:
            dict: Test results
        """
        print("\n=== Testing Special Characters ===")
        
        # Create variations with special characters
        variations = []
        
        for header in self.original_headers[:5]:  # Test with first 5 headers
            item_number = header['item_number']
            original_text = header['text']
            expected_page = header['page_number']
            
            # Create variations with different special characters
            variations.extend([
                {
                    'item_number': item_number,
                    'text': original_text.replace(' ', '-'),
                    'page_number': expected_page,
                    'variation': 'hyphens'
                },
                {
                    'item_number': item_number,
                    'text': original_text.replace(' ', '_'),
                    'page_number': expected_page,
                    'variation': 'underscores'
                },
                {
                    'item_number': item_number,
                    'text': f"ITEM {item_number} — {original_text.split(f'ITEM {item_number}')[1].strip() if len(original_text.split(f'ITEM {item_number}')) > 1 else ''}",
                    'page_number': expected_page,
                    'variation': 'em_dash'
                },
                {
                    'item_number': item_number,
                    'text': f"ITEM {item_number} • {original_text.split(f'ITEM {item_number}')[1].strip() if len(original_text.split(f'ITEM {item_number}')) > 1 else ''}",
                    'page_number': expected_page,
                    'variation': 'bullet'
                }
            ])
        
        # Test each variation
        results = []
        
        for variation in variations:
            print(f"\nTesting variation: {variation['variation']}")
            print(f"Original: {self.original_headers[variation['item_number']-1]['text']}")
            print(f"Modified: {variation['text']}")
            
            # Verify with pattern matching
            pattern_result = self._verify_with_patterns(variation)
            
            # Verify with NLP
            nlp_result = self._verify_with_nlp(variation)
            
            # Determine if the test passed
            passed = (pattern_result['verified'] or nlp_result['verified'])
            
            result = {
                'item_number': variation['item_number'],
                'variation': variation['variation'],
                'original_text': self.original_headers[variation['item_number']-1]['text'],
                'modified_text': variation['text'],
                'pattern_verified': pattern_result['verified'],
                'pattern_confidence': pattern_result['confidence'],
                'nlp_verified': nlp_result['verified'],
                'nlp_confidence': nlp_result['confidence'],
                'passed': passed
            }
            
            results.append(result)
            
            print(f"Pattern verification: {'✓' if pattern_result['verified'] else '✗'} ({pattern_result['confidence']:.2f})")
            print(f"NLP verification: {'✓' if nlp_result['verified'] else '✗'} ({nlp_result['confidence']:.2f})")
            print(f"Test {'passed' if passed else 'failed'}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['passed']) / len(results) if results else 0
        print(f"\nSpecial characters test success rate: {success_rate:.2%}")
        
        return {
            'success_rate': success_rate,
            'results': results
        }
    
    def _verify_with_patterns(self, header):
        """
        Verify a header using pattern matching
        
        Args:
            header: Header data
            
        Returns:
            dict: Verification result
        """
        item_number = header['item_number']
        header_text = header['text']
        expected_page = header['page_number']
        
        result = self.engine._verify_with_patterns(item_number, header_text, expected_page)
        
        return {
            'verified': result['status'] in ['verified', 'likely_correct'],
            'confidence': result['confidence'],
            'best_match_page': result['best_match_page']
        }
    
    def _verify_with_nlp(self, header):
        """
        Verify a header using NLP
        
        Args:
            header: Header data
            
        Returns:
            dict: Verification result
        """
        item_number = header['item_number']
        header_text = header['text']
        expected_page = header['page_number']
        
        # Get page text dictionary
        pdf_text_by_page = {}
        for page_num in range(1, self.pdf_processor.total_pages + 1):
            pdf_text_by_page[page_num] = self.pdf_processor.get_page_text(page_num)
        
        result = self.nlp_processor.verify_header_with_nlp(header_text, expected_page, pdf_text_by_page)
        
        return {
            'verified': result['verified'],
            'confidence': result['confidence'],
            'best_match_page': result['page_number']
        }
    
    def _introduce_ocr_errors(self, text, error_rate=0.1):
        """
        Introduce OCR-like errors to simulate low quality text
        
        Args:
            text: Original text
            error_rate: Rate of character errors to introduce
            
        Returns:
            str: Text with OCR errors
        """
        chars = list(text)
        num_errors = int(len(chars) * error_rate)
        
        for _ in range(num_errors):
            idx = random.randint(0, len(chars) - 1)
            
            # Skip spaces
            if chars[idx] == ' ':
                continue
            
            error_type = random.choice(['substitute', 'swap'])
            
            if error_type == 'substitute':
                # Common OCR substitutions
                substitutions = {
                    'O': '0', '0': 'O',
                    'I': '1', '1': 'I',
                    'l': '1', '1': 'l',
                    'S': '5', '5': 'S',
                    'B': '8', '8': 'B',
                    'G': '6', '6': 'G',
                    'Z': '2', '2': 'Z'
                }
                
                if chars[idx] in substitutions:
                    chars[idx] = substitutions[chars[idx]]
                else:
                    # Random similar character
                    similar_chars = {
                        'a': 'oa', 'b': 'dh', 'c': 'eo', 'd': 'bp', 'e': 'co',
                        'f': 'ft', 'g': 'qy', 'h': 'bn', 'i': 'jl', 'j': 'iy',
                        'k': 'hx', 'l': 'it', 'm': 'nw', 'n': 'mu', 'o': 'cq',
                        'p': 'bq', 'q': 'gp', 'r': 'fn', 's': 'cz', 't': 'fl',
                        'u': 'vn', 'v': 'uw', 'w': 'vm', 'x': 'ky', 'y': 'gj',
                        'z': 'sx'
                    }
                    
                    char_lower = chars[idx].lower()
                    if char_lower in similar_chars:
                        replacement = random.choice(similar_chars[char_lower])
                        if chars[idx].isupper():
                            replacement = replacement.upper()
                        chars[idx] = replacement
            
            elif error_type == 'swap' and idx < len(chars) - 1:
                # Swap with next character
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        
        return ''.join(chars)
    
    def _introduce_missing_chars(self, text, missing_rate=0.1):
        """
        Introduce missing characters to simulate low quality text
        
        Args:
            text: Original text
            missing_rate: Rate of characters to remove
            
        Returns:
            str: Text with missing characters
        """
        chars = list(text)
        num_missing = int(len(chars) * missing_rate)
        
        for _ in range(num_missing):
            idx = random.randint(0, len(chars) - 1)
            
            # Skip spaces and already removed characters
            if chars[idx] == ' ' or chars[idx] == '':
                continue
            
            chars[idx] = ''
        
        return ''.join(chars)
    
    def _get_standard_header_text(self, item_number):
        """
        Get standard header text for an item number
        
        Args:
            item_number: Item number
            
        Returns:
            str: Standard header text
        """
        standard_texts = {
            1: "THE FRANCHISOR, AND ANY PARENTS, PREDECESSORS, AND AFFILIATES",
            2: "BUSINESS EXPERIENCE",
            3: "LITIGATION",
            4: "BANKRUPTCY",
            5: "INITIAL FEES",
            6: "OTHER FEES",
            7: "ESTIMATED INITIAL INVESTMENT",
            8: "RESTRICTIONS ON SOURCES OF PRODUCTS AND SERVICES",
            9: "FRANCHISEE'S OBLIGATIONS",
            10: "FINANCING",
            11: "FRANCHISOR'S ASSISTANCE, ADVERTISING, COMPUTER SYSTEMS, AND TRAINING",
            12: "TERRITORY",
            13: "TRADEMARKS",
            14: "PATENTS, COPYRIGHTS, AND PROPRIETARY INFORMATION",
            15: "OBLIGATION TO PARTICIPATE IN THE ACTUAL OPERATION OF THE FRANCHISE BUSINESS",
            16: "RESTRICTIONS ON WHAT THE FRANCHISEE MAY SELL",
            17: "RENEWAL, TERMINATION, TRANSFER, AND DISPUTE RESOLUTION",
            18: "PUBLIC FIGURES",
            19: "FINANCIAL PERFORMANCE REPRESENTATIONS",
            20: "OUTLETS AND FRANCHISEE INFORMATION",
            21: "FINANCIAL STATEMENTS",
            22: "CONTRACTS",
            23: "RECEIPTS"
        }
        
        return standard_texts.get(item_number, f"ITEM {item_number}")


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
    
    # Run edge case tests on the first file pair
    if os.path.exists(pdf_files[0]) and os.path.exists(json_files[0]):
        print(f"\n{'='*50}")
        print(f"Running edge case tests on: {os.path.basename(pdf_files[0])}")
        
        test_suite = EdgeCaseTestSuite(pdf_files[0], json_files[0])
        results = test_suite.run_all_tests()
        
        # Print overall summary
        print("\n=== Edge Case Testing Summary ===")
        for test_name, test_result in results.items():
            print(f"{test_name}: {test_result['success_rate']:.2%}")
        
        overall_success = sum(r['success_rate'] for r in results.values()) / len(results)
        print(f"\nOverall success rate: {overall_success:.2%}")
        print(f"{'='*50}\n")
    else:
        print("Test files not found. Skipping edge case tests.")
