import os
import sys
import json
import torch
from typing import Dict, List, Any, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_processor import PDFProcessor, JSONProcessor
from enhanced_verification import EnhancedVerificationEngine, HeaderDatabase
from llm_optimizer import LLMOptimizer
from advanced_nlp import AdvancedNLPProcessor

def test_enhanced_verification(pdf_path, json_path):
    """
    Test the enhanced verification system with a PDF and JSON file
    
    Args:
        pdf_path (str): Path to the PDF file
        json_path (str): Path to the JSON file
        
    Returns:
        dict: Verification results
    """
    print(f"Testing enhanced verification with PDF: {os.path.basename(pdf_path)}")
    print(f"JSON: {os.path.basename(json_path)}")
    
    # Initialize processors
    pdf_processor = PDFProcessor(pdf_path)
    json_processor = JSONProcessor(json_path)
    
    # Initialize verification engine
    engine = EnhancedVerificationEngine(pdf_processor, json_processor)
    
    # Initialize NLP processor
    nlp_processor = AdvancedNLPProcessor()
    
    # Run verification
    print("Running pattern-based verification...")
    results = engine.verify_all_headers()
    
    # Get summary
    summary = engine.get_verification_summary()
    print("\nPattern-based Verification Summary:")
    print(f"Total headers: {summary['total']}")
    print(f"Verified: {summary['verified']}")
    print(f"Likely correct: {summary['likely_correct']}")
    print(f"Needs review: {summary['needs_review']}")
    print(f"Likely incorrect: {summary['likely_incorrect']}")
    print(f"Not found: {summary['not_found']}")
    
    # Get headers that need review
    needs_review = engine.get_headers_by_status("needs_review")
    likely_incorrect = engine.get_headers_by_status("likely_incorrect")
    not_found = engine.get_headers_by_status("not_found")
    
    problem_headers = needs_review + likely_incorrect + not_found
    
    if problem_headers:
        print(f"\nFound {len(problem_headers)} headers that need additional verification")
        
        # Create a dictionary of page text for NLP processing
        pdf_text_by_page = {}
        for page_num in range(1, pdf_processor.total_pages + 1):
            pdf_text_by_page[page_num] = pdf_processor.get_page_text(page_num)
        
        # Process each problem header with advanced NLP
        print("\nApplying advanced NLP techniques...")
        for header in problem_headers:
            item_number = header['item_number']
            header_text = header['header_text']
            expected_page = header['expected_page']
            
            print(f"\nProcessing Item {item_number}: {header_text}")
            print(f"  Initial verification: {header['status']} (confidence: {header['confidence']:.2f})")
            
            # Try NLP verification
            nlp_result = nlp_processor.verify_header_with_nlp(header_text, expected_page, pdf_text_by_page)
            
            print(f"  NLP verification: {'verified' if nlp_result['verified'] else 'not verified'} "
                  f"(confidence: {nlp_result['confidence']:.2f}, method: {nlp_result['method']})")
            print(f"  Suggested page: {nlp_result['page_number']}")
            
            # Update result if NLP has higher confidence
            if nlp_result['confidence'] > header['confidence']:
                results[item_number]['confidence'] = nlp_result['confidence']
                results[item_number]['best_match_page'] = nlp_result['page_number']
                results[item_number]['method'] = f"nlp_{nlp_result['method']}"
                
                if nlp_result['verified']:
                    if nlp_result['confidence'] > 0.9:
                        results[item_number]['status'] = "verified"
                    elif nlp_result['confidence'] > 0.8:
                        results[item_number]['status'] = "likely_correct"
                    else:
                        results[item_number]['status'] = "needs_review"
        
        # Get remaining problem headers for LLM verification
        remaining_problems = []
        for item_number, result in results.items():
            if result['status'] in ["needs_review", "likely_incorrect", "not_found"] and result['confidence'] < 0.7:
                remaining_problems.append({
                    'item_number': item_number,
                    'header_text': result['header_text'],
                    'expected_page': result['expected_page']
                })
        
        if remaining_problems:
            print(f"\nFound {len(remaining_problems)} headers that still need verification")
            print("These would be candidates for selective LLM verification in production")
            
            # Initialize LLM optimizer (but don't actually call API in test)
            llm_optimizer = LLMOptimizer()
            
            # Prioritize headers for LLM verification
            prioritized = llm_optimizer.prioritize_headers_for_llm(results)
            
            print("\nPrioritized headers for LLM verification:")
            for header in prioritized[:3]:  # Show top 3
                print(f"  Item {header['item_number']}: {header['header_text']}")
                print(f"    Current confidence: {header['current_confidence']:.2f}")
    
    # Get updated summary
    updated_verified = sum(1 for r in results.values() if r['status'] == "verified")
    updated_likely = sum(1 for r in results.values() if r['status'] == "likely_correct")
    updated_needs_review = sum(1 for r in results.values() if r['status'] == "needs_review")
    updated_incorrect = sum(1 for r in results.values() if r['status'] == "likely_incorrect")
    updated_not_found = sum(1 for r in results.values() if r['status'] == "not_found")
    
    print("\nEnhanced Verification Summary:")
    print(f"Total headers: {summary['total']}")
    print(f"Verified: {updated_verified} (was {summary['verified']})")
    print(f"Likely correct: {updated_likely} (was {summary['likely_correct']})")
    print(f"Needs review: {updated_needs_review} (was {summary['needs_review']})")
    print(f"Likely incorrect: {updated_incorrect} (was {summary['likely_incorrect']})")
    print(f"Not found: {updated_not_found} (was {summary['not_found']})")
    
    improvement = (updated_verified + updated_likely) - (summary['verified'] + summary['likely_correct'])
    print(f"\nImprovement: {improvement} more headers verified or likely correct")
    
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
            test_enhanced_verification(pdf_files[i], json_files[i])
            print(f"{'='*50}\n")
        else:
            print(f"File pair {i+1} not found. Skipping.")
