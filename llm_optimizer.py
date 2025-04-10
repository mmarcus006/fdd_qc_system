import os
import json
import requests
from typing import Dict, List, Any, Optional, Union

class LLMOptimizer:
    """
    Class for optimizing LLM API usage for header verification
    """
    
    def __init__(self, api_key=None, api_url=None):
        """
        Initialize the LLM optimizer
        
        Args:
            api_key (str): API key for the LLM service
            api_url (str): URL for the LLM API endpoint
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.api_url = api_url or "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.call_count = 0
        self.max_calls_per_session = 10  # Limit API calls for cost efficiency
        self.cache = {}  # Simple cache to avoid duplicate API calls
        
    def batch_verify_headers(self, headers_data: List[Dict[str, Any]], pdf_text_by_page: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Batch verify multiple headers at once to reduce API calls
        
        Args:
            headers_data: List of header data dictionaries with item_number, header_text, expected_page
            pdf_text_by_page: Dictionary mapping page numbers to page text
            
        Returns:
            List of verification results
        """
        # Group headers by expected page to minimize API calls
        headers_by_page = {}
        for header in headers_data:
            page = header.get('expected_page')
            if page not in headers_by_page:
                headers_by_page[page] = []
            headers_by_page[page].append(header)
        
        results = []
        
        # Process each page with its headers
        for page_num, page_headers in headers_by_page.items():
            # Skip if we've exceeded the maximum number of API calls
            if self.call_count >= self.max_calls_per_session:
                for header in page_headers:
                    results.append({
                        "item_number": header.get('item_number'),
                        "verified": False,
                        "confidence": 0.0,
                        "explanation": "Maximum number of LLM API calls exceeded",
                        "page_number": None
                    })
                continue
            
            # Get page text
            page_text = pdf_text_by_page.get(page_num, "")
            if not page_text:
                for header in page_headers:
                    results.append({
                        "item_number": header.get('item_number'),
                        "verified": False,
                        "confidence": 0.0,
                        "explanation": "Page text not available",
                        "page_number": None
                    })
                continue
            
            # Create a batch prompt for all headers on this page
            batch_result = self._verify_headers_batch(page_headers, page_text, page_num)
            results.extend(batch_result)
            
        return results
    
    def _verify_headers_batch(self, headers: List[Dict[str, Any]], page_text: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Verify multiple headers on the same page with a single API call
        
        Args:
            headers: List of header data dictionaries
            page_text: Text content of the page
            page_num: Page number
            
        Returns:
            List of verification results
        """
        # Create a cache key based on headers and page
        cache_key = f"{page_num}_{','.join([str(h.get('item_number')) for h in headers])}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Increment call count
        self.call_count += 1
        
        # Prepare the prompt
        headers_json = json.dumps([{
            "item_number": h.get('item_number'),
            "header_text": h.get('header_text')
        } for h in headers])
        
        prompt = f"""
        Task: Verify if the following FDD (Franchise Disclosure Document) headers appear on the provided page text.
        
        Headers: {headers_json}
        Page Number: {page_num}
        
        Page Text (first 2000 characters):
        {page_text[:2000]}
        
        For each header, analyze if it appears on this page. Respond in JSON format with an array of results, each with the following fields:
        - item_number: the item number from the input
        - verified: true/false
        - confidence: a number between 0 and 1
        - explanation: brief explanation of your decision
        - page_number: the page number where you think this header appears, or null if not found
        
        Example response format:
        [
          {
            "item_number": 1,
            "verified": true,
            "confidence": 0.95,
            "explanation": "Header text found at the beginning of the page",
            "page_number": 10
          },
          {
            "item_number": 2,
            "verified": false,
            "confidence": 0.2,
            "explanation": "Header text not found on this page",
            "page_number": null
          }
        ]
        """
        
        # If no API key is available, use mock responses
        if not self.api_key:
            print("No API key available, using mock LLM responses")
            results = self._mock_batch_response(headers, page_text, page_num)
            self.cache[cache_key] = results
            return results
        
        try:
            # Make the API call
            headers_dict = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers_dict,
                json=data
            )
            
            if response.status_code != 200:
                print(f"LLM API error: {response.status_code} - {response.text}")
                results = self._mock_batch_response(headers, page_text, page_num)
                self.cache[cache_key] = results
                return results
            
            # Parse the response
            response_data = response.json()
            response_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    results = json.loads(json_match.group(0))
                else:
                    # Fallback to parsing the entire response
                    results = json.loads(response_text)
                
                # Cache the results
                self.cache[cache_key] = results
                return results
            
            except json.JSONDecodeError:
                print(f"Failed to parse LLM response as JSON: {response_text}")
                results = self._mock_batch_response(headers, page_text, page_num)
                self.cache[cache_key] = results
                return results
        
        except Exception as e:
            print(f"Error calling LLM API: {str(e)}")
            results = self._mock_batch_response(headers, page_text, page_num)
            self.cache[cache_key] = results
            return results
    
    def _mock_batch_response(self, headers: List[Dict[str, Any]], page_text: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Generate mock LLM responses for a batch of headers
        
        Args:
            headers: List of header data dictionaries
            page_text: Text content of the page
            page_num: Page number
            
        Returns:
            List of mock verification results
        """
        results = []
        page_text_lower = page_text.lower()
        
        for header in headers:
            item_number = header.get('item_number')
            header_text = header.get('header_text', '')
            
            # Simple heuristic: check if the header text appears in the page text
            if header_text.lower() in page_text_lower:
                results.append({
                    "item_number": item_number,
                    "verified": True,
                    "confidence": 0.95,
                    "explanation": "Header text found in page content",
                    "page_number": page_num
                })
            else:
                # Check if parts of the header appear
                header_parts = header_text.split()
                matches = sum(1 for part in header_parts if part.lower() in page_text_lower)
                match_ratio = matches / len(header_parts) if header_parts else 0
                
                if match_ratio > 0.7:
                    results.append({
                        "item_number": item_number,
                        "verified": True,
                        "confidence": 0.8,
                        "explanation": "Most header words found in page content",
                        "page_number": page_num
                    })
                else:
                    results.append({
                        "item_number": item_number,
                        "verified": False,
                        "confidence": 0.2,
                        "explanation": "Header text not found in page content",
                        "page_number": None
                    })
        
        return results
    
    def prioritize_headers_for_llm(self, verification_results: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize which headers should be verified with LLM based on previous verification results
        
        Args:
            verification_results: Dictionary mapping item numbers to verification results
            
        Returns:
            List of headers that should be verified with LLM
        """
        headers_for_llm = []
        
        for item_number, result in verification_results.items():
            # Only use LLM for headers that couldn't be verified with high confidence
            if result.get('confidence', 0) < 0.7 or result.get('status') in ['needs_review', 'likely_incorrect', 'not_found']:
                headers_for_llm.append({
                    'item_number': item_number,
                    'header_text': result.get('header_text', ''),
                    'expected_page': result.get('expected_page'),
                    'best_match_page': result.get('best_match_page'),
                    'current_confidence': result.get('confidence', 0)
                })
        
        # Sort by confidence (lowest first)
        headers_for_llm.sort(key=lambda x: x.get('current_confidence', 0))
        
        return headers_for_llm
