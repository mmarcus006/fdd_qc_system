import os
import json
import re
import sqlite3
import pickle
from difflib import SequenceMatcher
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import requests
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class HeaderDatabase:
    """
    Database for storing and retrieving header corrections and patterns
    """
    
    def __init__(self, db_path="header_database.db"):
        """
        Initialize the header database
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize the database schema if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS header_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_number INTEGER,
            header_text TEXT,
            original_page INTEGER,
            corrected_page INTEGER,
            pdf_name TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS header_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_number INTEGER,
            pattern TEXT,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            last_used DATETIME
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS embedding_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            header_text TEXT UNIQUE,
            embedding BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_correction(self, item_number, header_text, original_page, corrected_page, pdf_name, confidence=1.0):
        """
        Add a header correction to the database
        
        Args:
            item_number (int): Item number (1-23)
            header_text (str): Header text
            original_page (int): Original page number
            corrected_page (int): Corrected page number
            pdf_name (str): Name of the PDF file
            confidence (float): Confidence score for the correction
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO header_corrections 
        (item_number, header_text, original_page, corrected_page, pdf_name, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (item_number, header_text, original_page, corrected_page, pdf_name, confidence))
        
        conn.commit()
        conn.close()
    
    def get_corrections_for_item(self, item_number):
        """
        Get all corrections for a specific item number
        
        Args:
            item_number (int): Item number (1-23)
            
        Returns:
            list: List of correction records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM header_corrections
        WHERE item_number = ?
        ORDER BY timestamp DESC
        ''', (item_number,))
        
        corrections = cursor.fetchall()
        conn.close()
        
        return corrections
    
    def add_pattern(self, item_number, pattern):
        """
        Add a header pattern to the database
        
        Args:
            item_number (int): Item number (1-23)
            pattern (str): Regex pattern for the header
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if pattern already exists
        cursor.execute('''
        SELECT id FROM header_patterns
        WHERE item_number = ? AND pattern = ?
        ''', (item_number, pattern))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update last_used timestamp
            cursor.execute('''
            UPDATE header_patterns
            SET last_used = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (existing[0],))
        else:
            # Insert new pattern
            cursor.execute('''
            INSERT INTO header_patterns 
            (item_number, pattern, last_used)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (item_number, pattern))
        
        conn.commit()
        conn.close()
    
    def update_pattern_success(self, item_number, pattern):
        """
        Increment success count for a pattern
        
        Args:
            item_number (int): Item number (1-23)
            pattern (str): Regex pattern for the header
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE header_patterns
        SET success_count = success_count + 1, last_used = CURRENT_TIMESTAMP
        WHERE item_number = ? AND pattern = ?
        ''', (item_number, pattern))
        
        conn.commit()
        conn.close()
    
    def update_pattern_failure(self, item_number, pattern):
        """
        Increment failure count for a pattern
        
        Args:
            item_number (int): Item number (1-23)
            pattern (str): Regex pattern for the header
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE header_patterns
        SET failure_count = failure_count + 1, last_used = CURRENT_TIMESTAMP
        WHERE item_number = ? AND pattern = ?
        ''', (item_number, pattern))
        
        conn.commit()
        conn.close()
    
    def get_patterns_for_item(self, item_number):
        """
        Get all patterns for a specific item number, ordered by success rate
        
        Args:
            item_number (int): Item number (1-23)
            
        Returns:
            list: List of pattern records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, pattern, success_count, failure_count 
        FROM header_patterns
        WHERE item_number = ?
        ORDER BY (success_count * 1.0 / (success_count + failure_count + 0.1)) DESC, last_used DESC
        ''', (item_number,))
        
        patterns = cursor.fetchall()
        conn.close()
        
        return patterns
    
    def store_embedding(self, header_text, embedding):
        """
        Store a header text embedding in the database
        
        Args:
            header_text (str): Header text
            embedding (numpy.ndarray): Vector embedding of the header text
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize the embedding
        embedding_blob = pickle.dumps(embedding)
        
        # Check if header text already exists
        cursor.execute('''
        SELECT id FROM embedding_cache
        WHERE header_text = ?
        ''', (header_text,))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing embedding
            cursor.execute('''
            UPDATE embedding_cache
            SET embedding = ?, timestamp = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (embedding_blob, existing[0]))
        else:
            # Insert new embedding
            cursor.execute('''
            INSERT INTO embedding_cache 
            (header_text, embedding)
            VALUES (?, ?)
            ''', (header_text, embedding_blob))
        
        conn.commit()
        conn.close()
    
    def get_embedding(self, header_text):
        """
        Get a stored embedding for a header text
        
        Args:
            header_text (str): Header text
            
        Returns:
            numpy.ndarray or None: Vector embedding if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT embedding FROM embedding_cache
        WHERE header_text = ?
        ''', (header_text,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Deserialize the embedding
            return pickle.loads(result[0])
        
        return None
    
    def get_all_embeddings(self):
        """
        Get all stored embeddings
        
        Returns:
            dict: Dictionary mapping header texts to embeddings
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT header_text, embedding FROM embedding_cache
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        embeddings = {}
        for header_text, embedding_blob in results:
            embeddings[header_text] = pickle.loads(embedding_blob)
        
        return embeddings


class TransformerEmbedder:
    """
    Class for generating text embeddings using pre-trained transformer models
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the transformer embedder
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        except Exception as e:
            print(f"Error loading transformer model: {str(e)}")
            # Fallback to a simpler model if the specified one fails
            try:
                self.model_name = "distilbert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            except Exception as e:
                print(f"Error loading fallback model: {str(e)}")
                raise
    
    def get_embedding(self, text):
        """
        Generate an embedding for the given text
        
        Args:
            text (str): Input text
            
        Returns:
            numpy.ndarray: Vector embedding of the input text
        """
        if not self.tokenizer or not self.model:
            raise ValueError("Model not loaded")
        
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling to get a single vector for the entire text
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convert to numpy array and return
        return embeddings.cpu().numpy()[0]
    
    def compute_similarity(self, text1, text2):
        """
        Compute cosine similarity between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        return float(similarity)


class LLMVerifier:
    """
    Class for verifying headers using LLM API calls (for difficult cases only)
    """
    
    def __init__(self, api_key=None, api_url=None):
        """
        Initialize the LLM verifier
        
        Args:
            api_key (str): API key for the LLM service
            api_url (str): URL for the LLM API endpoint
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.api_url = api_url or "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.call_count = 0
        self.max_calls_per_session = 10  # Limit API calls for cost efficiency
    
    def verify_header(self, header_text, page_text, expected_page, item_number):
        """
        Verify a header using the LLM API
        
        Args:
            header_text (str): Header text to verify
            page_text (str): Text content of the page
            expected_page (int): Expected page number
            item_number (int): Item number (1-23)
            
        Returns:
            dict: Verification result with confidence score and explanation
        """
        # Check if we've exceeded the maximum number of API calls
        if self.call_count >= self.max_calls_per_session:
            return {
                "verified": False,
                "confidence": 0.0,
                "explanation": "Maximum number of LLM API calls exceeded",
                "page_number": None
            }
        
        # Increment call count
        self.call_count += 1
        
        # Prepare the prompt
        prompt = f"""
        Task: Verify if the following FDD (Franchise Disclosure Document) header appears on the provided page text.
        
        Header: {header_text}
        Item Number: {item_number}
        Expected Page Number: {expected_page}
        
        Page Text (first 1000 characters):
        {page_text[:1000]}
        
        Please analyze if this header appears on this page. Respond in JSON format with the following fields:
        - verified: true/false
        - confidence: a number between 0 and 1
        - explanation: brief explanation of your decision
        - page_number: the page number where you think this header appears, or null if not found
        """
        
        # If no API key is available, use a mock response
        if not self.api_key:
            print("No API key available, using mock LLM response")
            return self._mock_llm_response(header_text, page_text, expected_page)
        
        try:
            # Make the API call
            headers = {
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
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                print(f"LLM API error: {response.status_code} - {response.text}")
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "explanation": f"API error: {response.status_code}",
                    "page_number": None
                }
            
            # Parse the response
            response_data = response.json()
            response_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    # Fallback to parsing the entire response
                    result = json.loads(response_text)
                
                return {
                    "verified": result.get("verified", False),
                    "confidence": result.get("confidence", 0.0),
                    "explanation": result.get("explanation", "No explanation provided"),
                    "page_number": result.get("page_number")
                }
            except json.JSONDecodeError:
                print(f"Failed to parse LLM response as JSON: {response_text}")
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "explanation": "Failed to parse LLM response",
                    "page_number": None
                }
        
        except Exception as e:
            print(f"Error calling LLM API: {str(e)}")
            return {
                "verified": False,
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "page_number": None
            }
    
    def _mock_llm_response(self, header_text, page_text, expected_page):
        """
        Generate a mock LLM response for testing without API calls
        
        Args:
            header_text (str): Header text to verify
            page_text (str): Text content of the page
            expected_page (int): Expected page number
            
        Returns:
            dict: Mock verification result
        """
        # Simple heuristic: check if the header text appears in the page text
        if header_text.lower() in page_text.lower():
            return {
                "verified": True,
                "confidence": 0.95,
                "explanation": "Header text found in page content",
                "page_number": expected_page
            }
        else:
            # Check if parts of the header appear
            header_words = set(word_tokenize(header_text.lower()))
            page_words = set(word_tokenize(page_text.lower()))
            common_words = header_words.intersection(page_words)
            
            if len(common_words) / len(header_words) > 0.7:
                return {
                    "verified": True,
                    "confidence": 0.8,
                    "explanation": "Most header words found in page content",
                    "page_number": expected_page
                }
            else:
                return {
                    "verified": False,
                    "confidence": 0.2,
                    "explanation": "Header text not found in page content",
                    "page_number": None
                }


class EnhancedVerificationEngine:
    """
    Enhanced engine for verifying FDD headers against PDF content
    """
    
    def __init__(self, pdf_processor, json_processor):
        """
        Initialize the enhanced verification engine
        
        Args:
            pdf_processor: PDF processor instance
            json_processor: JSON processor instance
        """
        self.pdf_processor = pdf_processor
        self.json_processor = json_processor
        self.verification_results = {}
        
        # Initialize components
        self.header_db = HeaderDatabase()
        self.transformer = TransformerEmbedder()
        self.llm_verifier = LLMVerifier()
        
        # Standard FDD header patterns
        self.standard_patterns = {
            1: r'ITEM\s+1\.?\s+THE\s+FRANCHISOR',
            2: r'ITEM\s+2\.?\s+BUSINESS\s+EXPERIENCE',
            3: r'ITEM\s+3\.?\s+LITIGATION',
            4: r'ITEM\s+4\.?\s+BANKRUPTCY',
            5: r'ITEM\s+5\.?\s+INITIAL\s+FEES',
            6: r'ITEM\s+6\.?\s+OTHER\s+FEES',
            7: r'ITEM\s+7\.?\s+(ESTIMATED\s+)?INITIAL\s+INVESTMENT',
            8: r'ITEM\s+8\.?\s+RESTRICTIONS\s+ON\s+(SOURCES\s+OF\s+)?PRODUCTS',
            9: r'ITEM\s+9\.?\s+FRANCHISEE\'?S?\s+OBLIGATIONS',
            10: r'ITEM\s+10\.?\s+FINANCING',
            11: r'ITEM\s+11\.?\s+FRANCHISOR\'?S?\s+(ASSISTANCE|OBLIGATIONS)',
            12: r'ITEM\s+12\.?\s+TERRITORY',
            13: r'ITEM\s+13\.?\s+TRADEMARKS',
            14: r'ITEM\s+14\.?\s+(PATENTS|COPYRIGHTS)',
            15: r'ITEM\s+15\.?\s+(OBLIGATION\s+TO\s+)?PARTICIPAT',
            16: r'ITEM\s+16\.?\s+RESTRICTIONS?\s+ON\s+WHAT',
            17: r'ITEM\s+17\.?\s+(RENEWAL|TERMINATION)',
            18: r'ITEM\s+18\.?\s+PUBLIC\s+FIGURES',
            19: r'ITEM\s+19\.?\s+FINANCIAL\s+PERFORMANCE',
            20: r'ITEM\s+20\.?\s+OUTLETS',
            21: r'ITEM\s+21\.?\s+FINANCIAL\s+STATEMENTS',
            22: r'ITEM\s+22\.?\s+CONTRACTS',
            23: r'ITEM\s+23\.?\s+RECEIPTS?'
        }
    
    def verify_all_headers(self):
        """
        Verify all headers in the JSON against the PDF using enhanced methods
        
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
        Verify a single header against the PDF using multiple methods
        
        Args:
            item_number (int): Item number
            header_text (str): Header text
            expected_page (int): Expected page number
            
        Returns:
            dict: Verification result
        """
        # Step 1: Try pattern-based verification first (fastest)
        pattern_result = self._verify_with_patterns(item_number, header_text, expected_page)
        
        # If high confidence, return the result
        if pattern_result.get('confidence', 0) > 0.9:
            return pattern_result
        
        # Step 2: Try transformer-based verification
        transformer_result = self._verify_with_transformer(item_number, header_text, expected_page)
        
        # If high confidence, return the result
        if transformer_result.get('confidence', 0) > 0.85:
            return transformer_result
        
        # Step 3: For difficult cases, use LLM verification as a last resort
        if pattern_result.get('confidence', 0) < 0.6 and transformer_result.get('confidence', 0) < 0.7:
            llm_result = self._verify_with_llm(item_number, header_text, expected_page)
            
            # If LLM verification is successful, return that result
            if llm_result.get('confidence', 0) > 0.8:
                return llm_result
        
        # Return the best result from pattern and transformer methods
        if pattern_result.get('confidence', 0) >= transformer_result.get('confidence', 0):
            return pattern_result
        else:
            return transformer_result
    
    def _verify_with_patterns(self, item_number, header_text, expected_page):
        """
        Verify a header using pattern matching
        
        Args:
            item_number (int): Item number
            header_text (str): Header text
            expected_page (int): Expected page number
            
        Returns:
            dict: Verification result
        """
        # Get patterns from database
        db_patterns = self.header_db.get_patterns_for_item(item_number)
        patterns = [p[1] for p in db_patterns]
        
        # Add standard pattern if available
        if item_number in self.standard_patterns:
            patterns.append(self.standard_patterns[item_number])
        
        # Add a pattern based on the header text itself
        try:
            # Extract the part after "ITEM X" or "ITEM X."
            header_parts = re.split(r'ITEM\s+\d+\.?\s+', header_text, 1)
            if len(header_parts) > 1:
                content_part = header_parts[1]
                # Create a pattern that allows for variations
                dynamic_pattern = r'ITEM\s+' + str(item_number) + r'\.?\s+' + re.escape(content_part)
                patterns.append(dynamic_pattern)
        except Exception as e:
            print(f"Error creating dynamic pattern for header {item_number}: {str(e)}")
        
        # Make patterns unique
        patterns = list(set(patterns))
        
        # Search for patterns in the PDF
        found_pages = {}
        best_pattern = None
        
        # First, check the expected page and nearby pages
        window_size = 5
        start = max(1, expected_page - window_size)
        end = min(self.pdf_processor.total_pages, expected_page + window_size)
        
        # Try each pattern
        for pattern in patterns:
            for page_num in range(start, end + 1):
                page_text = self.pdf_processor.get_page_text(page_num)
                
                # Try to match the pattern
                try:
                    matches = re.finditer(pattern, page_text, re.IGNORECASE)
                    for match in matches:
                        found_text = match.group(0)
                        similarity = SequenceMatcher(None, header_text, found_text).ratio()
                        
                        if page_num not in found_pages or similarity > found_pages[page_num]['confidence']:
                            found_pages[page_num] = {
                                'confidence': similarity,
                                'distance_from_expected': abs(page_num - expected_page),
                                'pattern': pattern
                            }
                            
                            if not best_pattern or similarity > found_pages[best_pattern[0]]['confidence']:
                                best_pattern = (page_num, pattern)
                except Exception as e:
                    print(f"Error matching pattern '{pattern}': {str(e)}")
        
        # If no pages found in the window, search the entire PDF
        if not found_pages:
            for pattern in patterns:
                for page_num in range(1, self.pdf_processor.total_pages + 1):
                    if page_num in range(start, end + 1):
                        continue  # Skip pages we've already checked
                    
                    page_text = self.pdf_processor.get_page_text(page_num)
                    
                    # Try to match the pattern
                    try:
                        matches = re.finditer(pattern, page_text, re.IGNORECASE)
                        for match in matches:
                            found_text = match.group(0)
                            similarity = SequenceMatcher(None, header_text, found_text).ratio()
                            
                            if page_num not in found_pages or similarity > found_pages[page_num]['confidence']:
                                found_pages[page_num] = {
                                    'confidence': similarity,
                                    'distance_from_expected': abs(page_num - expected_page),
                                    'pattern': pattern
                                }
                                
                                if not best_pattern or similarity > found_pages[best_pattern[0]]['confidence']:
                                    best_pattern = (page_num, pattern)
                    except Exception as e:
                        print(f"Error matching pattern '{pattern}': {str(e)}")
        
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
            
            # Update pattern success/failure in database
            if best_pattern:
                if confidence > 0.8 and page_num == expected_page:
                    self.header_db.update_pattern_success(item_number, best_pattern[1])
                    # Also add the pattern to the database if it's not already there
                    self.header_db.add_pattern(item_number, best_pattern[1])
                elif confidence < 0.5:
                    self.header_db.update_pattern_failure(item_number, best_pattern[1])
            
            # Determine status based on confidence and match with expected page
            if page_num == expected_page and confidence > 0.9:
                status = "verified"
            elif confidence > 0.8:
                status = "likely_correct"
            elif confidence > 0.6:
                status = "needs_review"
            else:
                status = "likely_incorrect"
        
        return {
            'item_number': item_number,
            'header_text': header_text,
            'expected_page': expected_page,
            'found_pages': found_pages,
            'best_match_page': best_page[0] if best_page else None,
            'confidence': confidence,
            'status': status,
            'method': 'pattern_matching',
            'best_pattern': best_pattern[1] if best_pattern else None
        }
    
    def _verify_with_transformer(self, item_number, header_text, expected_page):
        """
        Verify a header using transformer embeddings
        
        Args:
            item_number (int): Item number
            header_text (str): Header text
            expected_page (int): Expected page number
            
        Returns:
            dict: Verification result
        """
        # Get embedding for the header
        header_embedding = self.header_db.get_embedding(header_text)
        
        if header_embedding is None:
            try:
                header_embedding = self.transformer.get_embedding(header_text)
                self.header_db.store_embedding(header_text, header_embedding)
            except Exception as e:
                print(f"Error generating embedding for header: {str(e)}")
                return {
                    'item_number': item_number,
                    'header_text': header_text,
                    'expected_page': expected_page,
                    'found_pages': {},
                    'best_match_page': None,
                    'confidence': 0,
                    'status': "error",
                    'method': 'transformer',
                    'error': str(e)
                }
        
        # Search for similar text in the PDF
        found_pages = {}
        
        # First, check the expected page and nearby pages
        window_size = 5
        start = max(1, expected_page - window_size)
        end = min(self.pdf_processor.total_pages, expected_page + window_size)
        
        for page_num in range(start, end + 1):
            page_text = self.pdf_processor.get_page_text(page_num)
            
            # Skip empty pages
            if not page_text:
                continue
            
            # For efficiency, first check if the item number appears in the page
            item_pattern = f"ITEM\\s+{item_number}\\b"
            if not re.search(item_pattern, page_text, re.IGNORECASE):
                continue
            
            # Extract potential header sections (paragraphs starting with "ITEM")
            header_sections = re.findall(r'(ITEM\s+\d+\.?.*?)(?=\n\n|\Z)', page_text, re.IGNORECASE | re.DOTALL)
            
            if not header_sections:
                # If no clear sections, use the first 500 characters
                header_sections = [page_text[:500]]
            
            # Compare each section with the header
            for section in header_sections:
                try:
                    section_embedding = self.transformer.get_embedding(section)
                    similarity = np.dot(header_embedding, section_embedding) / (np.linalg.norm(header_embedding) * np.linalg.norm(section_embedding))
                    
                    if page_num not in found_pages or similarity > found_pages[page_num]['confidence']:
                        found_pages[page_num] = {
                            'confidence': float(similarity),
                            'distance_from_expected': abs(page_num - expected_page),
                            'section': section
                        }
                except Exception as e:
                    print(f"Error comparing embeddings for page {page_num}: {str(e)}")
        
        # If no pages found in the window, search a wider range
        if not found_pages:
            # Expand search to +/- 10 pages
            wider_start = max(1, expected_page - 10)
            wider_end = min(self.pdf_processor.total_pages, expected_page + 10)
            
            for page_num in range(wider_start, wider_end + 1):
                if page_num in range(start, end + 1):
                    continue  # Skip pages we've already checked
                
                page_text = self.pdf_processor.get_page_text(page_num)
                
                # Skip empty pages
                if not page_text:
                    continue
                
                # For efficiency, first check if the item number appears in the page
                item_pattern = f"ITEM\\s+{item_number}\\b"
                if not re.search(item_pattern, page_text, re.IGNORECASE):
                    continue
                
                # Extract potential header sections
                header_sections = re.findall(r'(ITEM\s+\d+\.?.*?)(?=\n\n|\Z)', page_text, re.IGNORECASE | re.DOTALL)
                
                if not header_sections:
                    # If no clear sections, use the first 500 characters
                    header_sections = [page_text[:500]]
                
                # Compare each section with the header
                for section in header_sections:
                    try:
                        section_embedding = self.transformer.get_embedding(section)
                        similarity = np.dot(header_embedding, section_embedding) / (np.linalg.norm(header_embedding) * np.linalg.norm(section_embedding))
                        
                        if page_num not in found_pages or similarity > found_pages[page_num]['confidence']:
                            found_pages[page_num] = {
                                'confidence': float(similarity),
                                'distance_from_expected': abs(page_num - expected_page),
                                'section': section
                            }
                    except Exception as e:
                        print(f"Error comparing embeddings for page {page_num}: {str(e)}")
        
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
        
        return {
            'item_number': item_number,
            'header_text': header_text,
            'expected_page': expected_page,
            'found_pages': found_pages,
            'best_match_page': best_page[0] if best_page else None,
            'confidence': confidence,
            'status': status,
            'method': 'transformer'
        }
    
    def _verify_with_llm(self, item_number, header_text, expected_page):
        """
        Verify a header using LLM API (for difficult cases only)
        
        Args:
            item_number (int): Item number
            header_text (str): Header text
            expected_page (int): Expected page number
            
        Returns:
            dict: Verification result
        """
        # Get the page text
        page_text = self.pdf_processor.get_page_text(expected_page) or ""
        
        # Call the LLM verifier
        llm_result = self.llm_verifier.verify_header(header_text, page_text, expected_page, item_number)
        
        # If LLM suggests a different page, check that page too
        if not llm_result.get('verified', False) and llm_result.get('page_number') and llm_result.get('page_number') != expected_page:
            suggested_page = llm_result.get('page_number')
            suggested_page_text = self.pdf_processor.get_page_text(suggested_page) or ""
            
            suggested_result = self.llm_verifier.verify_header(header_text, suggested_page_text, suggested_page, item_number)
            
            # Use the better result
            if suggested_result.get('confidence', 0) > llm_result.get('confidence', 0):
                llm_result = suggested_result
        
        # Format the result to match our standard format
        found_pages = {}
        if llm_result.get('verified', False) and llm_result.get('page_number'):
            page_num = llm_result.get('page_number')
            found_pages[page_num] = {
                'confidence': llm_result.get('confidence', 0),
                'distance_from_expected': abs(page_num - expected_page),
                'explanation': llm_result.get('explanation', '')
            }
        
        # Determine status
        confidence = llm_result.get('confidence', 0)
        if llm_result.get('verified', False) and confidence > 0.9:
            status = "verified"
        elif confidence > 0.8:
            status = "likely_correct"
        elif confidence > 0.6:
            status = "needs_review"
        else:
            status = "likely_incorrect"
        
        return {
            'item_number': item_number,
            'header_text': header_text,
            'expected_page': expected_page,
            'found_pages': found_pages,
            'best_match_page': llm_result.get('page_number'),
            'confidence': confidence,
            'status': status,
            'method': 'llm',
            'explanation': llm_result.get('explanation', '')
        }
    
    def get_verification_summary(self):
        """
        Get a summary of the verification results
        
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
            'not_found': 0,
            'by_method': {
                'pattern_matching': 0,
                'transformer': 0,
                'llm': 0
            }
        }
        
        for result in self.verification_results.values():
            status = result.get('status')
            method = result.get('method')
            
            if status in summary:
                summary[status] += 1
            
            if method in summary['by_method']:
                summary['by_method'][method] += 1
        
        return summary
    
    def get_headers_by_status(self, status):
        """
        Get headers with a specific verification status
        
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
        Get all verification results
        
        Returns:
            dict: All verification results
        """
        if not self.verification_results:
            self.verify_all_headers()
        
        return self.verification_results
    
    def update_header_verification(self, item_number, new_page_number, approved=True):
        """
        Update the verification result for a header and store the correction
        
        Args:
            item_number (int): Item number
            new_page_number (int): New page number
            approved (bool): Whether the verification is approved
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if item_number not in self.verification_results:
            return False
        
        result = self.verification_results[item_number]
        original_page = result.get('expected_page')
        header_text = result.get('header_text')
        
        # Update the result
        result['expected_page'] = new_page_number
        
        if approved:
            result['status'] = "verified"
            result['confidence'] = 1.0
            result['best_match_page'] = new_page_number
        
        # Store the correction in the database
        if original_page != new_page_number:
            pdf_name = os.path.basename(self.pdf_processor.pdf_path) if hasattr(self.pdf_processor, 'pdf_path') else ""
            self.header_db.add_correction(item_number, header_text, original_page, new_page_number, pdf_name, 1.0 if approved else 0.5)
        
        return True


# Example usage
if __name__ == "__main__":
    from pdf_processor import PDFProcessor, JSONProcessor
    
    # This is just for testing the module directly
    pdf_path = "/path/to/pdf"
    json_path = "/path/to/json"
    
    if os.path.exists(pdf_path) and os.path.exists(json_path):
        pdf_processor = PDFProcessor(pdf_path)
        json_processor = JSONProcessor(json_path)
        
        engine = EnhancedVerificationEngine(pdf_processor, json_processor)
        results = engine.verify_all_headers()
        
        print("Verification Summary:")
        print(engine.get_verification_summary())
        
        print("\nHeaders that need review:")
        for header in engine.get_headers_by_status("needs_review"):
            print(f"Item {header['item_number']}: {header['header_text']}")
            print(f"  Expected page: {header['expected_page']}")
            print(f"  Best match page: {header['best_match_page']}")
            print(f"  Confidence: {header['confidence']:.2f}")
            print(f"  Method: {header['method']}")
            print()
