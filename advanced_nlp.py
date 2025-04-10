import os
import re
import nltk
import spacy
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class AdvancedNLPProcessor:
    """
    Class for implementing advanced NLP techniques for header verification
    """
    
    def __init__(self):
        """Initialize the advanced NLP processor"""
        self.nlp = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.load_spacy_model()
        
    def load_spacy_model(self):
        """Load the spaCy model for NLP processing"""
        try:
            # Try to load a larger model first
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            try:
                # Fall back to the small model if the medium one isn't available
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # If no model is installed, download the small one
                os.system("python -m spacy download en_core_web_sm")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception as e:
                    print(f"Error loading spaCy model: {str(e)}")
                    self.nlp = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for NLP analysis
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join back into text
        return ' '.join(tokens)
    
    def extract_header_candidates(self, page_text: str) -> List[str]:
        """
        Extract potential header candidates from page text
        
        Args:
            page_text: Text content of the page
            
        Returns:
            List of potential header candidates
        """
        candidates = []
        
        # Look for lines starting with "ITEM"
        item_pattern = re.compile(r'^(ITEM\s+\d+\.?.*?)$', re.MULTILINE | re.IGNORECASE)
        item_matches = item_pattern.findall(page_text)
        candidates.extend(item_matches)
        
        # Look for lines with all caps (potential headers)
        lines = page_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and line.isupper() and len(line) > 10:
                candidates.append(line)
        
        # Use spaCy for more sophisticated extraction if available
        if self.nlp:
            doc = self.nlp(page_text)
            
            # Extract sentences that might be headers based on their structure
            for sent in doc.sents:
                sent_text = sent.text.strip()
                # Check if sentence is short and starts with a number or "ITEM"
                if (len(sent_text) < 100 and 
                    (sent_text.startswith("ITEM") or 
                     any(token.is_digit for token in sent[:2]))):
                    candidates.append(sent_text)
        
        # Remove duplicates while preserving order
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def compute_text_similarity(self, text1: str, text2: str, method: str = 'tfidf') -> float:
        """
        Compute similarity between two texts using various methods
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ('tfidf', 'spacy', 'levenshtein', 'ensemble')
            
        Returns:
            Similarity score (0-1)
        """
        if method == 'tfidf':
            return self._tfidf_similarity(text1, text2)
        elif method == 'spacy':
            return self._spacy_similarity(text1, text2)
        elif method == 'levenshtein':
            return self._levenshtein_similarity(text1, text2)
        elif method == 'ensemble':
            # Use an ensemble of methods for more robust similarity
            tfidf_sim = self._tfidf_similarity(text1, text2)
            spacy_sim = self._spacy_similarity(text1, text2)
            levenshtein_sim = self._levenshtein_similarity(text1, text2)
            
            # Weight the methods (can be adjusted based on performance)
            weights = [0.4, 0.4, 0.2]  # tfidf, spacy, levenshtein
            return (tfidf_sim * weights[0] + 
                    spacy_sim * weights[1] + 
                    levenshtein_sim * weights[2])
        else:
            # Default to TF-IDF
            return self._tfidf_similarity(text1, text2)
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Preprocess texts
        text1_processed = self.preprocess_text(text1)
        text2_processed = self.preprocess_text(text2)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        
        try:
            # Transform texts to TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform([text1_processed, text2_processed])
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error computing TF-IDF similarity: {str(e)}")
            # Fall back to Levenshtein similarity
            return self._levenshtein_similarity(text1, text2)
    
    def _spacy_similarity(self, text1: str, text2: str) -> float:
        """
        Compute spaCy similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not self.nlp:
            # Fall back to Levenshtein if spaCy is not available
            return self._levenshtein_similarity(text1, text2)
        
        try:
            # Process texts with spaCy
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # Compute similarity
            similarity = doc1.similarity(doc2)
            
            return float(similarity)
        except Exception as e:
            print(f"Error computing spaCy similarity: {str(e)}")
            # Fall back to Levenshtein similarity
            return self._levenshtein_similarity(text1, text2)
    
    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Levenshtein (sequence) similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def extract_structured_headers(self, page_text: str) -> List[Dict[str, Any]]:
        """
        Extract structured header information from page text
        
        Args:
            page_text: Text content of the page
            
        Returns:
            List of structured header dictionaries
        """
        structured_headers = []
        
        # Regular expression to match FDD headers
        header_pattern = re.compile(
            r'(ITEM\s+(\d+)\.?\s+([A-Z][A-Z\s\.,\-&\']+))',
            re.IGNORECASE
        )
        
        matches = header_pattern.finditer(page_text)
        
        for match in matches:
            full_match = match.group(1)
            item_number = int(match.group(2))
            header_text = match.group(3).strip()
            
            structured_headers.append({
                'full_text': full_match,
                'item_number': item_number,
                'header_text': header_text,
                'position': match.start()
            })
        
        # Use spaCy for more sophisticated extraction if available
        if self.nlp and not structured_headers:
            doc = self.nlp(page_text)
            
            # Look for patterns like "ITEM X" followed by uppercase text
            for i, token in enumerate(doc):
                if token.text.lower() == "item" and i + 1 < len(doc) and doc[i+1].is_digit:
                    item_number = int(doc[i+1].text)
                    
                    # Extract the header text (next few tokens that are uppercase)
                    header_text = ""
                    j = i + 2
                    while j < len(doc) and (doc[j].is_upper or doc[j].is_punct or doc[j].is_space):
                        header_text += doc[j].text_with_ws
                        j += 1
                    
                    if header_text:
                        structured_headers.append({
                            'full_text': f"ITEM {item_number} {header_text}".strip(),
                            'item_number': item_number,
                            'header_text': header_text.strip(),
                            'position': token.idx
                        })
        
        return structured_headers
    
    def analyze_document_structure(self, pdf_text_by_page: Dict[int, str]) -> Dict[str, Any]:
        """
        Analyze the structure of the document to identify patterns in header placement
        
        Args:
            pdf_text_by_page: Dictionary mapping page numbers to page text
            
        Returns:
            Dictionary with document structure analysis
        """
        structure = {
            'header_pages': {},
            'avg_pages_between_headers': 0,
            'header_positions': {},
            'toc_page': None
        }
        
        all_headers = []
        
        # Extract headers from each page
        for page_num, page_text in pdf_text_by_page.items():
            headers = self.extract_structured_headers(page_text)
            
            for header in headers:
                header['page'] = page_num
                all_headers.append(header)
                
                item_num = header['item_number']
                structure['header_pages'][item_num] = page_num
        
        # Sort headers by item number
        all_headers.sort(key=lambda x: x['item_number'])
        
        # Calculate average pages between headers
        if len(all_headers) > 1:
            page_diffs = []
            for i in range(1, len(all_headers)):
                prev_page = all_headers[i-1]['page']
                curr_page = all_headers[i]['page']
                page_diffs.append(curr_page - prev_page)
            
            if page_diffs:
                structure['avg_pages_between_headers'] = sum(page_diffs) / len(page_diffs)
        
        # Analyze header positions on page
        for header in all_headers:
            item_num = header['item_number']
            position = header['position']
            page_text = pdf_text_by_page[header['page']]
            
            # Calculate relative position in the page (0-1)
            relative_position = position / len(page_text) if page_text else 0
            structure['header_positions'][item_num] = relative_position
        
        # Try to identify table of contents page
        for page_num, page_text in pdf_text_by_page.items():
            if re.search(r'(TABLE\s+OF\s+CONTENTS|CONTENTS)', page_text, re.IGNORECASE):
                structure['toc_page'] = page_num
                break
        
        return structure
    
    def predict_header_page(self, item_number: int, document_structure: Dict[str, Any]) -> Optional[int]:
        """
        Predict the page number for a header based on document structure analysis
        
        Args:
            item_number: Item number to predict
            document_structure: Document structure analysis
            
        Returns:
            Predicted page number or None
        """
        header_pages = document_structure.get('header_pages', {})
        
        # If we already have this header, return its page
        if item_number in header_pages:
            return header_pages[item_number]
        
        # If we have headers before and after, interpolate
        prev_item = None
        next_item = None
        
        for i in sorted(header_pages.keys()):
            if i < item_number:
                prev_item = i
            elif i > item_number:
                next_item = i
                break
        
        if prev_item and next_item:
            prev_page = header_pages[prev_item]
            next_page = header_pages[next_item]
            
            # Simple linear interpolation
            items_between = next_item - prev_item
            pages_between = next_page - prev_page
            
            if items_between > 0:
                relative_position = (item_number - prev_item) / items_between
                predicted_page = prev_page + int(relative_position * pages_between)
                return predicted_page
        
        # If we only have headers before or after, use average pages between headers
        avg_pages = document_structure.get('avg_pages_between_headers', 0)
        
        if prev_item:
            return header_pages[prev_item] + int((item_number - prev_item) * avg_pages)
        elif next_item:
            return header_pages[next_item] - int((next_item - item_number) * avg_pages)
        
        return None
    
    def extract_keywords_from_header(self, header_text: str) -> List[str]:
        """
        Extract important keywords from a header text
        
        Args:
            header_text: Header text
            
        Returns:
            List of keywords
        """
        # Remove "ITEM X" prefix if present
        header_text = re.sub(r'^ITEM\s+\d+\.?\s+', '', header_text, flags=re.IGNORECASE)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(header_text.lower())
        keywords = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        return keywords
    
    def find_header_by_keywords(self, keywords: List[str], page_text: str) -> Dict[str, Any]:
        """
        Find a header in page text based on keywords
        
        Args:
            keywords: List of keywords to search for
            page_text: Text content of the page
            
        Returns:
            Dictionary with match information
        """
        # Preprocess page text
        page_text_lower = page_text.lower()
        page_tokens = word_tokenize(page_text_lower)
        page_tokens = [token for token in page_tokens if token.isalnum()]
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in page_tokens)
        match_ratio = matches / len(keywords) if keywords else 0
        
        # Find the best matching sentence
        best_sentence = None
        best_score = 0
        
        sentences = sent_tokenize(page_text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in keywords if keyword in sentence_lower)
            score = keyword_count / len(keywords) if keywords else 0
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return {
            'match_ratio': match_ratio,
            'best_sentence': best_sentence,
            'best_score': best_score
        }
    
    def verify_header_with_nlp(self, header_text: str, expected_page: int, pdf_text_by_page: Dict[int, str]) -> Dict[str, Any]:
        """
        Verify a header using advanced NLP techniques
        
        Args:
            header_text: Header text to verify
            expected_page: Expected page number
            pdf_text_by_page: Dictionary mapping page numbers to page text
            
        Returns:
            Dictionary with verification results
        """
        # Extract item number from header text
        item_match = re.search(r'ITEM\s+(\d+)', header_text, re.IGNORECASE)
        item_number = int(item_match.group(1)) if item_match else None
        
        # Extract keywords from header
        keywords = self.extract_keywords_from_header(header_text)
        
        # Check expected page first
        expected_page_text = pdf_text_by_page.get(expected_page, "")
        expected_page_result = self.find_header_by_keywords(keywords, expected_page_text)
        
        # If good match on expected page, return it
        if expected_page_result['match_ratio'] > 0.8:
            return {
                'verified': True,
                'confidence': expected_page_result['match_ratio'],
                'page_number': expected_page,
                'best_match': expected_page_result['best_sentence'],
                'method': 'keyword_matching'
            }
        
        # Check nearby pages
        window_size = 5
        start = max(1, expected_page - window_size)
        end = expected_page + window_size
        
        best_page = None
        best_result = None
        best_score = 0
        
        for page_num in range(start, end + 1):
            if page_num == expected_page or page_num not in pdf_text_by_page:
                continue
                
            page_text = pdf_text_by_page[page_num]
            result = self.find_header_by_keywords(keywords, page_text)
            
            if result['match_ratio'] > best_score:
                best_score = result['match_ratio']
                best_result = result
                best_page = page_num
        
        # If we found a good match on another page
        if best_score > 0.7:
            return {
                'verified': True,
                'confidence': best_score,
                'page_number': best_page,
                'best_match': best_result['best_sentence'],
                'method': 'keyword_matching'
            }
        
        # If no good matches found, try document structure analysis
        document_structure = self.analyze_document_structure(pdf_text_by_page)
        predicted_page = self.predict_header_page(item_number, document_structure) if item_number else None
        
        if predicted_page:
            predicted_page_text = pdf_text_by_page.get(predicted_page, "")
            predicted_result = self.find_header_by_keywords(keywords, predicted_page_text)
            
            if predicted_result['match_ratio'] > 0.6:
                return {
                    'verified': True,
                    'confidence': predicted_result['match_ratio'] * 0.9,  # Slightly lower confidence for structure-based prediction
                    'page_number': predicted_page,
                    'best_match': predicted_result['best_sentence'],
                    'method': 'document_structure'
                }
        
        # If still no good matches, return the best we found
        if best_page:
            return {
                'verified': False,
                'confidence': best_score,
                'page_number': best_page,
                'best_match': best_result['best_sentence'],
                'method': 'keyword_matching'
            }
        
        # No matches found
        return {
            'verified': False,
            'confidence': 0,
            'page_number': None,
            'best_match': None,
            'method': 'nlp'
        }


# Example usage
if __name__ == "__main__":
    # This is just for testing the module directly
    nlp_processor = AdvancedNLPProcessor()
    
    # Test text similarity
    text1 = "ITEM 1. THE FRANCHISOR, AND ANY PARENTS, PREDECESSORS, AND AFFILIATES"
    text2 = "ITEM 1 THE FRANCHISOR AND ANY PARENTS PREDECESSORS AND AFFILIATES"
    
    similarity = nlp_processor.compute_text_similarity(text1, text2, method='ensemble')
    print(f"Similarity: {similarity:.4f}")
    
    # Test header extraction
    sample_text = """
    ITEM 1. THE FRANCHISOR, AND ANY PARENTS, PREDECESSORS, AND AFFILIATES
    
    A. The Franchisor
    
    We are a Delaware limited liability company formed on January 5, 2015. Our principal business address is 
    500 East Broward Boulevard, Suite 1710, Fort Lauderdale, Florida 33394. We do business under our 
    corporate name, Ruth's Chris Steak House Franchise, LLC, and under the name "Ruth's Chris Steak 
    House." We have offered franchises for Ruth's Chris Steak House restaurants since January 2015.
    """
    
    headers = nlp_processor.extract_structured_headers(sample_text)
    print("\nExtracted Headers:")
    for header in headers:
        print(f"Item {header['item_number']}: {header['header_text']}")
