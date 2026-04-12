import re
import string
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

class Normalizer:
    """
    Handles text cleaning and tokenization for the Ngram project.
    
    This class ensures consistent text processing for training data, 
    user input during inference, and evaluation sets.
    """

    def __init__(self):
        """
        Initializes the Normalizer. 
        Downloads required NLTK resources if not already present.
        """
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def normalize(self, text: str) -> list[str]:
        """
        The single source of truth for text cleaning.
        
        This method applies:
        1. Lowercasing.
        2. Hyphen separation.
        3. Digit removal.
        4. Apostrophe/contraction merging.
        5. Punctuation removal (alpha characters only).

        Args:
            text (str): The raw text to clean.

        Returns:
            list[str]: A list of cleaned word tokens.
        """
        if not text:
            return []

        # 1. Lowercase
        text = text.lower()

        # 2. Handle Hyphenated Words: Separate them
        text = text.replace('-', ' ')

        # 3. Remove Numbers
        text = re.sub(r'\d+', ' ', text)

        # 4. Remove apostrophes to merge contractions (don't -> dont)
        text = text.replace("'", "").replace("’", "")

        # 5. Tokenize into Sentences then Words
        # This structure helps word_tokenize perform better
        sentences = sent_tokenize(text)
        flat_list = []

        for sent in sentences:
            words = word_tokenize(sent)
            for word in words:
                # Keep ONLY letters (a-z)
                clean_word = "".join(char for char in word if char.isalpha())
                
                if clean_word.strip():
                    flat_list.append(clean_word)

        return flat_list