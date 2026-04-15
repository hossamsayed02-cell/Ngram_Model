import json
from pathlib import Path
from collections import Counter, defaultdict

class NGramModel:
    """
    Handles the creation, storage, and retrieval of N-gram probabilities.
    Includes backoff logic for predicting the next word.
    """

    def __init__(self, n=4, unk_threshold=3):
        """
        Initializes the model structure.
        
        Args:
            n (int): Maximum N-gram order.
            unk_threshold (int): Minimum frequency for a word to stay in vocab.
        """
        self.n = n
        self.unk_threshold = unk_threshold
        self.vocab = set()
        # Structure: {order: {context_string: {next_word: probability}}}
        self.model_data = {str(i): defaultdict(dict) for i in range(1, n + 1)}

    def build_vocab(self, all_tokens: list[str]):
        """
        Identifies unique words and replaces rare words with <UNK>.
        
        Args:
            all_tokens (list[str]): List of normalized tokens from training data.
        """
        word_counts = Counter(all_tokens)
        self.vocab = {
            word for word, count in word_counts.items() 
            if count >= self.unk_threshold
        }
        self.vocab.add("<UNK>")

    def build_model(self, all_tokens: list[str]):
        """
        Calculates probabilities for all N-gram orders up to n.
        
        Args:
            all_tokens (list[str]): List of normalized tokens.
        """
        # 1. Process tokens to handle <UNK>
        processed_tokens = [
            w if w in self.vocab else "<UNK>" for w in all_tokens
        ]

        # 2. Build Unigrams (Order 1)
        total_count = len(processed_tokens)
        unigram_counts = Counter(processed_tokens)
        for word, count in unigram_counts.items():
            self.model_data["1"][word] = count / total_count

        # 3. Build Higher Order N-grams
        for order in range(2, self.n + 1):
            ngram_counts = defaultdict(Counter)
            
            for i in range(len(processed_tokens) - order + 1):
                context = " ".join(processed_tokens[i : i + order - 1])
                next_word = processed_tokens[i + order - 1]
                ngram_counts[context][next_word] += 1
            
            # Convert counts to probabilities
            for context, counts in ngram_counts.items():
                context_total = sum(counts.values())
                for next_word, count in counts.items():
                    self.model_data[str(order)][context][next_word] = count / context_total

    def save_model(self, model_path: str, vocab_path: str):
        """Saves vocab and model probabilities to JSON files."""
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(list(self.vocab), f, indent=4)
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(self.model_data, f, indent=4)

    def load_model(self, model_path: str, vocab_path: str):
        """Loads vocab and model probabilities from JSON files."""
        # This gets the directory part of your path and creates it if it's missing
        Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = set(json.load(f))
            
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)    
        with open(model_path, "r", encoding="utf-8") as f:
            self.model_data = json.load(f)

    def lookup(self, context_tokens: list[str], k: int = 5) -> list[tuple[str, float]]:
        """
        The single source of backoff logic for the predictor.
        
        Args:
            context_tokens (list[str]): List of previous words.
            k (int): Number of predictions to return.
            
        Returns:
            list[tuple]: List of (word, probability) pairs.
        """
        # Ensure context words are in vocab or converted to <UNK>
        clean_context = [
            w if w in self.vocab else "<UNK>" for w in context_tokens
        ]

        # Backoff logic: Start from highest order (n) down to unigram (1)
        for order in range(self.n, 0, -1):
            order_str = str(order)
            
            if order == 1:
                # Fallback to unigrams
                preds = self.model_data["1"]
                return sorted(preds.items(), key=lambda x: x[1], reverse=True)[:k]
            
            # Get the appropriate context slice for this order
            context_size = order - 1
            current_context = " ".join(clean_context[-context_size:])
            
            if current_context in self.model_data[order_str]:
                preds = self.model_data[order_str][current_context]
                return sorted(preds.items(), key=lambda x: x[1], reverse=True)[:k]

        return [("<UNK>", 0.0)]
