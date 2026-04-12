class Predictor:
    """
    Predicts the next word in a sequence using a trained NGramModel.
    
    This class acts as a bridge between the raw user input and the 
    mathematical model, ensuring input is normalized correctly before lookup.
    """

    def __init__(self, model, normalizer):
        """
        Initializes the Predictor with injected dependencies.

        Args:
            model (NGramModel): An instance of the NGramModel class (already loaded).
            normalizer (Normalizer): An instance of the Normalizer class.
        """
        self.model = model
        self.normalizer = normalizer

    def predict_next(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Processes raw text input and returns the top k word predictions.

        Args:
            text (str): The raw string provided by the user.
            k (int): The number of top predictions to return.

        Returns:
            list[tuple[str, float]]: A list of (word, probability) tuples.
        """
        # 1. Clean the input using the ALREADY existing normalizer logic
        # Requirement: Normalizer.normalize() reused - not re-implemented
        tokens = self.normalizer.normalize(text)

        # 2. Call the model's lookup function
        # Requirement: NGramModel.lookup() is the SINGLE source of logic
        if not tokens:
            # If no valid words were entered, fall back to unigram top-k via lookup
            return self.model.lookup([], k=k)

        # We pass the cleaned tokens directly to the model
        return self.model.lookup(tokens, k=k)