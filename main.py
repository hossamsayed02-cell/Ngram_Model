import os
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from gutenberg_cleaner import simple_cleaner

# Internal imports
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def ensure_paths(paths: list):
    """Checks if directories exist; if not, creates them."""
    for path in paths:
        path_obj = Path(path)
        # If it's a file path, get the parent directory
        directory = path_obj.parent if path_obj.suffix else path_obj
        
        if not directory.exists():
            print(f"Directory not found. Creating: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

def main():
    """
    Main orchestrator for the N-Gram Predictor.
    """
    load_dotenv("config/.env")
    
    # 1. Environment Variables
    RAW_TRAIN_DIR = os.getenv("RAW_TRAIN_DIR")
    PROCESSED_TRAIN_FILE = os.getenv("PROCESSED_TRAIN_FILE")
    MODEL_PATH = os.getenv("MODEL_PATH")
    VOCAB_PATH = os.getenv("VOCAB_PATH")
    NGRAM_ORDER = int(os.getenv("NGRAM_ORDER", 4))
    UNK_THRESHOLD = int(os.getenv("UNK_THRESHOLD", 3))
    TOP_K = int(os.getenv("TOP_K", 3))

    # --- FOLDER CHECK ---
    # Ensure raw data exists before starting, otherwise we can't train
    if not Path(RAW_TRAIN_DIR).exists():
        print(f"CRITICAL ERROR: Raw data directory '{RAW_TRAIN_DIR}' does not exist.")
        print("Please add your .txt files to that folder and try again.")
        sys.exit(1)

    # Ensure output directories exist for processed data and models
    ensure_paths([PROCESSED_TRAIN_FILE, MODEL_PATH])

    # 2. Dependency Injection
    normalizer = Normalizer()
    ngram_model = NGramModel(n=NGRAM_ORDER, unk_threshold=UNK_THRESHOLD)

    # 3. Argument Parsing
    parser = argparse.ArgumentParser(description="N-Gram Predictor CLI")
    parser.add_argument("--step", type=str, required=True, 
                        choices=["prep", "train", "predict", "all"],
                        help="Execution step: prep, train, predict, or all")
    args = parser.parse_args()

    # --- STEP: PREP ---
    if args.step in ["prep", "all"]:
        print("--- Step 1: Normalizing Raw Data ---")
        all_tokens = []
        raw_path = Path(RAW_TRAIN_DIR)
        
        files = list(raw_path.glob("*.txt"))
        if not files:
            print(f"Warning: No .txt files found in {RAW_TRAIN_DIR}")
        
        for file in files:
            print(f"Processing: {file.name}")
            with open(file, "r", encoding="utf-8") as f:
                clean_text = simple_cleaner(f.read())
                tokens = normalizer.normalize(clean_text)
                all_tokens.extend(tokens)
        
        with open(PROCESSED_TRAIN_FILE, "w", encoding="utf-8") as f:
            f.write(" ".join(all_tokens))
        print(f"Total tokens saved to {PROCESSED_TRAIN_FILE}")

    # --- STEP: TRAIN ---
    if args.step in ["train", "all"]:
        print("--- Step 2: Building N-Gram Model ---")
        if not Path(PROCESSED_TRAIN_FILE).exists():
            print("Error: Processed tokens not found. Run --step prep first.")
            return

        with open(PROCESSED_TRAIN_FILE, "r", encoding="utf-8") as f:
            tokens = f.read().split()
        
        ngram_model.build_vocab(tokens)
        ngram_model.build_model(tokens)
        ngram_model.save_model(MODEL_PATH, VOCAB_PATH)
        print(f"Model saved to {MODEL_PATH}")

    # --- STEP: PREDICT ---
    if args.step in ["predict", "all"]:
        print("--- Step 3: Interactive Prediction ---")
        if not Path(MODEL_PATH).exists():
            print("Error: Model file not found. Run --step train first.")
            return

        ngram_model.load_model(MODEL_PATH, VOCAB_PATH)
        predictor = Predictor(model=ngram_model, normalizer=normalizer)
        
        print(f"(N-Gram Order: {NGRAM_ORDER}, Top K: {TOP_K})")
        while True:
            text = input("\nEnter context (or 'q' to quit): ")
            if text.lower() == 'q':
                break
            
            predictions = predictor.predict_next(text, k=TOP_K)
            
            print(f"Top {TOP_K} Predictions:")
            for word, prob in predictions:
                print(f"  -> {word:15} | Prob: {prob:.4f}")

if __name__ == "__main__":
    main()