# N-Gram Next Word Predictor

An end-to-end N-Gram language model designed to process raw text data, build a probabilistic model, and provide real-time next-word predictions. This project demonstrates modular software design, dependency injection, and robust text normalization.

## 📋 Features
- **Flexible N-Gram Support:** Configurable $N$ (e.g., Unigrams through 4-grams).
- **Backoff Logic:** Automatically retreats to lower-order N-Grams if a specific context is unseen.
- **UNK Handling:** Robustly manages Out-of-Vocabulary (OOV) words using frequency thresholds.
- **Environment-Driven:** All file paths and hyperparameters are controlled via configuration files.

---

## 🛠 Tooling & Environment

### Prerequisites
- **VS Code** (Primary IDE)
- **Anaconda / Miniconda** (Environment Management)

### Environment Setup
1. **Create the Anaconda environment:**
   ```bash
   conda create -n ngram-predictor python=3.9 -y

2. **Activate the environment:**
   ```bash
   conda activate ngram-predictor

### Install dependencies:
```bash
pip install -r requirements.txt

### Usage
The project is executed through a single entry point, `main.py`, using the `--step` flag.

| Command | Description |
| :--- | :--- |
| `python main.py --step prep` | Cleans raw books and generates tokenized text files. |
| `python main.py --step train` | Processes tokens to build vocabulary and probability maps. |
| `python main.py --step predict` | Launches the interactive CLI for word prediction. |
| `python main.py --step all` | Runs the entire pipeline from raw data to trained model. |


## 📂 Project Structure

```text
ngram-predictor/
├── config/
│   └── .env                # Path and hyperparameter configurations
├── data/
│   ├── raw/                # Training and evaluation text files
│   ├── processed/          # Cleaned token files
│   └── model/              # Generated JSON model and vocabulary
├── src/
│   ├── data_prep/          # Normalizer class (Text cleaning)
│   ├── model/              # NGramModel class (Logic and Backoff)
│   └── inference/          # Predictor class (Inference logic)
├── main.py                 # Application entry point and dependency wiring
├── .gitignore              # Excludes data, JSON, and cache files
├── requirements.txt        # Pinned library dependencies
└── README.md               # Project documentation