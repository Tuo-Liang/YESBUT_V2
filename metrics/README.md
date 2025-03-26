# Evaluation Scripts

This folder contains two Python scripts for evaluating text-generation tasks:

1. **`evaluation.py`**  
   - Computes automated metrics (BLEU, ROUGE, METEOR, BERTScore, BLEURT, CIDEr, etc.) to compare generated text with references.  
   - Expects JSON input with fields like `output` (generated text) and a reference field (e.g., `caption` or `contradiction`).

2. **`G_eval.py`**  
   - Uses OpenAI’s GPT models to provide subjective ratings (1–5) for generated text.  
   - Supports caption evaluation (`evaluate_caption`) and contradiction evaluation (`evaluate_contradiction`).

## DEPENDENCIES

Install the following (Python 3.7+ recommended):
```bash
pip install pandas numpy scikit-learn nltk rouge-score bert-score evaluate pycocoevalcap openai
```

Then, within a Python shell:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

In `G_eval.py`, set:
```python
API_KEY = '<your_openai_api_key>'
```
before running the script.

## USAGE

1. **Run Automated Metrics** (`evaluation.py`):  
   ```bash
   python evaluation.py
   ```
   - This calculates metrics like BLEU, ROUGE, METEOR, etc. and prints or saves them as JSON.

2. **Run GPT-Based Scores** (`G_eval.py`):  
   ```bash
   python G_eval.py
   ```
   - This will produce 1–5 ratings and save them in the `g_eval` folder.
