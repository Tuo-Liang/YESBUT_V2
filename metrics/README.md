# Evaluation Scripts

This folder contains two Python scripts for evaluating text-generation tasks:

1. **`evaluation.py`**  
   - Computes automated metrics (BLEU, ROUGE, METEOR, BERTScore, BLEURT, CIDEr, etc.) to compare generated text with references.  
   - Expects JSON input with fields like `output` (generated text) and a reference field (`caption` or `contradiction`).

2. **`G_eval.py`**  
   - Uses OpenAI’s GPT models to provide subjective ratings (1–5) for generated text.  
   - Supports caption evaluation (`evaluate_caption`) and contradiction evaluation (`evaluate_contradiction`).

## Dependencies

Install the following (Python 3.7+ recommended):

pip install pandas numpy scikit-learn nltk rouge-score bert-score evaluate pycocoevalcap openai

Then, within a Python shell:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')


In G_eval.py, set API_KEY = '<your_openai_api_key>' before use.

## Usage

Run:
python evaluation.py
Metrics (BLEU, ROUGE, etc.) will be printed or saved as JSON.

Run:
python G_eval.py
Ratings (1–5) are output to the g_eval folder.
