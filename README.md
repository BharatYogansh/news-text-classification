# News Topic Classification & Analytics Dashboard

Real text classification on the **20 Newsgroups** dataset (scikit-learn's built-in fetch,
genuinely real Usenet posts, classic NLP benchmark). Compares three classifiers on the same
TF-IDF features, evaluates each with accuracy/F1/confusion matrix, digs into what each model
gets wrong, and puts all of it in a live Streamlit dashboard with an interactive predictor.

Replaces four vague "I did some ML/NLP/dashboards" claims with one real, runnable, verifiable
project.

## Categories

5 of the dataset's 20 categories, picked for a clean multi-class problem with minimal topic
overlap:

`comp.graphics` &middot; `sci.space` &middot; `rec.sport.baseball` &middot; `talk.politics.misc` &middot; `sci.med`

Change the list in `config.py` any time - `CATEGORIES`.

## How it works

1. **`train.py`** fetches the real dataset (needs internet the first time - scikit-learn
   downloads and caches it locally after that), strips headers/footers/quoted text so the
   models learn from post content rather than metadata, TF-IDF vectorizes it (max 10,000
   features, English stop words removed), and trains three classifiers on identical features:
   Logistic Regression, Multinomial Naive Bayes, Linear SVM.
2. For each model it computes accuracy, macro/weighted F1, per-class precision/recall/F1, and
   a full confusion matrix - saved to `results.json`.
3. Every test post's true label and each model's prediction gets saved to
   `test_predictions.csv`, so the dashboard can show misclassified examples without re-running
   inference.
4. **`app.py`** (Streamlit) lets you switch between the three models and see: accuracy/F1,
   confusion matrix heatmap, per-class F1 bar chart, real misclassified examples, and a
   textbox where you type any text and get a live prediction from the selected model.

## Run it

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Fetch data + train all 3 models (needs internet - first run downloads ~14MB)
python train.py

# 2. Launch the dashboard
streamlit run app.py
```

Open `http://localhost:8501`.

`train.py` prints accuracy/F1 per model as it goes, plus which model won on macro F1 - copy
those real numbers into the README and resume once you've run it.

## Project structure

```
news-text-classification/
├── config.py       # categories, model names, artifact paths - single source of truth
├── train.py         # fetch + TF-IDF + train 3 models + evaluate + save artifacts
├── app.py           # Streamlit dashboard: compare models, confusion matrix, error
│                     # analysis, live predictor
├── requirements.txt
└── .gitignore
```

Artifacts `train.py` produces locally (gitignored, regenerate anytime by re-running it):
`vectorizer.pkl`, `model_logistic_regression.pkl`, `model_naive_bayes.pkl`,
`model_linear_svm.pkl`, `results.json`, `test_predictions.csv`.

## What this demonstrates

- Real NLP feature engineering (TF-IDF) on real, unstructured text - not a toy CSV.
- Comparing multiple algorithms on identical features instead of picking one and hoping.
- Evaluating with the right metrics for multi-class text classification (macro F1, not just
  accuracy) plus a confusion matrix instead of a single number.
- Error analysis: looking at what the model actually gets wrong and why, not just reporting a
  score.
- An interactive dashboard, not just a static notebook.

## Resume bullet (fill in real numbers after running `train.py`)

**News Topic Classification & Analytics Dashboard** &mdash; Python, Scikit-learn, TF-IDF, Streamlit

Built an NLP text-classification pipeline on the 20 Newsgroups dataset using TF-IDF features;
trained and compared Logistic Regression, Multinomial Naive Bayes, and Linear SVM (best model:
______, accuracy ____%, macro F1 ____); evaluated with confusion matrices and per-class F1, and
built an interactive Streamlit dashboard for model comparison, error analysis, and live
predictions.

## Suggested next steps

- Try bigrams (`ngram_range=(1,2)`) in the TF-IDF vectorizer and see if it moves F1.
- Swap TF-IDF for embeddings (e.g. sentence-transformers) and compare against the classical
  baselines - honest ablation, not just a bigger model for its own sake.
- Add a `/predict` FastAPI endpoint alongside the dashboard if you want a second serving
  surface (optional - the dashboard alone already demonstrates the full pipeline).
