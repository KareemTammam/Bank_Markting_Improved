# 🏦 Bank Marketing Prediction

A modern, production-style Streamlit web application that predicts whether a bank
customer will subscribe to a term deposit, built on top of a full ML benchmarking
pipeline (8 models, tuned via cross-validated grid search).

**Live demo pages:**
- **Overview** — project summary, key stats, quick-start CTAs
- **Dataset** — EDA: target distribution, correlations, feature reference
- **Model Dashboard** — KPI cards, model comparison chart, confusion matrix, feature importance
- **Make a Prediction** — interactive form → instant prediction with confidence score
- **About the Project** — problem, dataset, preprocessing, models, evaluation, best model

---

## 🗂️ Project Structure

```
project/
│
├── app.py                  # Streamlit entry point (page routing)
├── pages_content.py         # All page render functions
├── style.py                 # Design system (colors, CSS)
├── utils.py                  # Cached data/model loaders
├── train.py                  # Reproducible training script
│
├── model/
│   ├── trained_model.pkl         # Best model (XGBoost)
│   ├── preprocessor.pkl          # Fitted ColumnTransformer
│   ├── feature_names.json
│   ├── meta.json
│   ├── model_comparison.json
│   ├── feature_importance.json
│   ├── confusion_matrices.json
│   ├── target_correlation.json
│   ├── raw_options.json
│   └── sample_data.csv
│
├── data/
│   └── bank.csv               # Source dataset (11,162 rows)
│
├── .streamlit/
│   └── config.toml             # Theme + server config
│
├── requirements.txt
└── README.md
```

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the **pre-trained** model and preprocessor from `model/` — it does
**not** retrain on startup, so it launches instantly.

## 🔁 Retraining

If you want to regenerate every artifact under `model/` from `data/bank.csv`:

```bash
python train.py
```

This re-fits the preprocessing pipeline and all 8 models (using the best
hyperparameters already found via grid search in the original notebook) and
overwrites the `model/*.json` and `model/*.pkl` files.

## ☁️ Deployment

This project is ready for **Streamlit Community Cloud**, Hugging Face Spaces, or
any container platform:

- All paths are relative — no local-only paths.
- The model and preprocessor are loaded once via `st.cache_resource`.
- All data reads are cached via `st.cache_data`.
- `requirements.txt` pins every dependency needed on a fresh environment.

To deploy on Streamlit Community Cloud: push this folder to a GitHub repo, then
point Streamlit Cloud at `app.py`.

## 🧠 Model Summary

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **XGBoost** ⭐ | ~86.6% | ~83.7% | ~89.0% | ~86.3% |
| HistGradientBoosting | ~86.4% | ~83.6% | ~88.8% | ~86.1% |
| Random Forest | ~86.0% | ~82.4% | ~89.6% | ~85.8% |
| SVM | ~85.7% | ~82.6% | ~88.6% | ~85.5% |
| Gradient Boosting | ~85.8% | ~83.5% | ~87.4% | ~85.4% |
| K-Nearest Neighbors | ~83.4% | ~83.9% | ~80.4% | ~82.1% |
| Decision Tree | ~81.7% | ~78.0% | ~85.4% | ~81.6% |
| Logistic Regression | ~82.7% | ~82.7% | ~80.2% | ~81.4% |

*(Exact values regenerate slightly on retrain due to model stochasticity;
the live app always reflects whatever is currently stored in `model/`.)*

## 📊 Dataset

[Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro et al., 2014) — direct phone-call marketing campaigns from a Portuguese
banking institution. 11,162 records, 17 raw fields, target: whether the client
subscribed to a term deposit.

## 🛠️ Tech Stack

Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly
