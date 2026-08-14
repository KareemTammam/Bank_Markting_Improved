"""
Training script for the Bank Marketing Deposit Prediction project.

This reproduces the EXACT preprocessing pipeline, train/test split, and model
configurations from the original notebook (bank_marketing_classification_fixed.ipynb).
Hyperparameters used here are the best ones already found in the notebook's
GridSearchCV runs, so results match the original analysis without re-running
the (slow) grid search.

Run once: `python train.py`
Produces everything under model/ that the Streamlit app needs.
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/bank.csv")

# ---------------------------------------------------------------------------
# 2. Preprocessing (identical to the notebook)
# ---------------------------------------------------------------------------
df = df.drop(columns=["previous"])

X = df.drop(columns=["deposit"]).copy()
y = df["deposit"].map({"no": 0, "yes": 1})

binary_columns = ["default", "housing", "loan"]
X[binary_columns] = X[binary_columns].apply(lambda c: c.map({"no": 0, "yes": 1}))

education_column = ["education"]
education_order = [["primary", "secondary", "tertiary", "unknown"]]
onehot_columns = ["job", "marital", "contact", "month", "poutcome"]
numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays"]

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_columns),
        ("education", OrdinalEncoder(categories=education_order), education_column),
        ("numerical", StandardScaler(), numerical_columns),
    ],
    remainder="passthrough",
)

X_encoded = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()
X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Models, using the best hyperparameters already found via GridSearchCV
#    in the original notebook.
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, C=0.1, penalty="l2", solver="lbfgs"
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11, weights="distance"),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE, max_depth=10, min_samples_split=10
    ),
    "Random Forest": RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, n_estimators=200, max_depth=20, min_samples_leaf=2
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE, n_estimators=200, learning_rate=0.1, max_depth=4
    ),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
    ),
    "SVM": SVC(random_state=RANDOM_STATE, C=1, kernel="rbf", gamma="scale", probability=True),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_iter=300,
        learning_rate=0.05,
        max_depth=10,
        l2_regularization=0.0,
    ),
}

results = []
confusion_matrices = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        }
    )
    confusion_matrices[name] = confusion_matrix(y_test, pred).tolist()
    trained_models[name] = model
    print(f"Trained {name}")

results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------------------------
# 4. Feature importance for the best model (native importance if available)
# ---------------------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
else:
    # Permutation importance fallback
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": r.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------------
# 5. Target correlation (for EDA page) - computed on encoded features
# ---------------------------------------------------------------------------
target_corr = (
    X_encoded_df.assign(deposit=y.values)
    .corr()["deposit"]
    .drop("deposit")
    .sort_values(key=abs, ascending=False)
)

# ---------------------------------------------------------------------------
# 6. Persist everything the Streamlit app needs
# ---------------------------------------------------------------------------
with open("model/trained_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("model/feature_names.json", "w") as f:
    json.dump(list(feature_names), f)

results_df.to_json("model/model_comparison.json", orient="records")
fi_df.head(15).to_json("model/feature_importance.json", orient="records")

with open("model/confusion_matrices.json", "w") as f:
    json.dump(confusion_matrices, f)

meta = {
    "best_model_name": best_model_name,
    "n_samples": int(df.shape[0]),
    "n_features_raw": int(X.shape[1]),
    "n_features_encoded": int(X_encoded_df.shape[1]),
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "class_balance": y.value_counts(normalize=True).round(4).to_dict(),
    "class_counts": y.value_counts().to_dict(),
}
with open("model/meta.json", "w") as f:
    json.dump(meta, f)

target_corr.to_json("model/target_correlation.json")

# Raw category options for building the prediction form dynamically
raw_options = {
    "job": sorted(X["job"].unique().tolist()),
    "marital": sorted(X["marital"].unique().tolist()),
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "contact": sorted(X["contact"].unique().tolist()),
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": sorted(X["poutcome"].unique().tolist()),
}
with open("model/raw_options.json", "w") as f:
    json.dump(raw_options, f)

# Small sample of raw data for the dataset overview page
df.sample(min(200, len(df)), random_state=RANDOM_STATE).to_csv(
    "model/sample_data.csv", index=False
)

print("\nAll artifacts saved to model/")
