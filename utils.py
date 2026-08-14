"""Cached loaders for model artifacts and data."""

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "model"
DATA_DIR = Path(__file__).parent / "data"


@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_DIR / "trained_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    with open(MODEL_DIR / "preprocessor.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_feature_names():
    with open(MODEL_DIR / "feature_names.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_meta():
    with open(MODEL_DIR / "meta.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_comparison():
    return pd.read_json(MODEL_DIR / "model_comparison.json")


@st.cache_data(show_spinner=False)
def load_feature_importance():
    return pd.read_json(MODEL_DIR / "feature_importance.json")


@st.cache_data(show_spinner=False)
def load_confusion_matrices():
    with open(MODEL_DIR / "confusion_matrices.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_target_correlation():
    s = pd.read_json(MODEL_DIR / "target_correlation.json", typ="series")
    return s


@st.cache_data(show_spinner=False)
def load_raw_options():
    with open(MODEL_DIR / "raw_options.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_sample_data():
    return pd.read_csv(MODEL_DIR / "sample_data.csv")


def prettify_feature(name: str) -> str:
    """Turn 'onehot__job_admin.' into 'Job: Admin.'"""
    name = name.split("__", 1)[-1]
    for prefix, label in [
        ("job_", "Job: "),
        ("marital_", "Marital: "),
        ("contact_", "Contact: "),
        ("month_", "Month: "),
        ("poutcome_", "Prev. Outcome: "),
    ]:
        if name.startswith(prefix):
            return label + name[len(prefix):].replace("_", " ").title()
    mapping = {
        "education": "Education Level",
        "age": "Age",
        "balance": "Account Balance",
        "day": "Contact Day",
        "duration": "Call Duration",
        "campaign": "Number of Contacts (Campaign)",
        "pdays": "Days Since Previous Contact",
        "default": "Has Credit in Default",
        "housing": "Has Housing Loan",
        "loan": "Has Personal Loan",
    }
    return mapping.get(name, name.replace("_", " ").title())


def build_raw_input_row(form_values: dict) -> pd.DataFrame:
    """Build a single-row DataFrame matching the raw column order the
    preprocessor was fit on."""
    columns = [
        "age", "job", "marital", "education", "default", "balance",
        "housing", "loan", "contact", "day", "month", "duration",
        "campaign", "pdays", "poutcome",
    ]
    row = {col: form_values[col] for col in columns}
    return pd.DataFrame([row], columns=columns)
