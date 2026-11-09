"""
preprocessing.py — Proactive Retention Engine 2.0
Shared data loading, cleaning, feature engineering, and NLP preprocessing.
All functions are reusable by model_training.py, inference.py, and app.py.
"""

import pandas as pd
import numpy as np
import joblib
import random
import os
import warnings

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Numeric features
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Categorical features (all non-numeric columns except Churn & Customer_Support_Notes)
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
]

TABULAR_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MAX_SEQ_LENGTH = 15
RANDOM_STATE = 42
ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ──────────────────────────────────────────────────────────
def load_and_clean_data(url=DATA_URL):
    """Load the Telco Churn CSV, clean TotalCharges, and map Churn to 0/1."""
    print("  Loading dataset...")
    df = pd.read_csv(url)
    print(f"  Loaded successfully — shape: {df.shape}")

    # Drop customerID (not useful)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Ensure SeniorCitizen is string for consistent encoding
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    # Map target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print(f"  Churn distribution:\n{df['Churn'].value_counts().to_string()}")
    return df


# ──────────────────────────────────────────────────────────
# 2. SYNTHETIC TEXT GENERATION
# ──────────────────────────────────────────────────────────
NEGATIVE_NOTES = [
    "Terrible service, internet keeps dropping every day.",
    "Worst customer support ever, cancelling my plan.",
    "Hidden fees on my bill, very unhappy with this.",
    "Speeds are way slower than advertised, completely unacceptable.",
    "Been waiting weeks for a technician, absolutely frustrating.",
    "Constant outages and no one from support is helping.",
    "Overcharged again this month, considering switching providers.",
    "Connection drops during video calls, very unreliable service.",
]

POSITIVE_NOTES = [
    "Great service, very reliable and fast.",
    "Support team resolved my issue quickly and professionally.",
    "Happy with the internet speed and fair price.",
    "No issues at all, everything works fine.",
    "Upgraded my plan and the new speeds are excellent.",
    "Friendly support staff, always helpful when I call.",
    "Best ISP I have used, very satisfied customer.",
    "Stable connection and reasonable pricing, highly recommended.",
]


def generate_synthetic_notes(df):
    """Add a Customer_Support_Notes column using rule-based logic."""
    random.seed(RANDOM_STATE)

    def _pick_note(churn):
        return random.choice(NEGATIVE_NOTES if churn == 1 else POSITIVE_NOTES)

    df['Customer_Support_Notes'] = df['Churn'].apply(_pick_note)
    return df


# ──────────────────────────────────────────────────────────
# 3. TABULAR FEATURE ENGINEERING (Full Pipeline)
# ──────────────────────────────────────────────────────────
def build_tabular_features(df, fit=True, pipeline=None):
    """
    Process ALL tabular features: scale numerics + one-hot encode categoricals.
    Returns (X_processed ndarray, fitted pipeline, feature_names).
    If fit=False, uses a pre-fitted pipeline (for inference).
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    if fit:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
        ])
        pipeline = ColumnTransformer(transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ])
        X_processed = pipeline.fit_transform(X)

        # Determine feature names
        num_names = NUMERIC_FEATURES
        cat_encoder = pipeline.named_transformers_['cat'].named_steps['onehot']
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = num_names + cat_names

        # Save
        joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, 'tabular_pipeline.pkl'))
        joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, 'feature_names.pkl'))
        print(f"  Total tabular features: {len(feature_names)} "
              f"({len(NUMERIC_FEATURES)} numeric + {len(cat_names)} encoded categorical)")
        print("  Saved tabular_pipeline.pkl, feature_names.pkl")
    else:
        X_processed = pipeline.transform(X)
        feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_names.pkl'))

    return X_processed, pipeline, feature_names


# ──────────────────────────────────────────────────────────
# 4. NLP / TEXT FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────
def build_text_features(df, fit=True, tokenizer=None, max_len=MAX_SEQ_LENGTH):
    """
    Tokenize + pad Customer_Support_Notes.
    Returns (padded ndarray, fitted Tokenizer).
    If fit=False, uses a pre-fitted tokenizer (for inference).
    """
    texts = df['Customer_Support_Notes'].tolist()

    if fit:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        joblib.dump(tokenizer, os.path.join(ARTIFACTS_DIR, 'tokenizer.pkl'))
        print(f"  Vocabulary size: {len(tokenizer.word_index)}")
        print("  Saved tokenizer.pkl")

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded, tokenizer


# ──────────────────────────────────────────────────────────
# 5. CONVENIENCE: Full preprocessing pipeline
# ──────────────────────────────────────────────────────────
def run_full_preprocessing(url=DATA_URL):
    """
    Execute the complete preprocessing pipeline.
    Returns (df, X_tabular, X_text, y, pipeline, tokenizer, feature_names).
    """
    df = load_and_clean_data(url)
    df = generate_synthetic_notes(df)
    y = df['Churn'].values

    X_tabular, pipeline, feature_names = build_tabular_features(df, fit=True)
    X_text, tokenizer = build_text_features(df, fit=True)

    print(f"  Tabular shape: {X_tabular.shape}  |  Text shape: {X_text.shape}")
    return df, X_tabular, X_text, y, pipeline, tokenizer, feature_names
