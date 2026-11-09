"""
pbl2_prep.py — Proactive Retention Engine 2.0
Thin entry-point script: runs the preprocessing pipeline with clear
phased output for presentation screenshots, then kicks off expert training.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from preprocessing import (
    load_and_clean_data,
    generate_synthetic_notes,
    build_tabular_features,
    build_text_features,
    TABULAR_FEATURES,
    MAX_SEQ_LENGTH,
)


def main():
    # ── PHASE 1: Data Loading & Setup ──
    print("=" * 60)
    print("PHASE 1: Data Loading & Setup")
    print("=" * 60)
    df = load_and_clean_data()
    print("Phase 1 complete.\n")

    # ── PHASE 2: Tabular Feature Engineering & Scaling ──
    print("=" * 60)
    print("PHASE 2: Tabular Feature Scaling & Encoding")
    print("=" * 60)
    X_tabular, pipeline, feature_names = build_tabular_features(df, fit=True)
    print(f"  Raw tabular features ({len(TABULAR_FEATURES)}): {TABULAR_FEATURES}")
    print(f"  Encoded features ({len(feature_names)}): {feature_names[:5]}...")
    print(f"  Engineered shape: {X_tabular.shape}")
    print("Phase 2 complete.\n")

    # ── PHASE 3: Unstructured Data Generation ──
    print("=" * 60)
    print("PHASE 3: Unstructured Data Generation")
    print("=" * 60)
    df = generate_synthetic_notes(df)
    sample_churn = df[df['Churn'] == 1]['Customer_Support_Notes'].iloc[0]
    sample_stay  = df[df['Churn'] == 0]['Customer_Support_Notes'].iloc[0]
    print(f'  Churned → "{sample_churn}"')
    print(f'  Retained → "{sample_stay}"')
    print("Phase 3 complete.\n")

    # ── PHASE 4: NLP Preprocessing ──
    print("=" * 60)
    print("PHASE 4: NLP Preprocessing")
    print("=" * 60)
    X_text, tokenizer = build_text_features(df, fit=True)
    idx = 0
    print(f"  Example: '{df['Customer_Support_Notes'].iloc[idx]}'")
    print(f"  Padded:  {X_text[idx]}")
    print("Phase 4 complete.\n")

    print("=" * 60)
    print("All preprocessing phases complete!")
    print("Run `python model_training.py` to train the 4 Expert models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
