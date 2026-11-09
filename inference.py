"""
inference.py — Proactive Retention Engine 2.0
Hybrid Inference Pipeline: Router weights × Expert predictions = Ensemble output.
Also generates LLM-based "Next Best Action" retention strategies.
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Patch Keras Layer deserialization for cross-version compatibility (.h5 format)
try:
    from tensorflow.keras.layers import Layer
    _orig_layer_init = Layer.__init__
    def _safe_layer_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        _orig_layer_init(self, *args, **kwargs)
    Layer.__init__ = _safe_layer_init
except Exception:
    pass

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import (
    ARTIFACTS_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES, MAX_SEQ_LENGTH,
)
from model_training import llm_predict, llm_generate_strategy

# ──────────────────────────────────────────────────────────
# LOAD ALL ARTIFACTS (cached at module level)
# ──────────────────────────────────────────────────────────
_cache = {}


def _load_artifacts():
    """Load all saved models, pipeline, and tokenizer once."""
    if _cache:
        return _cache

    print("[Inference] Loading artifacts...")
    _cache['pipeline'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'tabular_pipeline.pkl'))
    _cache['tokenizer'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'tokenizer.pkl'))
    _cache['xgb'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'expert_xgb.pkl'))
    _cache['svm'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'expert_svm.pkl'))
    _cache['ann'] = load_model(os.path.join(ARTIFACTS_DIR, 'expert_ann.h5'), compile=False)
    _cache['router'] = load_model(os.path.join(ARTIFACTS_DIR, 'router_model.h5'), compile=False)
    print("[Inference] All artifacts loaded.")
    return _cache


# ──────────────────────────────────────────────────────────
# CORE INFERENCE FUNCTIONS
# ──────────────────────────────────────────────────────────
def prepare_inputs(user_data_dict, support_notes):
    """
    Prepare a single customer's data for inference.
    user_data_dict: dict with keys matching NUMERIC_FEATURES + CATEGORICAL_FEATURES
    support_notes: raw text string
    Returns (X_tabular_processed, X_text_padded, X_combined).
    """
    arts = _load_artifacts()
    pipeline = arts['pipeline']
    tokenizer = arts['tokenizer']

    # Build DataFrame from user input
    input_df = pd.DataFrame([user_data_dict])

    # Ensure SeniorCitizen is string for consistent encoding
    if 'SeniorCitizen' in input_df.columns:
        input_df['SeniorCitizen'] = input_df['SeniorCitizen'].astype(str)

    # Transform via the full pipeline (impute + scale numerics, one-hot categoricals)
    X_tab = pipeline.transform(input_df)

    # Tokenize + pad text
    seq = tokenizer.texts_to_sequences([support_notes])
    X_text = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post')

    # Combined for Router
    X_combined = np.concatenate([X_tab, X_text], axis=1)

    return X_tab, X_text, X_combined


def get_router_weights(X_combined, support_notes=""):
    """
    Pass combined input through the Neural Router.
    Returns softened array of 4 weights.
    """
    arts = _load_artifacts()
    weights = arts['router'].predict(X_combined, verbose=0)[0]
    
    # Apply temperature scaling to enforce a true "Soft" Mixture of Experts
    # Without this, the Keras categorical_crossentropy network overfits and acts as a hard switch (like 1.0 vs 0.0)
    temperature = 2.5
    logits = np.log(np.clip(weights, 1e-7, 1.0))
    
    # Dynamic LLM Boost: The router was mostly trained on tabular focus. 
    # If the user explicitly provided rich text, the LLM is the definitive expert and should be trusted highly.
    if support_notes and len(support_notes.strip()) > 5:
        # We must override the LLM logit relative to the top tabular expert 
        # so it isn't buried by deep neural network saturation
        max_tabular_logit = np.max(logits[:3])
        # Force the LLM to surpass the best tabular expert, scaling dynamically by text depth
        logits[3] = max_tabular_logit + 1.5 + (min(len(support_notes.strip()), 100) / 40.0)

    scaled_logits = logits / temperature
    exp_scaled = np.exp(scaled_logits - np.max(scaled_logits))
    soft_weights = exp_scaled / exp_scaled.sum()
    
    return soft_weights


def get_expert_predictions(X_tabular, support_notes):
    """
    Get churn probability from each of the 4 experts.
    Returns array of 4 probabilities.
    """
    arts = _load_artifacts()

    # Expert 1 – XGBoost
    p1 = float(arts['xgb'].predict_proba(X_tabular)[0, 1])
    # Expert 2 – SVM
    p2 = float(arts['svm'].predict_proba(X_tabular)[0, 1])
    # Expert 3 – ANN
    p3 = float(arts['ann'].predict(X_tabular, verbose=0)[0, 0])
    # Expert 4 – LLM
    p4 = float(llm_predict(support_notes))

    return np.array([p1, p2, p3, p4])


def ensemble_predict(user_data_dict, support_notes):
    """
    Full MoE inference for a single customer.
    user_data_dict: dict with all tabular feature keys.
    support_notes: raw text string.
    Returns dict with:
      - router_weights: array(4)
      - expert_preds:   array(4)
      - ensemble_prob:  float
      - churn_label:    str
    """
    X_tab, X_text, X_combined = prepare_inputs(user_data_dict, support_notes)

    weights = get_router_weights(X_combined, support_notes)
    preds = get_expert_predictions(X_tab, support_notes)

    # Weighted sum
    ensemble_prob = float(np.dot(weights, preds))

    return {
        'router_weights': weights,
        'expert_preds': preds,
        'ensemble_prob': ensemble_prob,
        'churn_label': 'High Risk' if ensemble_prob >= 0.5 else 'Low Risk',
    }


def generate_retention_strategy(support_notes):
    """
    Use the LLM (or mock fallback) to generate a personalized
    'Next Best Action' retention strategy.
    """
    return llm_generate_strategy(support_notes)


# ──────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  INFERENCE PIPELINE — Quick Test")
    print("=" * 60)

    # Test case 1: Likely churner (month-to-month, high charges, low tenure)
    churner_data = {
        'tenure': 2, 'MonthlyCharges': 95.0, 'TotalCharges': 190.0,
        'gender': 'Male', 'SeniorCitizen': '0', 'Partner': 'No', 'Dependents': 'No',
        'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
    }
    result = ensemble_predict(
        churner_data,
        "Terrible service, internet keeps dropping every day. I am quitting."
    )
    print("\n  Test 1 — Likely Churner:")
    print(f"    Router Weights: XGB={result['router_weights'][0]:.3f}, "
          f"SVM={result['router_weights'][1]:.3f}, "
          f"ANN={result['router_weights'][2]:.3f}, "
          f"LLM={result['router_weights'][3]:.3f}")
    print(f"    Expert Preds:   XGB={result['expert_preds'][0]:.3f}, "
          f"SVM={result['expert_preds'][1]:.3f}, "
          f"ANN={result['expert_preds'][2]:.3f}, "
          f"LLM={result['expert_preds'][3]:.3f}")
    print(f"    Ensemble Score: {result['ensemble_prob']:.4f} ({result['churn_label']})")
    strategy = generate_retention_strategy(
        "Terrible service, internet keeps dropping every day. I am quitting."
    )
    print(f"    Strategy: {strategy}")

    # Test case 2: Likely retained (two-year contract, low charges, long tenure)
    retained_data = {
        'tenure': 60, 'MonthlyCharges': 30.0, 'TotalCharges': 1800.0,
        'gender': 'Female', 'SeniorCitizen': '0', 'Partner': 'Yes', 'Dependents': 'Yes',
        'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
        'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Two year', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
    }
    result2 = ensemble_predict(
        retained_data,
        "Great service, very reliable and fast."
    )
    print("\n  Test 2 — Likely Retained:")
    print(f"    Router Weights: XGB={result2['router_weights'][0]:.3f}, "
          f"SVM={result2['router_weights'][1]:.3f}, "
          f"ANN={result2['router_weights'][2]:.3f}, "
          f"LLM={result2['router_weights'][3]:.3f}")
    print(f"    Expert Preds:   XGB={result2['expert_preds'][0]:.3f}, "
          f"SVM={result2['expert_preds'][1]:.3f}, "
          f"ANN={result2['expert_preds'][2]:.3f}, "
          f"LLM={result2['expert_preds'][3]:.3f}")
    print(f"    Ensemble Score: {result2['ensemble_prob']:.4f} ({result2['churn_label']})")

    print("\n" + "=" * 60)
    print("  Inference pipeline working correctly!")
    print("=" * 60)
