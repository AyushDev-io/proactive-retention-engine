"""
model_training.py — Proactive Retention Engine 2.0
Trains, evaluates, and saves the 4 Expert models + the Neural Router
for the Soft MoE architecture.

Expert 1: XGBoost Classifier       (Tabular)
Expert 2: Support Vector Machine    (Tabular)
Expert 3: Deep Keras ANN            (Tabular)
Expert 4: Local Quantized LLM       (Text — with mock fallback)
Router  : Keras Gating Network      (Tabular + Text → Softmax(4))
"""

import os
import sys
import numpy as np
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from scipy.special import softmax as scipy_softmax

from preprocessing import (
    run_full_preprocessing,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ARTIFACTS_DIR,
    RANDOM_STATE,
    NEGATIVE_NOTES,
)

# ══════════════════════════════════════════════════════════
# EXPERT 1 — XGBoost Classifier
# ══════════════════════════════════════════════════════════
def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("EXPERT 1: XGBoost Classifier")
    print("=" * 60)

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
    )
    xgb.fit(X_train, y_train)

    preds = xgb.predict(X_test)
    proba = xgb.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds, target_names=['Retained', 'Churned']))
    print(f"  Recall: {recall_score(y_test, preds):.4f}")
    print(f"  Sample probabilities (first 5): {np.round(proba[:5], 4)}")

    path = os.path.join(ARTIFACTS_DIR, 'expert_xgb.pkl')
    joblib.dump(xgb, path)
    print(f"  Saved → {path}")
    return xgb


# ══════════════════════════════════════════════════════════
# EXPERT 2 — Support Vector Machine
# ══════════════════════════════════════════════════════════
def train_svm(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("EXPERT 2: Support Vector Machine (SVM)")
    print("=" * 60)

    svm = SVC(
        kernel='rbf',
        probability=True,
        random_state=RANDOM_STATE,
    )
    svm.fit(X_train, y_train)

    preds = svm.predict(X_test)
    proba = svm.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds, target_names=['Retained', 'Churned']))
    print(f"  Recall: {recall_score(y_test, preds):.4f}")
    print(f"  Sample probabilities (first 5): {np.round(proba[:5], 4)}")

    path = os.path.join(ARTIFACTS_DIR, 'expert_svm.pkl')
    joblib.dump(svm, path)
    print(f"  Saved → {path}")
    return svm


# ══════════════════════════════════════════════════════════
# EXPERT 3 — Deep Keras ANN
# ══════════════════════════════════════════════════════════
def train_ann(X_train, y_train, X_test, y_test, input_dim):
    print("\n" + "=" * 60)
    print("EXPERT 3: Deep Artificial Neural Network (Keras)")
    print("=" * 60)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluate
    proba = model.predict(X_test, verbose=0).flatten()
    preds = (proba >= 0.5).astype(int)

    print(classification_report(y_test, preds, target_names=['Retained', 'Churned']))
    print(f"  Recall: {recall_score(y_test, preds):.4f}")
    print(f"  Sample probabilities (first 5): {np.round(proba[:5], 4)}")

    path = os.path.join(ARTIFACTS_DIR, 'expert_ann.h5')
    model.save(path)
    print(f"  Saved → {path}")
    return model


# ══════════════════════════════════════════════════════════
# EXPERT 4 — Local Quantized LLM  (with mock fallback)
# ══════════════════════════════════════════════════════════

# ---------- Mock / Fallback ----------
_NEGATIVE_KEYWORDS = [
    'terrible', 'worst', 'hidden', 'slower', 'unacceptable',
    'frustrating', 'outages', 'overcharged', 'drops', 'cancelling',
    'unhappy', 'unreliable', 'waiting', 'quit', 'quitting',
    'cancel', 'leave', 'switch', 'bad', 'awful', 'poor', 'hate',
    'horrible', 'annoying', 'disappointed', 'problem', 'issue',
]


def mock_llm_predict(text):
    """
    Rule-based sentiment fallback.
    Returns a churn probability between 0 and 1.
    Scans for negative keywords — more matches → higher score.
    """
    text_lower = text.lower()
    hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)
    # Scale: 0 hits → 0.10,  1 hit → 0.45,  2+ hits → 0.70+
    if hits == 0:
        score = 0.10
    elif hits == 1:
        score = 0.45
    else:
        score = min(0.45 + hits * 0.15, 0.95)
    return round(score, 4)


def mock_llm_generate_strategy(text):
    """
    Rule-based retention strategy fallback.
    Returns a short 'Next Best Action' string.
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['price', 'fee', 'overcharg', 'bill', 'charge']):
        return "Offer a 15% loyalty discount on the next 3 billing cycles."
    elif any(kw in text_lower for kw in ['speed', 'slow', 'drop', 'outage', 'connection', 'internet']):
        return "Escalate to network engineering for priority line diagnostics and a free speed upgrade."
    elif any(kw in text_lower for kw in ['support', 'waiting', 'technician', 'help']):
        return "Assign a dedicated account manager for proactive follow-ups within 24 hours."
    elif any(kw in text_lower for kw in ['cancel', 'quit', 'quitting', 'switch', 'leave']):
        return "Initiate an urgent win-back call with an exclusive 12-month retention bundle at 20% off."
    elif any(kw in text_lower for kw in ['bad', 'worst', 'terrible', 'awful', 'horrible']):
        return "Offer a service credit and schedule a priority callback from senior support."
    else:
        return "Schedule a courtesy check-in call to proactively gather feedback and offer an upgrade."


# ---------- Real LLM (with auto-download) ----------
_llm_model = None

# Hugging Face model configuration
GGUF_REPO = "bartowski/Phi-3-mini-4k-instruct-GGUF"
GGUF_FILE = "Phi-3-mini-4k-instruct-Q4_K_M.gguf"


def _download_gguf(progress_callback=None):
    """
    Download the quantized GGUF model from Hugging Face if not already present.
    Returns the local path, or None on failure.
    """
    dest_path = os.path.join(ARTIFACTS_DIR, GGUF_FILE)
    if os.path.exists(dest_path):
        return dest_path

    try:
        from huggingface_hub import hf_hub_download
        if progress_callback:
            progress_callback("downloading")
        print(f"  [LLM] Downloading {GGUF_FILE} (~2.4 GB) from Hugging Face...")
        local = hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            local_dir=ARTIFACTS_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"  [LLM] Download complete → {local}")
        if progress_callback:
            progress_callback("done")
        return local
    except ImportError:
        print("  [LLM] huggingface_hub not installed — using mock fallback.")
        return None
    except Exception as e:
        print(f"  [LLM] Download failed: {e} — using mock fallback.")
        return None


def _load_llm(progress_callback=None):
    """
    Load an EXISTING local GGUF model. Does NOT auto-download.
    Tries ctransformers first, then llama-cpp-python, then returns None (→ mock).
    """
    global _llm_model
    if _llm_model is not None:
        return _llm_model

    # Check for existing .gguf files only — no auto-download
    gguf_files = [f for f in os.listdir(ARTIFACTS_DIR) if f.endswith('.gguf')]
    if not gguf_files:
        print("  [LLM] No .gguf model found — using mock fallback.")
        return None

    model_path = os.path.join(ARTIFACTS_DIR, gguf_files[0])
    if progress_callback:
        progress_callback("loading")
    print(f"  [LLM] Loading quantized model: {gguf_files[0]}...")

    # ── Attempt 1: ctransformers (prebuilt wheel, no C++ needed) ──
    try:
        from ctransformers import AutoModelForCausalLM
        _llm_model = AutoModelForCausalLM.from_pretrained(
            ARTIFACTS_DIR,
            model_file=gguf_files[0],
            model_type='llama',
            max_new_tokens=100,
            context_length=512,
            threads=4,
        )
        _llm_model._backend = 'ctransformers'
        print("  [LLM] Model loaded via ctransformers ✓")
        if progress_callback:
            progress_callback("ready")
        return _llm_model
    except ImportError:
        print("  [LLM] ctransformers not installed — trying llama-cpp-python...")
    except Exception as e:
        print(f"  [LLM] ctransformers failed: {e} — trying llama-cpp-python...")

    # ── Attempt 2: llama-cpp-python ──
    try:
        from llama_cpp import Llama
        _llm_model = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=4,
            verbose=False,
        )
        _llm_model._backend = 'llama_cpp'
        print("  [LLM] Model loaded via llama-cpp-python ✓")
        if progress_callback:
            progress_callback("ready")
        return _llm_model
    except ImportError:
        print("  [LLM] llama-cpp-python not installed — using mock fallback.")
        return None
    except Exception as e:
        print(f"  [LLM] Failed to load model: {e} — using mock fallback.")
        return None


def llm_predict(text):
    """Use the local LLM or mock fallback to estimate churn probability."""
    llm = _load_llm()
    if llm is None:
        return mock_llm_predict(text)

    prompt = (
        "You are a customer churn analyst. Given the following customer support note, "
        "estimate the probability (0.0 to 1.0) that this customer will churn. "
        "Reply with ONLY a single decimal number.\n\n"
        f"Note: \"{text}\"\n\nChurn probability:"
    )
    try:
        backend = getattr(llm, '_backend', 'unknown')
        if backend == 'ctransformers':
            response = llm(prompt, max_new_tokens=10, temperature=0.1)
        else:
            output = llm(prompt, max_tokens=10, temperature=0.1)
            response = output['choices'][0]['text']
        # Extract first float from response
        import re
        match = re.search(r'(\d+\.?\d*)', response.strip())
        if match:
            score = float(match.group(1))
            return round(max(0.0, min(score, 1.0)), 4)
        return mock_llm_predict(text)
    except Exception as e:
        print(f"  [LLM] Prediction error: {e}")
        return mock_llm_predict(text)


def llm_generate_strategy(text):
    """Use the local LLM or mock fallback to generate a retention strategy."""
    llm = _load_llm()
    if llm is None:
        return mock_llm_generate_strategy(text)

    prompt = (
        "You are a customer retention strategist. Based on this customer's support note, "
        "suggest ONE short, actionable 'Next Best Action' to prevent churn. "
        "Keep it to 1-2 sentences.\n\n"
        f"Note: \"{text}\"\n\nRetention strategy:"
    )
    try:
        backend = getattr(llm, '_backend', 'unknown')
        if backend == 'ctransformers':
            response = llm(prompt, max_new_tokens=100, temperature=0.7)
        else:
            output = llm(prompt, max_tokens=100, temperature=0.7)
            response = output['choices'][0]['text']
        result = response.strip()
        return result if result else mock_llm_generate_strategy(text)
    except Exception as e:
        print(f"  [LLM] Strategy error: {e}")
        return mock_llm_generate_strategy(text)


# ══════════════════════════════════════════════════════════
# NEURAL ROUTER — Soft MoE Gating Network
# ══════════════════════════════════════════════════════════
def build_router_targets(X_tabular, notes, y_true, xgb_model, svm_model, ann_model):
    """
    For each sample, compute SOFT targets based on how close each expert's
    prediction is to ground truth. Uses temperature-scaled softmax of
    inverse errors so all experts get proportional credit.
    Returns soft labels of shape (n_samples, 4).
    """
    n = len(y_true)

    # Expert 1 – XGBoost
    p1 = xgb_model.predict_proba(X_tabular)[:, 1]
    # Expert 2 – SVM
    p2 = svm_model.predict_proba(X_tabular)[:, 1]
    # Expert 3 – ANN
    p3 = ann_model.predict(X_tabular, verbose=0).flatten()
    # Expert 4 – LLM mock
    p4 = np.array([mock_llm_predict(note) for note in notes])

    preds = np.stack([p1, p2, p3, p4], axis=1)  # (n, 4)

    # Compute inverse errors → higher = better expert for this sample
    errors = np.abs(preds - y_true.reshape(-1, 1))  # (n, 4)
    
    # NOTE: Penalize the mock LLM error during training. 
    # Because our synthetic text is heavily correlated with the ground truth, 
    # the mock LLM appears 'too perfect' and dominates the weights.
    # Adding a penalty forces the Router to rely on the tabular experts equally.
    errors[:, 3] = errors[:, 3] + 0.20

    inv_errors = 1.0 / (errors + 0.05)  # small epsilon to avoid division by zero

    # Temperature-scaled softmax → soft targets (not hard one-hot)
    temperature = 1.5
    soft_targets = np.zeros((n, 4))
    for i in range(n):
        scaled = inv_errors[i] / temperature
        soft_targets[i] = scipy_softmax(scaled)

    # Print distribution info
    avg_weights = soft_targets.mean(axis=0)
    print(f"  Avg soft target weights: XGB={avg_weights[0]:.3f}, "
          f"SVM={avg_weights[1]:.3f}, ANN={avg_weights[2]:.3f}, LLM={avg_weights[3]:.3f}")
    return soft_targets


def train_router(X_tabular, X_text, router_targets):
    """
    Build and train the Keras Gating Network.
    Input:  concatenated [tabular + text_padded]
    Output: Softmax(4) — one weight per expert.
    """
    print("\n" + "=" * 60)
    print("NEURAL ROUTER: Soft MoE Gating Network")
    print("=" * 60)

    # Concatenate tabular + text into a single input vector
    X_combined = np.concatenate([X_tabular, X_text], axis=1)
    input_dim = X_combined.shape[1]
    print(f"  Combined input dimension: {input_dim} "
          f"(tabular={X_tabular.shape[1]} + text={X_text.shape[1]})")

    # Build the Gating Network
    inp = Input(shape=(input_dim,), name='router_input')
    x = Dense(64, activation='relu', name='router_dense1')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', name='router_dense2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(4, activation='softmax', name='router_softmax')(x)

    router = Model(inputs=inp, outputs=out, name='MoE_Router')
    router.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    router.summary()

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    router.fit(
        X_combined, router_targets,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    # Save
    path = os.path.join(ARTIFACTS_DIR, 'router_model.h5')
    router.save(path)
    print(f"  Saved → {path}")

    # Demonstrate on a few samples
    print("\n  Sample Router weight distributions (first 5):")
    sample_weights = router.predict(X_combined[:5], verbose=0)
    labels = ['XGB', 'SVM', 'ANN', 'LLM']
    for i, w in enumerate(sample_weights):
        parts = ' | '.join(f"{l}={v:.3f}" for l, v in zip(labels, w))
        total = sum(w)
        print(f"    Sample {i}: [{parts}]  sum={total:.4f}")

    return router


# ══════════════════════════════════════════════════════════
# MAIN — Training Orchestration
# ══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PROACTIVE RETENTION ENGINE 2.0 — EXPERT + ROUTER TRAINING")
    print("=" * 60)

    # ── Preprocessing ──
    print("\n▶ Running full preprocessing pipeline...")
    df, X_tabular, X_text, y, pipeline, tokenizer, feature_names = run_full_preprocessing()
    print(f"  Feature names ({len(feature_names)}): {feature_names[:5]} ... {feature_names[-3:]}")

    # ── Train/Test Split ──
    print("\n▶ Splitting data (80/20, stratified)...")
    (
        X_tab_train, X_tab_test,
        X_txt_train, X_txt_test,
        y_train, y_test,
    ) = train_test_split(
        X_tabular, X_text, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ── SMOTE (tabular only) ──
    print("▶ Applying SMOTE to tabular training set...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_tab_train_res, y_train_res = smote.fit_resample(X_tab_train, y_train)
    print(f"  Before SMOTE: {np.bincount(y_train)}")
    print(f"  After  SMOTE: {np.bincount(y_train_res)}")

    input_dim = X_tab_train_res.shape[1]

    # ── Train Experts 1-3 on tabular data ──
    xgb_model = train_xgboost(X_tab_train_res, y_train_res, X_tab_test, y_test)
    svm_model = train_svm(X_tab_train_res, y_train_res, X_tab_test, y_test)
    ann_model = train_ann(X_tab_train_res, y_train_res, X_tab_test, y_test, input_dim)

    # ── Expert 4: LLM sentiment on text test samples ──
    print("\n" + "=" * 60)
    print("EXPERT 4: Local Quantized LLM (Sentiment)")
    print("=" * 60)
    print("  Testing LLM / mock on 5 sample notes...")
    sample_notes = df['Customer_Support_Notes'].iloc[:5].tolist()
    sample_labels = df['Churn'].iloc[:5].tolist()
    for note, label in zip(sample_notes, sample_labels):
        score = llm_predict(note)
        strategy = llm_generate_strategy(note)
        print(f'  Label={label} | Score={score:.4f} | Note="{note[:50]}..."')
        print(f'    → Strategy: {strategy}')

    # ── Train Neural Router ──
    print("\n▶ Building Router training targets (soft labels)...")
    notes_train = df['Customer_Support_Notes'].iloc[:len(y_train)].tolist()
    router_targets = build_router_targets(
        X_tab_train, notes_train,
        y_train, xgb_model, svm_model, ann_model
    )
    router_model = train_router(X_tab_train, X_txt_train, router_targets)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  ALL EXPERTS + ROUTER TRAINED SUCCESSFULLY")
    print("=" * 60)
    saved = [f for f in os.listdir(ARTIFACTS_DIR)
             if f.endswith(('.pkl', '.h5'))]
    print("  Saved artifacts:")
    for f in sorted(saved):
        print(f"    • {f}")


if __name__ == "__main__":
    main()
