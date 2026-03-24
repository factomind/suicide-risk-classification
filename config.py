"""
config.py
=========
Central configuration for the Suicide Risk Classification project.
All hyperparameters, paths, and constants live here so that every module
shares a single source of truth.

CRISP-DM Reference: Business Understanding – defines scope & parameters.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 1. PATHS
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULT_DIR = OUTPUT_DIR / "results"

# Create directories on import
for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 2. LABEL SCHEMA  (based on Columbia-Suicide Severity Rating Scale)
# ──────────────────────────────────────────────────────────────────────
# Ordered from lowest to highest clinical severity so that any ordinal
# analysis or visualisation respects the risk gradient.
LABEL_ORDER = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]
NUM_CLASSES = len(LABEL_ORDER)

# Mapping for model training (label -> integer)
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# ──────────────────────────────────────────────────────────────────────
# 3. DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5                     # Stratified K-Fold for stability analysis

# TF-IDF settings
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)       # Unigrams + bigrams
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# ──────────────────────────────────────────────────────────────────────
# 4. MODELING – Traditional ML
# ──────────────────────────────────────────────────────────────────────
# All traditional models use class_weight="balanced" to address imbalance.
LR_MAX_ITER = 2000
SVM_KERNEL = "linear"            # Linear kernel for interpretability
SVM_C = 1.0
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None

# ──────────────────────────────────────────────────────────────────────
# 5. MODELING – BERT Fine-tuning
# ──────────────────────────────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE = 8
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 5
BERT_WARMUP_RATIO = 0.1
BERT_WEIGHT_DECAY = 0.01

# ──────────────────────────────────────────────────────────────────────
# 6. EXPLAINABILITY (SHAP / LIME)
# ──────────────────────────────────────────────────────────────────────
SHAP_MAX_DISPLAY = 20            # Top-N features in SHAP summary plot
LIME_NUM_FEATURES = 15           # Features shown per LIME explanation
LIME_NUM_SAMPLES = 2000          # Perturbation samples for LIME

# ──────────────────────────────────────────────────────────────────────
# 7. VISUALISATION PALETTE
# ──────────────────────────────────────────────────────────────────────
# Colour-blind-friendly palette aligned to severity gradient.
LABEL_COLORS = {
    "Supportive": "#2ecc71",     # green  – lowest risk
    "Indicator":  "#f39c12",     # amber
    "Ideation":   "#e67e22",     # orange
    "Behavior":   "#e74c3c",     # red
    "Attempt":    "#8e44ad",     # purple – highest risk
}
