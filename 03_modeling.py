"""
03_modeling.py
==============
CRISP-DM Stage 4 – Modeling

Trains four classifiers on the Reddit suicide-risk dataset:
  A) Traditional ML (on TF-IDF features)
     1. Logistic Regression  – strong linear baseline, highly interpretable
     2. Support Vector Machine (linear) – max-margin classifier
     3. Random Forest – non-linear ensemble, feature importance for free
  B) Transformer
     4. BERT (bert-base-uncased) – fine-tuned end-to-end

All models use class weighting to handle the label imbalance.
Trained models are persisted so that evaluation (04) and explainability
(05) can reload them without re-training.

Usage
-----
    python 03_modeling.py                  # trains all four
    python 03_modeling.py --skip-bert      # traditional ML only (no GPU)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from config import (
    LR_MAX_ITER, SVM_C, RF_N_ESTIMATORS, RF_MAX_DEPTH,
    BERT_MODEL_NAME, BERT_MAX_LENGTH, BERT_BATCH_SIZE,
    BERT_LEARNING_RATE, BERT_EPOCHS, BERT_WARMUP_RATIO,
    BERT_WEIGHT_DECAY, NUM_CLASSES, LABEL2ID, ID2LABEL,
    MODEL_DIR, RANDOM_STATE,
)
from utils import logger


# ══════════════════════════════════════════════════════════════════════
#  LOAD PREPARED DATA
# ══════════════════════════════════════════════════════════════════════
def load_prepared_data():
    """Load artefacts saved by 02_data_preparation.py."""
    data = joblib.load(MODEL_DIR / "prepared_data.joblib")
    logger.info("Loaded prepared data artefacts.")
    return data


# ══════════════════════════════════════════════════════════════════════
#  A. TRADITIONAL ML MODELS
# ══════════════════════════════════════════════════════════════════════
def train_traditional_models(X_train, y_train, class_weights: dict):
    """
    Train LR, SVM, and RF on TF-IDF features.

    Notes
    -----
    • LinearSVC does not natively output probabilities (needed for LIME
      and SHAP KernelExplainer).  We wrap it with CalibratedClassifierCV
      so that `predict_proba()` is available.
    • All models use `class_weight="balanced"` to up-weight minority
      classes (Attempt, Behavior) during training.
    """
    models = {}

    # ── 1. Logistic Regression ────────────────────────────────────────
    logger.info("Training Logistic Regression ...")
    lr = LogisticRegression(
        max_iter=LR_MAX_ITER,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    # ── 2. SVM (linear, calibrated for probability output) ────────────
    logger.info("Training Linear SVM (with Platt calibration) ...")
    base_svm = LinearSVC(
        C=SVM_C,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train, y_train)
    models["svm"] = svm

    # ── 3. Random Forest ──────────────────────────────────────────────
    logger.info("Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # Persist
    for name, model in models.items():
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        logger.info(f"  Saved → {name}.joblib")

    return models


# ══════════════════════════════════════════════════════════════════════
#  B. BERT FINE-TUNING
# ══════════════════════════════════════════════════════════════════════
def train_bert(train_df: pd.DataFrame, test_df: pd.DataFrame,
               class_weights: dict):
    """
    Fine-tune BERT-base-uncased for multi-class suicide-risk
    classification.

    Key design choices
    ------------------
    1. **Weighted loss**: CrossEntropy with per-class weights derived
       from the training distribution, so the model is penalised more
       for misclassifying Attempt / Behavior.
    2. **Warm-up**: Linear warm-up for the first 10 % of training steps
       prevents large gradient updates before the classification head
       has adapted.
    3. **Max length 512**: Reddit posts can be very long; truncation to
       512 tokens is standard for BERT and retains the most salient
       opening content (users tend to express distress early in posts).
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        BertTokenizer, BertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"BERT device: {device}")

    # ── Tokeniser ─────────────────────────────────────────────────────
    tokeniser = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # ── Dataset class ─────────────────────────────────────────────────
    class RedditDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokeniser(
                texts, truncation=True, padding="max_length",
                max_length=BERT_MAX_LENGTH, return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels":         self.labels[idx],
            }

    train_ds = RedditDataset(
        train_df["cleaned_text_bert"].tolist(),
        train_df["label_id"].tolist(),
    )
    test_ds = RedditDataset(
        test_df["cleaned_text_bert"].tolist(),
        test_df["label_id"].tolist(),
    )

    train_loader = DataLoader(train_ds, batch_size=BERT_BATCH_SIZE,
                              shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BERT_BATCH_SIZE)

    # ── Model ─────────────────────────────────────────────────────────
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # ── Weighted loss ─────────────────────────────────────────────────
    weight_tensor = torch.tensor(
        [class_weights[i] for i in range(NUM_CLASSES)],
        dtype=torch.float,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    # ── Optimiser & scheduler ─────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=BERT_LEARNING_RATE,
        weight_decay=BERT_WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(total_steps * BERT_WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────
    logger.info(f"Starting BERT training ({BERT_EPOCHS} epochs, "
                f"{total_steps} total steps) ...")

    best_val_loss = float("inf")

    for epoch in range(1, BERT_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimiser.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                val_loss += loss_fn(outputs.logits, labels).item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(test_loader)
        val_acc = correct / total

        logger.info(f"  Epoch {epoch}/{BERT_EPOCHS}  "
                    f"Train Loss: {avg_train_loss:.4f}  "
                    f"Val Loss: {avg_val_loss:.4f}  "
                    f"Val Acc: {val_acc:.4f}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(MODEL_DIR / "bert_best")
            tokeniser.save_pretrained(MODEL_DIR / "bert_best")
            logger.info("    ↳ New best model saved.")

    logger.info("BERT training complete.")
    return model, tokeniser


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("=" * 60)
    logger.info("STAGE 3 – MODELING")
    logger.info("=" * 60)

    data = load_prepared_data()
    train_df      = data["train_df"]
    test_df       = data["test_df"]
    X_train_tfidf = data["X_train_tfidf"]
    class_weights = data["class_weights"]

    # A. Traditional ML
    y_train = train_df["label_id"].values
    models = train_traditional_models(X_train_tfidf, y_train, class_weights)

    # B. BERT (skip with --skip-bert flag when no GPU is available)
    if "--skip-bert" in sys.argv:
        logger.info("Skipping BERT (--skip-bert flag detected).")
    else:
        train_bert(train_df, test_df, class_weights)

    logger.info("✓ Modeling stage complete.")


if __name__ == "__main__":
    main()
