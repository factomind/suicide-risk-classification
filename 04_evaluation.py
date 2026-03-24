"""
04_evaluation.py
================
CRISP-DM Stage 5 – Evaluation

Comprehensive model evaluation covering:
  • Per-class Precision / Recall / F1
  • Macro- and weighted-average F1
  • Confusion matrices (normalised & raw)
  • Cross-validation stability analysis (fold-wise F1 variance)
  • Error analysis (most-confused class pairs)

Usage
-----
    python 04_evaluation.py                # evaluate all (incl. BERT)
    python 04_evaluation.py --skip-bert    # skip BERT evaluation
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import (
    LABEL_ORDER, ID2LABEL, NUM_CLASSES, CV_FOLDS, RANDOM_STATE,
    MODEL_DIR, PLOT_DIR, RESULT_DIR,
)
from utils import logger

sns.set_theme(style="whitegrid", font_scale=1.05)


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def _save(fig, name: str) -> None:
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved plot → {path}")


def _get_bert_predictions(test_df: pd.DataFrame):
    """Load the best BERT checkpoint and predict on the test set."""
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_DIR / "bert_best"
    tokeniser = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    texts = test_df["cleaned_text_bert"].tolist()
    all_preds = []

    # Process in batches to avoid OOM
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokeniser(
            batch_texts, truncation=True, padding="max_length",
            max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(
                input_ids=encodings["input_ids"].to(device),
                attention_mask=encodings["attention_mask"].to(device),
            )
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


# ══════════════════════════════════════════════════════════════════════
#  1. CLASSIFICATION REPORT
# ══════════════════════════════════════════════════════════════════════
def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """
    Print and persist a full classification report.

    Returns a summary dict with macro-F1, weighted-F1, and accuracy
    for cross-model comparison.
    """
    report_str = classification_report(
        y_true, y_pred, target_names=LABEL_ORDER, digits=4, zero_division=0,
    )
    logger.info(f"\n{'─'*50}\n{model_name} – Classification Report\n{'─'*50}\n"
                f"{report_str}")

    report_dict = classification_report(
        y_true, y_pred, target_names=LABEL_ORDER,
        output_dict=True, zero_division=0,
    )

    # Save as CSV
    pd.DataFrame(report_dict).T.to_csv(
        RESULT_DIR / f"report_{model_name}.csv", float_format="%.4f",
    )

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════════
#  2. CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """
    Side-by-side normalised and raw confusion matrices.
    Normalised matrices reveal per-class recall at a glance (diagonal).
    """
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, cm, title, fmt in zip(
        axes,
        [cm_raw, cm_norm],
        [f"{model_name} – Raw Counts", f"{model_name} – Normalised (Recall)"],
        ["d", ".2f"],
    ):
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    _save(fig, f"07_cm_{model_name}")


# ══════════════════════════════════════════════════════════════════════
#  3. CROSS-VALIDATION STABILITY
# ══════════════════════════════════════════════════════════════════════
def cross_validation_stability(X_train, y_train, models: dict) -> pd.DataFrame:
    """
    Run Stratified K-Fold on each traditional model and report per-fold
    macro-F1.  Low variance across folds indicates a *stable* model,
    which is critical before trusting the SHAP / LIME explanations
    derived from it.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    records = []

    for name, model in models.items():
        fold_scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring="f1_macro", n_jobs=-1,
        )
        for i, score in enumerate(fold_scores, 1):
            records.append({"model": name, "fold": i, "macro_f1": score})
        logger.info(f"  {name}: mean F1={fold_scores.mean():.4f} "
                    f"± {fold_scores.std():.4f}")

    cv_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=cv_df, x="model", y="macro_f1", hue="model",
                palette="Set2", ax=ax, legend=False)
    sns.stripplot(data=cv_df, x="model", y="macro_f1", color="black",
                  size=6, ax=ax, legend=False)
    ax.set_title(f"Cross-Validation Stability ({CV_FOLDS}-Fold Macro-F1)")
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Model")
    _save(fig, "08_cv_stability")

    cv_df.to_csv(RESULT_DIR / "cv_stability.csv", index=False)
    return cv_df


# ══════════════════════════════════════════════════════════════════════
#  4. ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def error_analysis(test_df: pd.DataFrame, y_true, y_pred,
                   model_name: str) -> pd.DataFrame:
    """
    Identify the most common misclassification pairs and save example
    texts for qualitative review.

    This is especially important for adjacent risk levels (e.g.,
    Ideation vs. Behavior) where linguistic boundaries are subtle.
    """
    mask = y_true != y_pred
    errors = test_df[mask].copy()
    errors["true_label"] = [ID2LABEL[y] for y in y_true[mask]]
    errors["pred_label"] = [ID2LABEL[y] for y in y_pred[mask]]
    errors["pair"] = errors["true_label"] + " → " + errors["pred_label"]

    pair_counts = errors["pair"].value_counts().head(10)
    logger.info(f"\nTop misclassification pairs ({model_name}):\n"
                f"{pair_counts.to_string()}")

    # Save a sample of misclassified texts for manual inspection
    sample_errors = errors[["true_label", "pred_label", "text"]].head(20)
    sample_errors["text"] = sample_errors["text"].str[:300]  # truncate
    sample_errors.to_csv(
        RESULT_DIR / f"error_analysis_{model_name}.csv", index=False,
    )
    return errors


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("=" * 60)
    logger.info("STAGE 4 – EVALUATION")
    logger.info("=" * 60)

    # Load data & models
    data = joblib.load(MODEL_DIR / "prepared_data.joblib")
    train_df      = data["train_df"]
    test_df       = data["test_df"]
    X_train_tfidf = data["X_train_tfidf"]
    X_test_tfidf  = data["X_test_tfidf"]
    y_train = train_df["label_id"].values
    y_test  = test_df["label_id"].values

    # ── Evaluate traditional models ───────────────────────────────────
    summary_rows = []
    trad_models = {}

    for name in ["logistic_regression", "svm", "random_forest"]:
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        trad_models[name] = model
        y_pred = model.predict(X_test_tfidf)
        row = evaluate_model(y_test, y_pred, name)
        summary_rows.append(row)
        plot_confusion_matrix(y_test, y_pred, name)
        error_analysis(test_df, y_test, y_pred, name)

    # ── Cross-validation stability (traditional only) ─────────────────
    logger.info("\nRunning cross-validation stability analysis ...")
    cross_validation_stability(X_train_tfidf, y_train, trad_models)

    # ── Evaluate BERT ─────────────────────────────────────────────────
    if "--skip-bert" not in sys.argv and (MODEL_DIR / "bert_best").exists():
        logger.info("\nEvaluating BERT ...")
        y_pred_bert = _get_bert_predictions(test_df)
        row = evaluate_model(y_test, y_pred_bert, "bert")
        summary_rows.append(row)
        plot_confusion_matrix(y_test, y_pred_bert, "bert")
        error_analysis(test_df, y_test, y_pred_bert, "bert")
    else:
        logger.info("BERT evaluation skipped (no checkpoint found or --skip-bert).")

    # ── Summary table ─────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.round(4)
    summary_df.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    logger.info(f"\nModel comparison:\n{summary_df.to_string(index=False)}")
    logger.info("✓ Evaluation stage complete.")


if __name__ == "__main__":
    main()
