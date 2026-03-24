"""
05_knowledge_extraction.py
==========================
CRISP-DM Stage 6 – Knowledge Extraction (Explainable AI)

The centrepiece of this thesis: extracting *interpretable linguistic
insights* from the trained models using Explainable AI techniques.

Methods
-------
A) **SHAP (SHapley Additive exPlanations)**
   - Global feature importance → which words/n-grams drive predictions
     across the entire dataset.
   - Per-class feature importance → dominant vocabulary per risk level.

B) **LIME (Local Interpretable Model-agnostic Explanations)**
   - Instance-level explanations → which words pushed *this specific
     post* toward a particular risk class.
   - Sampled from each class for qualitative review.

C) **Dominant-word extraction per class**
   - Top TF-IDF-weighted terms & top SHAP-valued terms per class,
     cross-referenced for triangulation.

Usage
-----
    python 05_knowledge_extraction.py
    python 05_knowledge_extraction.py --skip-bert
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

from config import (
    LABEL_ORDER, ID2LABEL, SHAP_MAX_DISPLAY, LIME_NUM_FEATURES,
    LIME_NUM_SAMPLES, MODEL_DIR, PLOT_DIR, RESULT_DIR, NUM_CLASSES,
)
from utils import logger


# ══════════════════════════════════════════════════════════════════════
#  A. SHAP – GLOBAL FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
def shap_analysis(model, X_train, X_test, feature_names: np.ndarray,
                  model_name: str) -> None:
    """
    Compute SHAP values for a traditional model and generate:
      1. Global summary bar plot (mean |SHAP| per feature)
      2. Per-class summary bar plot
      3. CSV of top features per class (for the dashboard)
    """
    import shap

    logger.info(f"Computing SHAP values for {model_name} ...")

    # Choose the right explainer based on model type
    if model_name == "logistic_regression":
        # LinearExplainer is exact for linear models
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
    elif model_name == "random_forest":
        # TreeExplainer is exact for tree ensembles
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        # KernelExplainer for any black-box (SVM, etc.)
        # Use a small background sample for speed
        bg = shap.sample(X_train, min(100, X_train.shape[0]))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(
            X_test[:50],      # limit test samples for speed
            nsamples=200,
        )

    # ── 1. Global summary bar plot ────────────────────────────────────
    # Aggregate mean |SHAP| across all classes
    if isinstance(shap_values, list):
        # Multi-class: list of arrays, one per class
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values],
                          axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    # Ensure it is 1-D
    if hasattr(mean_abs, 'A1'):
        mean_abs = mean_abs.A1
    mean_abs = np.asarray(mean_abs).flatten()

    top_idx = np.argsort(mean_abs)[-SHAP_MAX_DISPLAY:]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(feature_names[top_idx], mean_abs[top_idx], color="#3498db")
    ax.set_title(f"SHAP – Global Feature Importance ({model_name})")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"09_shap_global_{model_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Per-class top features ─────────────────────────────────────
    if isinstance(shap_values, list) and len(shap_values) == NUM_CLASSES:
        class_top = {}
        for cls_idx, cls_name in enumerate(LABEL_ORDER):
            sv = shap_values[cls_idx]
            if hasattr(sv, 'toarray'):
                sv = sv.toarray()
            sv = np.asarray(sv)
            class_mean = np.abs(sv).mean(axis=0).flatten()
            top = np.argsort(class_mean)[-SHAP_MAX_DISPLAY:]
            class_top[cls_name] = list(
                zip(feature_names[top][::-1],
                    class_mean[top][::-1].round(5))
            )

        # Save as a readable CSV
        rows = []
        for cls, feats in class_top.items():
            for rank, (feat, val) in enumerate(feats, 1):
                rows.append({"class": cls, "rank": rank,
                             "feature": feat, "mean_abs_shap": val})
        pd.DataFrame(rows).to_csv(
            RESULT_DIR / f"shap_top_features_{model_name}.csv", index=False,
        )

        # Per-class bar plots
        fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(25, 7), sharey=False)
        fig.suptitle(f"SHAP – Top Features per Class ({model_name})",
                     fontsize=14, y=1.02)
        for ax, cls_name in zip(axes, LABEL_ORDER):
            feats = class_top[cls_name][:15]
            names = [f[0] for f in feats][::-1]
            vals  = [f[1] for f in feats][::-1]
            ax.barh(names, vals, color="#e74c3c")
            ax.set_title(cls_name, fontweight="bold")
            ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        fig.savefig(PLOT_DIR / f"10_shap_perclass_{model_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"SHAP analysis complete for {model_name}.")


# ══════════════════════════════════════════════════════════════════════
#  B. LIME – LOCAL EXPLANATIONS
# ══════════════════════════════════════════════════════════════════════
def lime_analysis(model, vectoriser, test_df: pd.DataFrame,
                  y_test: np.ndarray, model_name: str,
                  n_samples_per_class: int = 2) -> None:
    """
    Generate LIME explanations for a handful of representative posts
    from each risk class (correct predictions only, for interpretive
    clarity).

    Each explanation shows which words *locally* pushed the prediction
    toward or away from the true class.
    """
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(
        class_names=LABEL_ORDER,
        split_expression=r"\W+",
        random_state=42,
    )

    # Prediction function for LIME: raw text → probability vector
    def predict_fn(texts):
        transformed = vectoriser.transform(texts)
        return model.predict_proba(transformed)

    y_pred = model.predict(vectoriser.transform(test_df["cleaned_text"]))

    all_explanations = []

    for cls_idx, cls_name in enumerate(LABEL_ORDER):
        # Pick correctly-predicted samples for clean explanations
        mask = (y_test == cls_idx) & (y_pred == cls_idx)
        candidates = test_df[mask]

        if len(candidates) == 0:
            logger.warning(f"No correct predictions for {cls_name}, skipping.")
            continue

        sample = candidates.sample(
            n=min(n_samples_per_class, len(candidates)),
            random_state=42,
        )

        for idx, row in sample.iterrows():
            exp = explainer.explain_instance(
                row["cleaned_text"],
                predict_fn,
                num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_SAMPLES,
                labels=[cls_idx],
            )

            # Save HTML for interactive inspection
            html_path = PLOT_DIR / f"11_lime_{model_name}_{cls_name}_{idx}.html"
            exp.save_to_file(str(html_path))

            # Collect top features for CSV export
            for feat, weight in exp.as_list(label=cls_idx):
                all_explanations.append({
                    "model": model_name,
                    "class": cls_name,
                    "sample_idx": idx,
                    "feature": feat,
                    "weight": round(weight, 5),
                })

    lime_df = pd.DataFrame(all_explanations)
    lime_df.to_csv(RESULT_DIR / f"lime_explanations_{model_name}.csv",
                   index=False)
    logger.info(f"LIME analysis complete for {model_name}. "
                f"HTML files saved to {PLOT_DIR}")


# ══════════════════════════════════════════════════════════════════════
#  C. DOMINANT-WORD EXTRACTION PER CLASS
# ══════════════════════════════════════════════════════════════════════
def extract_dominant_words(train_df: pd.DataFrame,
                           vectoriser, top_n: int = 30) -> pd.DataFrame:
    """
    For each risk class, identify the top-N words by *mean TF-IDF
    weight*.  These are the terms that are most distinctive for each
    class in the feature space.

    This complements SHAP by providing a model-agnostic view of
    class-specific vocabulary.
    """
    feature_names = vectoriser.get_feature_names_out()
    X = vectoriser.transform(train_df["cleaned_text"])

    records = []
    for cls_name in LABEL_ORDER:
        mask = train_df["Label"] == cls_name
        X_cls = X[mask.values]
        mean_tfidf = np.asarray(X_cls.mean(axis=0)).flatten()
        top_idx = np.argsort(mean_tfidf)[-top_n:][::-1]

        for rank, i in enumerate(top_idx, 1):
            records.append({
                "class": cls_name,
                "rank": rank,
                "word": feature_names[i],
                "mean_tfidf": round(mean_tfidf[i], 5),
            })

    dom_df = pd.DataFrame(records)
    dom_df.to_csv(RESULT_DIR / "dominant_words_tfidf.csv", index=False)
    logger.info("Dominant-word extraction (TF-IDF) complete.")
    return dom_df


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("=" * 60)
    logger.info("STAGE 5 – KNOWLEDGE EXTRACTION (Explainable AI)")
    logger.info("=" * 60)

    # Load artefacts
    data = joblib.load(MODEL_DIR / "prepared_data.joblib")
    train_df      = data["train_df"]
    test_df       = data["test_df"]
    X_train_tfidf = data["X_train_tfidf"]
    X_test_tfidf  = data["X_test_tfidf"]
    y_test        = test_df["label_id"].values

    vectoriser = joblib.load(MODEL_DIR / "tfidf_vectoriser.joblib")
    feature_names = vectoriser.get_feature_names_out()

    # ── Dominant words (model-agnostic) ───────────────────────────────
    extract_dominant_words(train_df, vectoriser)

    # ── SHAP & LIME for each traditional model ───────────────────────
    # Focus SHAP and LIME on the best-performing traditional model
    # (typically Logistic Regression for interpretability), but also
    # run SHAP on Random Forest for comparison.
    for name in ["logistic_regression", "random_forest"]:
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        shap_analysis(model, X_train_tfidf, X_test_tfidf,
                      feature_names, name)

    # LIME on the Logistic Regression (most interpretable baseline)
    lr_model = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    lime_analysis(lr_model, vectoriser, test_df, y_test,
                  "logistic_regression")

    # ── SHAP for SVM as well ──────────────────────────────────────────
    svm_model = joblib.load(MODEL_DIR / "svm.joblib")
    lime_analysis(svm_model, vectoriser, test_df, y_test, "svm")

    logger.info("✓ Knowledge Extraction stage complete.")


if __name__ == "__main__":
    main()
