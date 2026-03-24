"""
02_data_preparation.py
======================
CRISP-DM Stage 3 – Data Preparation

Transforms raw text into model-ready features:
  1. Text cleaning (case folding, tokenisation, lemmatisation)
  2. Feature extraction
     a. TF-IDF  → for Logistic Regression, SVM, Random Forest
     b. BERT tokenisation → for fine-tuning (handled in 03_modeling.py)
  3. Imbalance handling strategy selection
  4. Train / test split (stratified)
  5. Persist artefacts (vectoriser, split indices) for reproducibility

Usage
-----
    python 02_data_preparation.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF,
    TEST_SIZE, RANDOM_STATE, MODEL_DIR, RESULT_DIR, LABEL_ORDER,
)
from utils import load_dataset, clean_text, clean_text_minimal, logger


# ══════════════════════════════════════════════════════════════════════
#  1. CLEAN & TRANSFORM TEXT
# ══════════════════════════════════════════════════════════════════════
def prepare_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce two cleaned variants of every post:

    • `cleaned_text`  – Full pipeline (lowercase → remove noise →
      remove stopwords → lemmatise).  Used as input for TF-IDF.
    • `cleaned_text_bert` – Light cleaning only (lowercase → remove
      noise).  Preserves stopwords and word forms so that the BERT
      tokeniser receives near-natural language.
    """
    logger.info("Applying full cleaning pipeline (TF-IDF) ...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    logger.info("Applying minimal cleaning pipeline (BERT) ...")
    df["cleaned_text_bert"] = df["text"].apply(clean_text_minimal)

    return df


# ══════════════════════════════════════════════════════════════════════
#  2. TF-IDF VECTORISATION
# ══════════════════════════════════════════════════════════════════════
def build_tfidf(train_texts: pd.Series,
                test_texts: pd.Series):
    """
    Fit a TF-IDF vectoriser on the *training set only* (to prevent
    data leakage), then transform both splits.

    Returns
    -------
    X_train_tfidf, X_test_tfidf : sparse matrices
    vectoriser : fitted TfidfVectorizer (persisted for SHAP later)
    """
    vectoriser = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,           # log-scale TF → reduces impact of
                                     # very long posts that repeat terms
    )

    X_train = vectoriser.fit_transform(train_texts)
    X_test = vectoriser.transform(test_texts)

    logger.info(f"TF-IDF vocabulary size : {len(vectoriser.vocabulary_)}")
    logger.info(f"TF-IDF matrix (train)  : {X_train.shape}")
    logger.info(f"TF-IDF matrix (test)   : {X_test.shape}")

    # Persist for reproducibility & SHAP feature-name lookup
    joblib.dump(vectoriser, MODEL_DIR / "tfidf_vectoriser.joblib")
    return X_train, X_test, vectoriser


# ══════════════════════════════════════════════════════════════════════
#  3. STRATIFIED TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════
def split_dataset(df: pd.DataFrame):
    """
    80/20 stratified split.  Stratification is essential here because
    the Attempt class has only 45 samples—random splitting without
    stratification could leave the test set with very few (or zero)
    Attempt examples.

    Returns
    -------
    train_df, test_df : DataFrames with all columns preserved.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )
    logger.info(f"Train set : {len(train_df)}  |  Test set : {len(test_df)}")

    # Sanity-check: print per-class counts in each split
    for name, split in [("train", train_df), ("test", test_df)]:
        dist = split["Label"].value_counts().reindex(LABEL_ORDER)
        logger.info(f"  {name} → {dict(dist)}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  4. IMBALANCE HANDLING STRATEGY
# ══════════════════════════════════════════════════════════════════════
def compute_class_weights(y: pd.Series) -> dict:
    """
    Compute sklearn-compatible class weights using the 'balanced'
    formula:  w_j = n_samples / (n_classes * n_samples_j).

    These weights are passed to the `class_weight` parameter of
    Logistic Regression, SVM, and Random Forest.  They penalise
    misclassification of minority classes (Attempt, Behavior) more
    heavily, which is clinically desirable—failing to detect a
    high-risk individual is far worse than a false alarm.

    Returns
    -------
    dict  {label_id: weight}
    """
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight("balanced",
                                   classes=np.unique(y),
                                   y=y)
    weight_dict = dict(zip(np.unique(y), weights))
    logger.info(f"Class weights (balanced): {weight_dict}")
    return weight_dict


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("=" * 60)
    logger.info("STAGE 2 – DATA PREPARATION")
    logger.info("=" * 60)

    # 1. Load & clean
    df = load_dataset()
    df = prepare_texts(df)

    # 2. Split
    train_df, test_df = split_dataset(df)

    # 3. TF-IDF (fit on train only)
    X_train_tfidf, X_test_tfidf, vectoriser = build_tfidf(
        train_df["cleaned_text"], test_df["cleaned_text"]
    )

    # 4. Class weights
    class_weights = compute_class_weights(train_df["label_id"])

    # 5. Persist everything for downstream stages
    artefacts = {
        "train_df": train_df,
        "test_df": test_df,
        "X_train_tfidf": X_train_tfidf,
        "X_test_tfidf": X_test_tfidf,
        "class_weights": class_weights,
    }
    joblib.dump(artefacts, MODEL_DIR / "prepared_data.joblib")
    logger.info(f"All artefacts saved → {MODEL_DIR}")
    logger.info("✓ Data Preparation stage complete.")


if __name__ == "__main__":
    main()
