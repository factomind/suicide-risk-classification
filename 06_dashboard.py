"""
06_dashboard.py
===============
CRISP-DM Stage 7 – Knowledge Visualisation

Interactive Streamlit dashboard that brings together all the analysis
outputs from stages 1–5 into a single explorable interface.

Sections
--------
1. Overview       – dataset stats, class distribution
2. EDA            – text lengths, lexical richness, sentiment
3. Word Analysis  – n-grams, word clouds, dominant words
4. Model Results  – comparison table, confusion matrices
5. Explainability – SHAP global importance, LIME local explanations
6. Prediction     – live prediction on user-entered text

Usage
-----
    streamlit run 06_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

from config import (
    LABEL_ORDER, LABEL_COLORS, ID2LABEL, NUM_CLASSES,
    MODEL_DIR, PLOT_DIR, RESULT_DIR,
)

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Suicide Risk – Knowledge Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def load_image(name: str):
    """Load a PNG from the plots directory, or return None."""
    path = PLOT_DIR / f"{name}.png"
    if path.exists():
        return Image.open(path)
    return None


def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV from the results directory, or return an empty DF."""
    path = RESULT_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_prediction_artefacts():
    """Load the TF-IDF vectoriser and Logistic Regression model."""
    vec = joblib.load(MODEL_DIR / "tfidf_vectoriser.joblib")
    model = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    return vec, model


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════
st.sidebar.title("🧠 Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "📊 Overview",
        "🔍 EDA – Text Analysis",
        "💬 Word Analysis",
        "🤖 Model Results",
        "🔬 Explainability (XAI)",
        "🎯 Live Prediction",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Suicide Risk Classification from Reddit Posts**\n\n"
    "CRISP-DM Knowledge Discovery Pipeline\n\n"
    "Nicholas J. Tanuwijaya & Benedictus W. F. Sukresna\n\n"
    "Universitas Bina Nusantara – 2026"
)


# ══════════════════════════════════════════════════════════════════════
#  1. OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if section == "📊 Overview":
    st.title("📊 Dataset Overview")
    st.markdown(
        "This dashboard presents the knowledge extraction results from "
        "a Reddit suicide-risk classification study following the **CRISP-DM** "
        "methodology. The dataset contains **500 Reddit posts** labelled into "
        "five risk classes based on the Columbia-Suicide Severity Rating Scale "
        "(C-SSRS) framework."
    )

    dist_df = load_csv("class_distribution")
    if not dist_df.empty:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            img = load_image("01_class_distribution")
            if img:
                st.image(img, use_container_width=True)
        with col2:
            st.subheader("Class Distribution")
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
            majority = dist_df["count"].max()
            minority = dist_df["count"].min()
            st.metric("Imbalance Ratio (majority / minority)",
                      f"{majority / minority:.2f}x")
    else:
        st.info("Run `01_data_understanding.py` first to generate EDA outputs.")


# ══════════════════════════════════════════════════════════════════════
#  2. EDA – TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif section == "🔍 EDA – Text Analysis":
    st.title("🔍 Text-Level Exploratory Analysis")

    tab1, tab2, tab3 = st.tabs(["📏 Text Length", "📚 Lexical Richness",
                                 "💭 Sentiment"])

    with tab1:
        st.subheader("Text Length Distribution by Risk Class")
        st.markdown(
            "Longer posts may indicate more elaborate emotional expression. "
            "Box plots show the spread; strip points show individual samples."
        )
        for metric in ["char_len", "word_count"]:
            img = load_image(f"02_text_length_{metric}")
            if img:
                st.image(img, use_container_width=True)

    with tab2:
        st.subheader("Lexical Richness (Type-Token Ratio)")
        st.markdown(
            "A **low TTR** suggests repetitive language, which prior "
            "research associates with ruminative thought patterns common "
            "in higher-risk individuals."
        )
        img = load_image("03_lexical_richness")
        if img:
            st.image(img, use_container_width=True)
        lr_df = load_csv("lexical_richness")
        if not lr_df.empty:
            st.dataframe(lr_df, use_container_width=True)

    with tab3:
        st.subheader("Sentiment Polarity & Subjectivity")
        for metric in ["polarity", "subjectivity"]:
            img = load_image(f"06_sentiment_{metric}")
            if img:
                st.image(img, use_container_width=True)
        sent_df = load_csv("sentiment_summary")
        if not sent_df.empty:
            st.dataframe(sent_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  3. WORD ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif section == "💬 Word Analysis":
    st.title("💬 Word & N-gram Analysis")

    tab1, tab2, tab3 = st.tabs(["🔤 Unigrams", "🔤🔤 Bigrams", "☁️ Word Clouds"])

    with tab1:
        img = load_image("04_unigram_frequency")
        if img:
            st.image(img, use_container_width=True)

    with tab2:
        img = load_image("04_bigram_frequency")
        if img:
            st.image(img, use_container_width=True)

    with tab3:
        img = load_image("05_wordclouds")
        if img:
            st.image(img, use_container_width=True)

    st.subheader("Dominant Words per Class (TF-IDF)")
    dom_df = load_csv("dominant_words_tfidf")
    if not dom_df.empty:
        selected_class = st.selectbox("Select risk class", LABEL_ORDER)
        filtered = dom_df[dom_df["class"] == selected_class].head(20)
        st.dataframe(filtered, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
#  4. MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════
elif section == "🤖 Model Results":
    st.title("🤖 Model Comparison & Evaluation")

    comp_df = load_csv("model_comparison")
    if not comp_df.empty:
        st.subheader("Performance Summary")
        st.dataframe(
            comp_df.style.highlight_max(
                subset=["accuracy", "macro_f1", "weighted_f1"],
                color="#d4edda",
            ),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Confusion Matrices")
    model_sel = st.selectbox(
        "Select model",
        ["logistic_regression", "svm", "random_forest", "bert"],
    )
    img = load_image(f"07_cm_{model_sel}")
    if img:
        st.image(img, use_container_width=True)
    else:
        st.warning(f"No confusion matrix found for '{model_sel}'.")

    st.subheader("Cross-Validation Stability")
    img = load_image("08_cv_stability")
    if img:
        st.image(img, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  5. EXPLAINABILITY (XAI)
# ══════════════════════════════════════════════════════════════════════
elif section == "🔬 Explainability (XAI)":
    st.title("🔬 Explainable AI – Knowledge Extraction")

    tab1, tab2, tab3 = st.tabs(["🌍 SHAP Global", "🏠 SHAP Per-Class",
                                 "🔎 LIME Local"])

    with tab1:
        st.subheader("SHAP – Global Feature Importance")
        st.markdown(
            "These are the words and n-grams with the highest average "
            "impact on model predictions across the entire test set."
        )
        model_sel = st.selectbox(
            "Model", ["logistic_regression", "random_forest"],
            key="shap_global",
        )
        img = load_image(f"09_shap_global_{model_sel}")
        if img:
            st.image(img, use_container_width=True)

    with tab2:
        st.subheader("SHAP – Top Features per Risk Class")
        st.markdown(
            "Class-specific SHAP values reveal which words are most "
            "important for distinguishing each risk level."
        )
        model_sel = st.selectbox(
            "Model", ["logistic_regression", "random_forest"],
            key="shap_class",
        )
        img = load_image(f"10_shap_perclass_{model_sel}")
        if img:
            st.image(img, use_container_width=True)

        shap_df = load_csv(f"shap_top_features_{model_sel}")
        if not shap_df.empty:
            cls = st.selectbox("Filter by class", LABEL_ORDER, key="shap_tbl")
            st.dataframe(
                shap_df[shap_df["class"] == cls].head(20),
                use_container_width=True, hide_index=True,
            )

    with tab3:
        st.subheader("LIME – Local Explanations")
        st.markdown(
            "LIME explains individual predictions by highlighting "
            "which words pushed the model toward or away from the "
            "predicted class."
        )
        lime_df = load_csv("lime_explanations_logistic_regression")
        if not lime_df.empty:
            cls = st.selectbox("Filter by class", LABEL_ORDER, key="lime_cls")
            filtered = lime_df[lime_df["class"] == cls]
            if not filtered.empty:
                st.dataframe(filtered, use_container_width=True,
                             hide_index=True)
            else:
                st.info(f"No LIME explanations available for {cls}.")

        # Show LIME HTML files if available
        lime_files = sorted(PLOT_DIR.glob("11_lime_*.html"))
        if lime_files:
            sel_file = st.selectbox(
                "View LIME HTML explanation",
                lime_files,
                format_func=lambda f: f.stem,
            )
            with open(sel_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=500, scrolling=True)


# ══════════════════════════════════════════════════════════════════════
#  6. LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════
elif section == "🎯 Live Prediction":
    st.title("🎯 Live Risk Prediction")
    st.markdown(
        "Enter a text below to see the model's predicted risk class "
        "and the confidence for each label. This uses the Logistic "
        "Regression model with TF-IDF features."
    )
    st.warning(
        "⚠️ **Disclaimer**: This is a research prototype, NOT a clinical "
        "diagnostic tool. If you or someone you know is in crisis, please "
        "contact a mental health professional or crisis helpline."
    )

    user_text = st.text_area("Paste or type a Reddit-style post here:",
                             height=200)

    if st.button("Predict", type="primary") and user_text.strip():
        from utils import clean_text

        vec, model = load_prediction_artefacts()
        cleaned = clean_text(user_text)
        X = vec.transform([cleaned])
        proba = model.predict_proba(X)[0]
        pred_idx = proba.argmax()
        pred_label = LABEL_ORDER[pred_idx]

        st.subheader(f"Predicted Class: **{pred_label}**")

        # Show probability bars
        prob_df = pd.DataFrame({
            "Risk Class": LABEL_ORDER,
            "Probability": proba,
        })
        st.bar_chart(prob_df.set_index("Risk Class"), height=300)

        # Quick LIME explanation for this input
        try:
            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(
                class_names=LABEL_ORDER, split_expression=r"\W+",
            )
            def predict_fn(texts):
                return model.predict_proba(vec.transform(texts))

            exp = explainer.explain_instance(
                cleaned, predict_fn, num_features=10,
                num_samples=1000, labels=[pred_idx],
            )
            st.subheader("LIME Explanation (top contributing words)")
            lime_data = exp.as_list(label=pred_idx)
            lime_df = pd.DataFrame(lime_data, columns=["Word", "Weight"])
            st.dataframe(lime_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"LIME explanation failed: {e}")
