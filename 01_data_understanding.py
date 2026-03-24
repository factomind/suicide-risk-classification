"""
01_data_understanding.py
========================
CRISP-DM Stage 2 – Data Understanding

Performs exploratory data analysis on the Reddit suicide-risk dataset:
  • Class distribution & imbalance ratio
  • Text-length statistics per risk class
  • Vocabulary size & lexical richness
  • Word-frequency analysis & n-gram extraction
  • Word-cloud generation per class
  • Sentiment polarity distribution (TextBlob)

All plots are saved to outputs/plots/ for later use in the Streamlit
dashboard and thesis write-up.

Usage
-----
    python 01_data_understanding.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob
from wordcloud import WordCloud

from config import (LABEL_ORDER, LABEL_COLORS, PLOT_DIR, RESULT_DIR,
                    TFIDF_NGRAM_RANGE)
from utils import load_dataset, clean_text, logger

sns.set_theme(style="whitegrid", font_scale=1.1)


# ══════════════════════════════════════════════════════════════════════
#  HELPER: save a matplotlib figure consistently
# ══════════════════════════════════════════════════════════════════════
def _save(fig, name: str) -> None:
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved plot → {path}")


# ══════════════════════════════════════════════════════════════════════
#  1. CLASS DISTRIBUTION & IMBALANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def analyse_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bar-chart of class counts + imbalance ratio table.

    The imbalance ratio is computed as  (majority_count / class_count).
    A ratio >> 1 signals that the class is under-represented and may
    require class-weighting or resampling during modelling.
    """
    counts = df["Label"].value_counts().reindex(LABEL_ORDER)

    # -- Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[LABEL_COLORS[l] for l in counts.index],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha="center", fontweight="bold")
    ax.set_title("Class Distribution of Suicide-Risk Labels", fontsize=14)
    ax.set_ylabel("Number of Samples")
    ax.set_xlabel("Risk Class")
    _save(fig, "01_class_distribution")

    # -- Imbalance ratio table
    majority = counts.max()
    stats = pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "percentage": (counts.values / counts.sum() * 100).round(2),
        "imbalance_ratio": (majority / counts.values).round(2),
    })
    stats.to_csv(RESULT_DIR / "class_distribution.csv", index=False)
    logger.info(f"Imbalance ratios:\n{stats.to_string(index=False)}")
    return stats


# ══════════════════════════════════════════════════════════════════════
#  2. TEXT-LENGTH STATISTICS
# ══════════════════════════════════════════════════════════════════════
def analyse_text_lengths(df: pd.DataFrame) -> None:
    """
    Box-plot + descriptive statistics of character and word counts
    per risk class.  Long posts in high-risk classes may indicate
    elaborate emotional expression (a potential linguistic signal).
    """
    df = df.copy()
    df["char_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    for metric, ylabel in [("char_len", "Character Count"),
                           ("word_count", "Word Count")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=df, x="Label", y=metric, hue="Label",
                    order=LABEL_ORDER, hue_order=LABEL_ORDER,
                    palette=LABEL_COLORS, ax=ax, showfliers=False,
                    legend=False)
        sns.stripplot(data=df, x="Label", y=metric, order=LABEL_ORDER,
                      color="black", alpha=0.25, size=3, ax=ax,
                      legend=False)
        ax.set_title(f"Text Length Distribution by Risk Class ({ylabel})")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Risk Class")
        _save(fig, f"02_text_length_{metric}")

    # Descriptive stats table
    stats = (df.groupby("Label")[["char_len", "word_count"]]
               .describe()
               .round(1))
    stats.to_csv(RESULT_DIR / "text_length_stats.csv")
    logger.info("Text-length analysis complete.")


# ══════════════════════════════════════════════════════════════════════
#  3. VOCABULARY SIZE & LEXICAL RICHNESS
# ══════════════════════════════════════════════════════════════════════
def analyse_lexical_richness(df: pd.DataFrame) -> None:
    """
    For each class, compute:
      - Total tokens
      - Unique tokens (vocabulary size)
      - Type-Token Ratio (TTR) = unique / total

    A *low* TTR implies repetitive language, which prior literature
    associates with ruminative thought patterns common in higher-risk
    individuals.
    """
    df = df.copy()
    df["tokens"] = df["cleaned_text"].str.split()
    df["token_count"] = df["tokens"].str.len()
    df["unique_count"] = df["tokens"].apply(lambda t: len(set(t)))
    df["ttr"] = df["unique_count"] / df["token_count"]

    richness = (df.groupby("Label")
                  .agg(
                      total_tokens=("token_count", "sum"),
                      mean_tokens=("token_count", "mean"),
                      mean_vocab=("unique_count", "mean"),
                      mean_ttr=("ttr", "mean"),
                  )
                  .reindex(LABEL_ORDER)
                  .round(3))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(richness.index, richness["mean_ttr"],
                  color=[LABEL_COLORS[l] for l in richness.index],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, richness["mean_ttr"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("Mean Lexical Richness (Type-Token Ratio) by Risk Class")
    ax.set_ylabel("Type-Token Ratio")
    ax.set_xlabel("Risk Class")
    ax.set_ylim(0, min(1.0, richness["mean_ttr"].max() + 0.1))
    _save(fig, "03_lexical_richness")

    richness.to_csv(RESULT_DIR / "lexical_richness.csv")
    logger.info(f"Lexical richness:\n{richness.to_string()}")


# ══════════════════════════════════════════════════════════════════════
#  4. WORD FREQUENCY & N-GRAM ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def analyse_word_frequency(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Extract the most frequent unigrams and bigrams per class.
    Results reveal the dominant vocabulary for each risk level—
    a first step toward identifying linguistic markers.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    for n, label in [(1, "unigram"), (2, "bigram")]:
        fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(24, 6),
                                 sharey=False)
        fig.suptitle(f"Top {top_n} {label.title()}s per Risk Class",
                     fontsize=15, y=1.02)

        for ax, cls in zip(axes, LABEL_ORDER):
            texts = df.loc[df["Label"] == cls, "cleaned_text"]
            vec = CountVectorizer(ngram_range=(n, n), max_features=top_n)
            X = vec.fit_transform(texts)
            freq = dict(zip(vec.get_feature_names_out(), X.sum(axis=0).A1))
            freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

            ax.barh(list(freq.keys())[::-1], list(freq.values())[::-1],
                    color=LABEL_COLORS[cls], edgecolor="black", linewidth=0.3)
            ax.set_title(cls, fontsize=12, fontweight="bold")
            ax.tick_params(axis="y", labelsize=9)

        plt.tight_layout()
        _save(fig, f"04_{label}_frequency")
    logger.info("Word-frequency & n-gram analysis complete.")


# ══════════════════════════════════════════════════════════════════════
#  5. WORD CLOUDS
# ══════════════════════════════════════════════════════════════════════
def generate_wordclouds(df: pd.DataFrame) -> None:
    """
    One word-cloud per risk class.  Visual inspection often reveals
    thematic clusters (e.g., "pain", "hopeless" in Attempt).
    """
    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(25, 5))
    fig.suptitle("Word Clouds by Risk Class", fontsize=15, y=1.02)

    for ax, cls in zip(axes, LABEL_ORDER):
        corpus = " ".join(df.loc[df["Label"] == cls, "cleaned_text"])
        wc = WordCloud(width=600, height=400, background_color="white",
                       colormap="Reds", max_words=150,
                       random_state=42).generate(corpus)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(cls, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    _save(fig, "05_wordclouds")
    logger.info("Word clouds generated.")


# ══════════════════════════════════════════════════════════════════════
#  6. SENTIMENT ANALYSIS (exploratory)
# ══════════════════════════════════════════════════════════════════════
def analyse_sentiment(df: pd.DataFrame) -> None:
    """
    Compute TextBlob polarity (−1 = negative … +1 = positive) and
    subjectivity (0 = objective … 1 = subjective) per post, then
    visualise distributions across risk classes.

    This is an *exploratory* step; advanced emotion features (NRC,
    VADER) may be added during data preparation if useful.
    """
    df = df.copy()
    df["polarity"] = df["text"].apply(lambda t: TextBlob(t).sentiment.polarity)
    df["subjectivity"] = df["text"].apply(lambda t: TextBlob(t).sentiment.subjectivity)

    for metric, title in [("polarity", "Sentiment Polarity"),
                          ("subjectivity", "Subjectivity Score")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.violinplot(data=df, x="Label", y=metric, hue="Label",
                       order=LABEL_ORDER, hue_order=LABEL_ORDER,
                       palette=LABEL_COLORS, inner="quartile", ax=ax,
                       legend=False)
        ax.set_title(f"{title} Distribution by Risk Class")
        ax.set_xlabel("Risk Class")
        ax.set_ylabel(title)
        _save(fig, f"06_sentiment_{metric}")

    sentiment_summary = (df.groupby("Label")[["polarity", "subjectivity"]]
                           .agg(["mean", "median", "std"])
                           .round(3)
                           .reindex(LABEL_ORDER))
    sentiment_summary.to_csv(RESULT_DIR / "sentiment_summary.csv")
    logger.info("Sentiment analysis complete.")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("=" * 60)
    logger.info("STAGE 1 – DATA UNDERSTANDING")
    logger.info("=" * 60)

    df = load_dataset()

    # Pre-compute cleaned text (used by several analyses)
    logger.info("Cleaning text for EDA (stopwords removed, lemmatised) ...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    analyse_class_distribution(df)
    analyse_text_lengths(df)
    analyse_lexical_richness(df)
    analyse_word_frequency(df)
    generate_wordclouds(df)
    analyse_sentiment(df)

    logger.info("✓ Data Understanding stage complete. "
                f"Plots → {PLOT_DIR}, Tables → {RESULT_DIR}")


if __name__ == "__main__":
    main()
