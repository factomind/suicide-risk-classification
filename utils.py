"""
utils.py
========
Shared helpers that are imported by every pipeline stage.
Keeping them here avoids code duplication and guarantees consistent
text pre-processing across EDA, modelling, and explainability steps.
"""

import ast
import re
import logging
from typing import List

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import DATA_PATH, LABEL_ORDER

# ──────────────────────────────────────────────────────────────────────
# Logging setup  (one shared logger for the whole project)
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("suicide_risk")

# ──────────────────────────────────────────────────────────────────────
# NLTK resources  (downloaded once, then cached)
# ──────────────────────────────────────────────────────────────────────
_NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                   "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]

def ensure_nltk_resources() -> None:
    """Download any missing NLTK data bundles quietly."""
    for res in _NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenize/{res}" if "punkt" in res else res)
        except LookupError:
            nltk.download(res, quiet=True)

ensure_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ──────────────────────────────────────────────────────────────────────
# Data loading & parsing
# ──────────────────────────────────────────────────────────────────────
def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load the Reddit suicide-risk dataset and parse the Post column.

    The raw Post column is a *string representation* of a Python list
    (e.g. "['post_1', 'post_2']").  This function safely evaluates it
    and joins the individual posts into a single text per user.

    Returns
    -------
    pd.DataFrame with columns: User, Post (raw), text (cleaned join),
    label_id (integer-encoded).
    """
    path = path or DATA_PATH
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} cols")

    # Parse list-of-strings → single text
    df["text"] = df["Post"].apply(_parse_post_column)

    # Integer-encode labels following the severity order
    label_map = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    df["label_id"] = df["Label"].map(label_map)

    logger.info(f"Label distribution:\n{df['Label'].value_counts().to_string()}")
    return df


def _parse_post_column(raw: str) -> str:
    """
    Safely convert the stringified list into one concatenated document.
    Falls back to the raw string if literal_eval fails.
    """
    try:
        posts: List[str] = ast.literal_eval(raw)
        return " ".join(posts)
    except (ValueError, SyntaxError):
        return str(raw)


# ──────────────────────────────────────────────────────────────────────
# Text cleaning pipeline
# ──────────────────────────────────────────────────────────────────────
def clean_text(text: str,
               remove_stopwords: bool = True,
               lemmatize: bool = True) -> str:
    """
    Full text normalisation pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove HTML entities & tags
      4. Remove non-alphabetic characters (keeps spaces)
      5. Tokenise
      6. (Optional) Remove English stopwords
      7. (Optional) Lemmatise

    Parameters
    ----------
    text : str
        Raw social-media text.
    remove_stopwords : bool
        Whether to strip NLTK English stopwords.
    lemmatize : bool
        Whether to apply WordNet lemmatisation.

    Returns
    -------
    str – cleaned, space-joined tokens.
    """
    # Step 1: Case folding
    text = text.lower()

    # Step 2: Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Step 3: Remove HTML artefacts
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 4: Keep only alphabetic tokens
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 5: Tokenise (simple whitespace split – fast & sufficient here)
    tokens = text.split()

    # Step 6: Stopword removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # Step 7: Lemmatisation
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_text_minimal(text: str) -> str:
    """
    Light cleaning (no stopword removal, no lemmatisation).
    Useful for BERT inputs where the tokeniser needs natural language.
    """
    return clean_text(text, remove_stopwords=False, lemmatize=False)
