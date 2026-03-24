# Suicide Risk Classification from Reddit Posts

**Using Text Mining and Explainable AI (CRISP-DM)**

Nicholas Justin Tanuwijaya · Benedictus William Fabiano Sukresna  
Universitas Bina Nusantara — Computer Science — 2026

---

## Project Structure

```
suicide_risk_classification/
│
├── config.py                      # All hyperparameters, paths, constants
├── utils.py                       # Shared text-processing utilities
│
├── 01_data_understanding.py       # EDA: distribution, text stats, n-grams, sentiment
├── 02_data_preparation.py         # Cleaning, TF-IDF, train/test split
├── 03_modeling.py                 # LR, SVM, RF, BERT fine-tuning
├── 04_evaluation.py               # Metrics, confusion matrix, CV stability
├── 05_knowledge_extraction.py     # SHAP (global) + LIME (local) explanations
├── 06_dashboard.py                # Streamlit knowledge-visualisation dashboard
│
├── dataset.csv                    # Reddit suicide-risk dataset (500 posts)
├── requirements.txt               # Python dependencies
└── outputs/
    ├── plots/                     # All generated visualisations (.png, .html)
    ├── models/                    # Saved models & artefacts (.joblib, BERT checkpoint)
    └── results/                   # CSV tables (metrics, SHAP features, etc.)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (stages are meant to be executed in order)

```bash
# Stage 1 — Exploratory Data Analysis
python 01_data_understanding.py

# Stage 2 — Data Preparation (cleaning, TF-IDF, split)
python 02_data_preparation.py

# Stage 3 — Model Training
python 03_modeling.py              # full (with BERT, needs GPU)
python 03_modeling.py --skip-bert  # traditional ML only (CPU-friendly)

# Stage 4 — Evaluation
python 04_evaluation.py
python 04_evaluation.py --skip-bert

# Stage 5 — Knowledge Extraction (SHAP + LIME)
python 05_knowledge_extraction.py

# Stage 6 — Launch the dashboard
streamlit run 06_dashboard.py
```

### 3. View results

- **Plots** → `outputs/plots/`
- **Metric tables** → `outputs/results/`
- **Interactive dashboard** → `http://localhost:8501` (after `streamlit run`)

## CRISP-DM Methodology Mapping

| CRISP-DM Stage        | Script(s)                    | Key Outputs                        |
| ---------------------- | ---------------------------- | ---------------------------------- |
| Business Understanding | `config.py`                  | Label schema, parameters           |
| Data Understanding     | `01_data_understanding.py`   | Distribution, text stats, n-grams  |
| Data Preparation       | `02_data_preparation.py`     | Cleaned text, TF-IDF, splits       |
| Modeling               | `03_modeling.py`             | LR, SVM, RF, BERT                  |
| Evaluation             | `04_evaluation.py`           | F1 scores, confusion matrices, CV  |
| Knowledge Extraction   | `05_knowledge_extraction.py` | SHAP global, LIME local, insights  |
| Knowledge Visualisation| `06_dashboard.py`            | Streamlit dashboard                |

## Models

| Model                | Type           | Features      | Interpretable? |
| -------------------- | -------------- | ------------- | -------------- |
| Logistic Regression  | Linear         | TF-IDF        | ✅ High         |
| Linear SVM           | Max-margin     | TF-IDF        | ✅ Medium       |
| Random Forest        | Tree ensemble  | TF-IDF        | ✅ Medium       |
| BERT (fine-tuned)    | Transformer    | Token embeds  | ⚠️ Needs XAI   |

## Notes

- **Class imbalance**: Handled via `class_weight="balanced"` in all traditional
  models and weighted CrossEntropy loss in BERT.
- **No GPU?** Pass `--skip-bert` to stages 3 and 4 to run the full pipeline
  on CPU with the three traditional models.
- **SHAP on SVM**: Uses `KernelExplainer` (slow but model-agnostic). The
  `LinearExplainer` is used for Logistic Regression (exact & fast), and
  `TreeExplainer` for Random Forest (exact & fast).
- **LIME HTML files**: Individual explanation pages are saved in `outputs/plots/`
  and can be viewed directly in a browser or inside the dashboard.

## Dataset

The dataset contains 500 Reddit posts annotated into five risk categories
referencing the Columbia-Suicide Severity Rating Scale (C-SSRS):

| Class       | Count | Description                                   |
| ----------- | ----- | --------------------------------------------- |
| Supportive  | 108   | Empathetic/helpful responses to at-risk users  |
| Indicator   | 99    | Posts showing early warning signs              |
| Ideation    | 171   | Posts expressing suicidal thoughts             |
| Behavior    | 77    | Posts describing self-harm behaviours          |
| Attempt     | 45    | Posts describing suicide attempts              |
