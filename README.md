# High-Dimensional Toxicity Classification Pipeline

This repository contains a robust, optimized machine learning pipeline designed to classify chemical/biological samples as **Toxic** or **NonToxic** based on molecular descriptors.

## 🔬 The Data Science Challenge: The $p \\gg n$ Curse

The dataset (Toxicity/CRY1 from the UCI Machine Learning Repository) presents an extreme dimensionality problem:
* **Features ($p$):** 1,203 molecular descriptors
* **Samples ($n$):** 171 observations
* **Class Imbalance:** 115 NonToxic vs. 56 Toxic

### Critical Diagnostic Discoveries
* **No Linear Signal:** An ANOVA F-test on the raw features reveals that only **18 features** have a $p$-value $< 0.05$ (by random chance, we would expect roughly 60 features to pass this threshold). This indicates the raw feature space is overwhelmed by noise.
* **SMOTE Pitfall:** Applying SMOTE in the high-dimensional space synthesizes samples by interpolating along noisy features, inflating the noise and causing the model to overfit heavily (In-sample F1 of 0.78 collapsing to 0.519 in CV).
* **Dimensionality Projection:** By applying Kernel Principal Component Analysis (Kernel PCA) with a Radial Basis Function (RBF) kernel to project the features down to **2 components**, we project away the noise and expose the stable, underlying non-linear biological signal.

---

## 📊 Performance Comparison

| Pipeline Architecture | CV Macro F1 Score | In-Sample Macro F1 | Generalization Status |
| :--- | :---: | :---: | :---: |
| **Baseline** (Lasso FS + SMOTE + Random Forest) | `0.5190` | `0.7800` | ⚠️ Severe Overfitting |
| **Linear PCA** (PCA [2 comp] + L1 Logistic Regression) | `0.5454` | `0.5600` | ✅ Stable, Good Generalization |
| **Optimized Champion** (Kernel PCA [2 comp, RBF] + L2 LR) | **`0.5572`** | `0.5700` | 🏆 Stable & Best Performing |

---

## 📁 Repository Structure

```
├── data/
│   └── data.csv                  # Toxicity dataset (included directly in the repo)
├── src/
│   ├── download_data.py          # Script to fetch dataset from UCI Repository
│   ├── feature_analysis.py       # ANOVA & Mutual Information diagnostics
│   ├── eda_and_experiment.py     # Initial algorithms benchmarks
│   ├── comprehensive_modeling.py # Tests PLS-DA, PCA, SVMs, Ridge, etc.
│   ├── grid_search.py            # Optimized hyperparameter grid search
│   ├── optimize_kpca.py          # Sequential hyperparameter optimization for Kernel PCA
│   └── pipeline.py               # Optimized production pipeline script
├── toxic_non_toxic_prediction.ipynb # Jupyter notebook documenting research
├── requirements.txt              # Project package dependencies
└── .gitignore                    # Git file exclusions
```

---

## 🚀 Quick Start

### 1. Setup Virtual Environment
Create and activate a Python virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Execute the Production Pipeline
The dataset is already included in `data/data.csv`. Execute the pipeline:
```bash
python src/pipeline.py
```

---

## 🧪 Experiment Scripts
You can run the experimental steps of the Data Science workflow individually:
* **Feature Diagnostics:** `python src/feature_analysis.py`
* **Baseline Benchmarking:** `python src/eda_and_experiment.py`
* **Advanced Architecture Benchmarking:** `python src/comprehensive_modeling.py`
* **Linear Grid Search:** `python src/grid_search.py`
* **Kernel PCA Grid Search:** `python src/optimize_kpca.py`
