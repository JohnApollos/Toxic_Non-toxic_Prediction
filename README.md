
# High-Dimensional Toxicity Classification Pipeline

This repository contains a robust machine learning pipeline designed to classify chemical/biological samples as **Toxic** or **NonToxic**. 

##  The Data Science Challenge: The $p \gg n$ Problem
This dataset presents a classic "Curse of Dimensionality" problem:
* **Features ($p$):** 1,204
* **Samples ($n$):** 171
* **Class Imbalance:** 115 NonToxic vs. 56 Toxic

Because there are vastly more features than observations, standard machine learning models will immediately overfit to the noise and memorize the training data. This pipeline implements strict regularization and data augmentation to find true mathematical signals.

##  Methodology

To prevent overfitting and handle the class imbalance, this pipeline utilizes:
1. **Standardization:** `StandardScaler` to normalize all 1,200+ features.
2. **Aggressive Feature Selection:** An L1-Regularized (Lasso) Logistic Regression wrapper. By applying a strong penalty (`C=0.1`), the algorithm forces the coefficients of over 98% of the features to exactly `0.0`, discarding useless noise and reducing the feature space from 1,204 down to roughly 23 meaningful variables.
3. **Data Augmentation:** `SMOTE` (Synthetic Minority Over-sampling Technique) generates synthetic examples of the minority class ("Toxic") during training to prevent the model from becoming biased toward the majority class.
4. **Ensemble Modeling:** A `RandomForestClassifier` with strictly limited tree depth (`max_depth=3`) to ensure generalizability.
5. **Robust Evaluation:** Hyperparameters are optimized using `GridSearchCV` wrapped in a `RepeatedStratifiedKFold` (5 splits, 3 repeats) to ensure the reported metrics are highly robust and not dependent on a lucky random seed.

##  Quick Start

### Prerequisites
Requires Python 3.9+. Install the dependencies:
```bash
pip install -r requirements.txt
```
### Running the Pipeline

Place your `data.csv` file inside the `data/` directory, then execute the script:
```bash
python src/pipeline.py
```
## Results Summary

-   **Dimensionality Reduction:** The L1 feature selector successfully reduced the dataset from 1,204 features to the **top 23 statistically significant features**.
    
-   **Cross-Validated Macro F1 Score:** `~0.519`
    
-   **Interpretation:** The cross-validated metric represents the mathematical ceiling of the current dataset. The aggressive regularization successfully prevented the model from collapsing under the noise of the 1,200 useless features, but the underlying data signal for toxicity is inherently weak. Future improvements require increasing the sample size ($n$) rather than algorithmic tuning.
