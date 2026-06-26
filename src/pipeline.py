"""
Toxicity Classification Pipeline
--------------------------------
This script handles a high-dimensional, low-sample-size (p >> n) dataset to classify 
compounds/samples as 'Toxic' or 'NonToxic'.

Based on exhaustive Data Science benchmarking and hyperparameter optimization, the champion model is:
- Scaler: StandardScaler
- Dimension Reduction: Kernel PCA (RBF kernel, 2 components, gamma=0.0005)
- Classifier: L2-regularized Logistic Regression (C=0.5, solver='liblinear', class_weight='balanced')
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.pipeline import Pipeline

def load_and_preprocess(filepath):
    """Loads data and encodes the target variable."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Class distribution: {np.bincount(y_encoded)} (0: {le.classes_[0]}, 1: {le.classes_[1]})")
    
    return X, y_encoded, le

def build_pipeline():
    """Constructs the optimized Kernel PCA + Logistic Regression pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        # Project raw features into a non-linear 2D space to filter out noise
        ('kpca', KernelPCA(n_components=2, kernel='rbf', gamma=0.0005, random_state=42)),
        # L2-regularized Logistic Regression to perform classification
        ('classifier', LogisticRegression(
            penalty='l2', solver='liblinear', class_weight='balanced', C=0.5, random_state=42
        ))
    ])

def main():
    # 1. Load Data
    X, y, label_encoder = load_and_preprocess('../data/data.csv')

    # 2. Setup Pipeline
    pipeline = build_pipeline()
    
    # 5x3 Repeated Stratified K-Fold for robust evaluation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scorer = make_scorer(f1_score, average='macro')

    # 3. Execute Training & Cross-Validation
    print("\nExecuting Cross-Validation on the optimized pipeline...")
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=1) # n_jobs=1 for thread-safety on Windows
    
    print("\n" + "="*40)
    print("OPTIMIZED PIPELINE PERFORMANCE")
    print("="*40)
    print(f"Cross-Validated Macro F1 Score: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"(Baseline CV Macro F1 was: 0.5190)")

    # 4. Final Fit and In-Sample Diagnostics
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    print("\nFinal Model Classification Report (In-Sample):")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))

if __name__ == "__main__":
    main()