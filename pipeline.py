"""
Toxicity Classification Pipeline
--------------------------------
This script handles a high-dimensional, low-sample-size (p >> n) dataset to classify 
compounds/samples as 'Toxic' or 'NonToxic'. 

It implements a robust machine learning pipeline featuring:
1. Variance Thresholding & L1 (Lasso) Regularization for aggressive feature selection.
2. SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
3. Random Forest Classification optimized via Grid Search.
4. Repeated Stratified K-Fold Cross-Validation for robust metric estimation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer

# Use imblearn's pipeline to properly integrate SMOTE during cross-validation
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def load_and_preprocess(filepath):
    """Loads data and encodes the target variable."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Class distribution: {np.bincount(y_encoded)} (0: NonToxic, 1: Toxic)")
    
    return X, y_encoded, le

def build_pipeline():
    """Constructs the SMOTE + Feature Selection + Random Forest pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        # L1 Regularization to force noisy feature coefficients to zero
        ('feature_selection', SelectFromModel(LogisticRegression(
            penalty='l1', solver='liblinear', class_weight='balanced', random_state=42
        ))),
        # Synthesize minority class samples
        ('smote', SMOTE(random_state=42)),
        # Base estimator
        ('classifier', RandomForestClassifier(random_state=42))
    ])

def main():
    # 1. Load Data
    # Note: Adjust the path if running from a different directory
    X, y, label_encoder = load_and_preprocess('../data/data.csv')

    # 2. Setup Pipeline & Grid Search
    pipeline = build_pipeline()
    
    param_grid = {
        'feature_selection__estimator__C': [0.1, 0.5, 1.0], # Controls feature sparsity
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],            # Restrict depth to prevent overfitting
        'classifier__min_samples_leaf': [2, 4]
    }

    # 3x5 Repeated Stratified K-Fold for robust evaluation on small data
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scorer = make_scorer(f1_score, average='macro')

    grid = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring=scorer, 
        n_jobs=-1, 
        verbose=1
    )

    # 3. Execute Training
    print("\nExecuting Grid Search and Cross-Validation...")
    grid.fit(X, y)

    # 4. Extract and Print Results
    print("\n" + "="*40)
    print("OPTIMIZATION COMPLETE")
    print("="*40)
    print(f"Best CV Macro F1 Score: {grid.best_score_:.3f}")
    print("\nBest Hyperparameters:")
    for k, v in grid.best_params_.items():
        print(f" - {k}: {v}")

    # Extract surviving features
    best_fs = grid.best_estimator_.named_steps['feature_selection']
    surviving_mask = best_fs.get_support()
    surviving_features = X.columns[surviving_mask]
    
    print(f"\nFeature Selection: {len(surviving_features)} out of {X.shape[1]} features retained.")
    print("Top 5 Retained Features:", list(surviving_features[:5]))

    # Final Classification Report
    y_pred = grid.predict(X)
    print("\nFinal Model Classification Report (In-Sample):")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))

if __name__ == "__main__":
    main()