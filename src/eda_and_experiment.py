import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load data
    df = pd.read_csv('data/data.csv')
    X = df.drop(columns=['Class'])
    y = LabelEncoder().fit_transform(df['Class'])
    
    # Check for missing values
    missing = X.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    
    # Check if there are constant columns
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    print(f"Number of constant columns: {len(constant_cols)}")
    
    # CV setup
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scorer = make_scorer(f1_score, average='macro')
    
    # Let's test different combinations of models and feature selection/oversampling
    experiments = {
        # 1. Baseline: StandardScaler -> Lasso Selection -> SMOTE -> RandomForest
        "Baseline (RF with Lasso FS & SMOTE)": ImbPipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1))),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(max_depth=3, min_samples_leaf=2, n_estimators=200, random_state=42))
        ]),
        
        # 2. Baseline without SMOTE
        "RF with Lasso FS (No SMOTE)": Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1))),
            ('classifier', RandomForestClassifier(max_depth=3, min_samples_leaf=2, n_estimators=200, random_state=42, class_weight='balanced'))
        ]),
        
        # 3. Simple Logistic Regression with L1 (Lasso) penalty (no intermediate feature selection needed since L1 does it)
        "Lasso Logistic Regression (balanced)": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42))
        ]),
        
        # 4. Logistic Regression with L2 (Ridge) penalty
        "Ridge Logistic Regression (balanced)": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(penalty='l2', class_weight='balanced', C=0.01, random_state=42))
        ]),
        
        # 5. Linear Support Vector Classifier (SVC)
        "Linear SVM (balanced)": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear', class_weight='balanced', C=0.01, random_state=42))
        ]),
        
        # 6. RBF Support Vector Classifier (SVC)
        "RBF SVM (balanced)": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', class_weight='balanced', C=1.0, random_state=42))
        ]),
        
        # 7. RandomForest directly on all features (using class_weight='balanced' or balanced_subsample)
        "RF on all features (balanced)": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(max_depth=3, class_weight='balanced', random_state=42))
        ]),
        
        # 8. ExtraTreesClassifier (tends to overfit less on high-dimensional data)
        "ExtraTrees on all features": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', ExtraTreesClassifier(max_depth=3, class_weight='balanced', random_state=42))
        ]),
    }
    
    print("\nRunning experiments (5-fold, 3-repeat CV)...")
    for name, pipeline in experiments.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        print(f" - {name:40s}: Mean Macro F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

if __name__ == "__main__":
    main()
