import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

def main():
    df = pd.read_csv('data/data.csv')
    X = df.drop(columns=['Class'])
    y = LabelEncoder().fit_transform(df['Class'])
    
    # 5-fold, 3-repeat CV (15 fits per configuration)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scorer = make_scorer(f1_score, average='macro')
    
    # Grid 1: PCA + Logistic Regression
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=42)),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    
    param_grid_lr = {
        'pca__n_components': [2, 3, 5],
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__penalty': ['l2', 'l1'],
        'clf__solver': ['liblinear']
    }
    
    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=cv, scoring=scorer, n_jobs=-1)
    print("Running optimized Grid Search for PCA + Logistic Regression...")
    grid_lr.fit(X, y)
    print(f"Best LR CV Macro F1: {grid_lr.best_score_:.4f}")
    print("Best LR params:", grid_lr.best_params_)
    
    # Grid 2: PCA + SVM
    pipe_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=42)),
        ('clf', SVC(class_weight='balanced', random_state=42))
    ])
    
    param_grid_svm = {
        'pca__n_components': [2, 3, 5],
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__kernel': ['linear', 'rbf']
    }
    
    grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=cv, scoring=scorer, n_jobs=-1)
    print("\nRunning optimized Grid Search for PCA + SVM...")
    grid_svm.fit(X, y)
    print(f"Best SVM CV Macro F1: {grid_svm.best_score_:.4f}")
    print("Best SVM params:", grid_svm.best_params_)
    
    # Overall Best Model
    best_overall_score = max(grid_lr.best_score_, grid_svm.best_score_)
    if grid_lr.best_score_ > grid_svm.best_score_:
        best_model = grid_lr.best_estimator_
        best_name = "PCA + Logistic Regression"
        best_params = grid_lr.best_params_
    else:
        best_model = grid_svm.best_estimator_
        best_name = "PCA + SVM"
        best_params = grid_svm.best_params_
        
    print("\n" + "="*40)
    print(f"OVERALL BEST MODEL: {best_name}")
    print(f"Best CV Macro F1: {best_overall_score:.4f}")
    print("="*40)
    print("Parameters:")
    for k, v in best_params.items():
        print(f" - {k}: {v}")
        
if __name__ == "__main__":
    main()
