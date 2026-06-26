import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

def main():
    df = pd.read_csv('../data/data.csv')
    X = df.drop(columns=['Class'])
    y = LabelEncoder().fit_transform(df['Class'])
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scorer = make_scorer(f1_score, average='macro')
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kpca', KernelPCA(n_components=2, kernel='rbf', random_state=42)),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    
    # Moderate grid for sequential search
    param_grid = {
        'kpca__gamma': [None, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        'clf__C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        'clf__penalty': ['l2', 'l1'],
        'clf__solver': ['liblinear']
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=scorer, n_jobs=1)
    grid.fit(X, y)
    
    print("="*40)
    print("KERNEL PCA OPTIMIZATION COMPLETE")
    print("="*40)
    print(f"Best CV Macro F1: {grid.best_score_:.6f}")
    print("\nBest Hyperparameters:")
    for k, v in grid.best_params_.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
