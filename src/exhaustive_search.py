import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load data
    df = pd.read_csv('data/data.csv')
    X = df.drop(columns=['Class'])
    y = LabelEncoder().fit_transform(df['Class'])
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scorer = make_scorer(f1_score, average='macro')
    
    # Define exhaustive strategies
    strategies = {}
    
    # 1. Base case / Baseline
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    strategies["Baseline (L1 FS + SMOTE + RF)"] = ImbPipeline([
        ('scaler', StandardScaler()),
        ('fs', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42, C=0.1))),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(max_depth=3, min_samples_leaf=2, n_estimators=200, random_state=42))
    ])
    
    # 2. Linear PCA + Logistic Regression (our previous champion)
    strategies["PCA (comp=2) + L1 LR (C=1.0)"] = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2, random_state=42)),
        ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=1.0, random_state=42))
    ])
    strategies["PCA (comp=3) + L1 LR (C=1.0)"] = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=3, random_state=42)),
        ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=1.0, random_state=42))
    ])
    
    # 3. Kernel PCA + Logistic Regression (checking non-linear PCA projections)
    for kernel in ['rbf', 'poly']:
        strategies[f"KernelPCA (comp=2, {kernel}) + LR"] = Pipeline([
            ('scaler', StandardScaler()),
            ('kpca', KernelPCA(n_components=2, kernel=kernel, random_state=42)),
            ('clf', LogisticRegression(class_weight='balanced', C=1.0, random_state=42))
        ])
        strategies[f"KernelPCA (comp=3, {kernel}) + LR"] = Pipeline([
            ('scaler', StandardScaler()),
            ('kpca', KernelPCA(n_components=3, kernel=kernel, random_state=42)),
            ('clf', LogisticRegression(class_weight='balanced', C=1.0, random_state=42))
        ])

    # 4. PCA + SVM (Linear vs RBF)
    strategies["PCA (comp=2) + Linear SVM"] = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2, random_state=42)),
        ('clf', SVC(kernel='linear', class_weight='balanced', C=0.1, random_state=42))
    ])
    strategies["PCA (comp=2) + RBF SVM"] = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2, random_state=42)),
        ('clf', SVC(kernel='rbf', class_weight='balanced', C=1.0, random_state=42))
    ])
    
    # 5. LDA (Linear Discriminant Analysis) with shrinkage directly on PCA vs raw
    strategies["PCA (comp=2) + LDA (shrinkage)"] = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2, random_state=42)),
        ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])
    strategies["LDA with shrinkage on raw features"] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])

    # 6. Gaussian Process Classification (RBF Kernel) on raw vs. PCA
    strategies["PCA (comp=2) + Gaussian Process"] = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2, random_state=42)),
        ('clf', GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=42))
    ])
    strategies["SelectKBest (k=2) + Gaussian Process"] = Pipeline([
        ('scaler', StandardScaler()),
        ('kbest', SelectKBest(score_func=f_classif, k=2)),
        ('clf', GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=42))
    ])

    # 7. SelectKBest Feature Selection + Classifiers
    for k in [2, 5, 10]:
        strategies[f"SelectKBest (k={k}) + RBF SVM"] = Pipeline([
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest(score_func=f_classif, k=k)),
            ('clf', SVC(kernel='rbf', class_weight='balanced', C=1.0, random_state=42))
        ])
        strategies[f"SelectKBest (k={k}) + L1 LR"] = Pipeline([
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest(score_func=f_classif, k=k)),
            ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=1.0, random_state=42))
        ])

    # 8. Stacking Ensembles (combining the best linear and non-linear pipelines)
    # Base estimators
    estimators = [
        ('pca_lr', Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2, random_state=42)),
            ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=1.0, random_state=42))
        ])),
        ('pca_svm', Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2, random_state=42)),
            ('clf', SVC(kernel='linear', class_weight='balanced', C=0.1, probability=True, random_state=42))
        ])),
        ('pca_gp', Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2, random_state=42)),
            ('clf', GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=42))
        ]))
    ]
    
    strategies["Stacking Ensemble (PCA-LR + PCA-SVM + PCA-GP)"] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
        cv=5,
        n_jobs=-1
    )

    # Run evaluations
    print("Executing exhaustive search...")
    results = []
    for name, model in strategies.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            results.append((name, scores.mean(), scores.std()))
            print(f" - {name:50s} : Mean Macro F1 = {scores.mean():.4f} +/- {scores.std():.4f}")
        except Exception as e:
            print(f"Failed {name}: {e}")
            
    results_df = pd.DataFrame(results, columns=['Strategy', 'Mean Macro F1', 'Std Macro F1'])
    results_df = results_df.sort_values(by='Mean Macro F1', ascending=False)
    
    print("\n" + "="*50)
    print("FINAL EXHAUSTIVE BENCHMARK RESULTS")
    print("="*50)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
