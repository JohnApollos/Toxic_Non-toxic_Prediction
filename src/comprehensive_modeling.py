import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Custom PLS-DA classifier wrapper for scikit-learn
class PLSDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = None
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Convert target to dummy columns
        y_dummy = pd.get_dummies(y).values
        self.pls = PLSRegression(n_components=self.n_components)
        self.pls.fit(X, y_dummy)
        return self
        
    def predict(self, X):
        preds = self.pls.predict(X)
        return np.argmax(preds, axis=1)

def main():
    df = pd.read_csv('data/data.csv')
    X = df.drop(columns=['Class'])
    y = LabelEncoder().fit_transform(df['Class'])
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scorer = make_scorer(f1_score, average='macro')
    
    # 1. Scaling baseline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Let's define the models to test
    models = {}
    
    # --- Linear Models ---
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        models[f"Lasso LR (C={C})"] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=C, random_state=42))
        ])
        models[f"Ridge LR (C={C})"] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='l2', class_weight='balanced', C=C, random_state=42))
        ])
        models[f"ElasticNet LR (C={C}, l1_ratio=0.5)"] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', C=C, l1_ratio=0.5, random_state=42, max_iter=2000))
        ])
        
    # --- PLS-DA ---
    for n_comp in [2, 3, 5, 10, 15]:
        models[f"PLS-DA (comp={n_comp})"] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', PLSDA(n_components=n_comp))
        ])
        
    # --- PCA + Classification ---
    for n_comp in [2, 5, 10, 20]:
        models[f"PCA (comp={n_comp}) + Linear SVM"] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp)),
            ('clf', SVC(kernel='linear', class_weight='balanced', C=0.1, random_state=42))
        ])
        models[f"PCA (comp={n_comp}) + RBF SVM"] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp)),
            ('clf', SVC(kernel='rbf', class_weight='balanced', C=1.0, random_state=42))
        ])
        models[f"PCA (comp={n_comp}) + Logistic Regression"] = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp)),
            ('clf', LogisticRegression(class_weight='balanced', C=0.1, random_state=42))
        ])

    # --- SelectKBest + Classification ---
    for k in [5, 10, 20, 50]:
        models[f"SelectKBest (k={k}) + RBF SVM"] = Pipeline([
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest(score_func=f_classif, k=k)),
            ('clf', SVC(kernel='rbf', class_weight='balanced', C=1.0, random_state=42))
        ])
        models[f"SelectKBest (k={k}) + Logistic Regression"] = Pipeline([
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest(score_func=f_classif, k=k)),
            ('clf', LogisticRegression(class_weight='balanced', C=0.1, random_state=42))
        ])

    # --- SVM (RBF & Linear) directly on all features ---
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        models[f"SVM Linear (C={C})"] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', class_weight='balanced', C=C, random_state=42))
        ])
        models[f"SVM RBF (C={C})"] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', class_weight='balanced', C=C, random_state=42))
        ])

    # Let's run CV and find the best models
    results = []
    print("Evaluating models...")
    for name, pipeline in models.items():
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            results.append((name, scores.mean(), scores.std()))
        except Exception as e:
            print(f"Failed to run {name}: {e}")
            
    # Sort results
    results_df = pd.DataFrame(results, columns=['Model', 'Mean Macro F1', 'Std Macro F1'])
    results_df = results_df.sort_values(by='Mean Macro F1', ascending=False)
    
    print("\nTop 15 Models by Mean Macro F1:")
    print(results_df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
