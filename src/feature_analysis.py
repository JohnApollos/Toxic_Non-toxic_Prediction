import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif

def main():
    df = pd.read_csv('data/data.csv')
    X = df.drop(columns=['Class'])
    y = LabelEncoder().fit_transform(df['Class'])
    
    # 1. Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 2. ANOVA F-value
    f_val, p_val = f_classif(X_scaled, y)
    anova_df = pd.DataFrame({'Feature': X.columns, 'F-value': f_val, 'p-value': p_val})
    anova_df = anova_df.sort_values(by='F-value', ascending=False)
    
    print("Top 10 features by ANOVA F-value:")
    print(anova_df.head(10))
    print("\nNumber of features with p-value < 0.05:", (anova_df['p-value'] < 0.05).sum())
    print("Number of features with p-value < 0.01:", (anova_df['p-value'] < 0.01).sum())
    
    # 3. Mutual Information
    mi = mutual_info_classif(X_scaled, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI': mi})
    mi_df = mi_df.sort_values(by='MI', ascending=False)
    print("\nTop 10 features by Mutual Information:")
    print(mi_df.head(10))
    print("Number of features with MI > 0:", (mi_df['MI'] > 0).sum())

if __name__ == "__main__":
    main()
