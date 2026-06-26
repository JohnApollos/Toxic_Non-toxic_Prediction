import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def main():
    print("Fetching toxicity dataset (ID: 728)...")
    dataset = fetch_ucirepo(id=728)
    
    # Access data
    X = dataset.data.features
    y = dataset.data.targets
    
    print("Features shape:", X.shape)
    print("Targets shape:", y.shape)
    print("Targets head:\n", y.head())
    print("Target value counts:\n", y.value_counts())
    
    # Combine X and y into a single dataframe
    # We want to make sure the target column is named 'Class'
    # Let's inspect the target column name
    target_col = y.columns[0]
    print(f"Target column name: {target_col}")
    
    # Rename target column to 'Class' if it isn't already
    y_renamed = y.rename(columns={target_col: 'Class'})
    
    df = pd.concat([X, y_renamed], axis=1)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/data.csv', index=False)
    print("Successfully saved data to data/data.csv")

if __name__ == "__main__":
    main()
