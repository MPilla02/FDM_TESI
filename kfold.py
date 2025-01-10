import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

def keyfold(input_csv, target_csv, feature_columns, target_column, test_size=0.2, val_size=0.2, n_splits=5, random_state=42):
    input_df = pd.read_csv(input_csv)
    target_df = pd.read_csv(target_csv)
    
    input_df['ID_main'] = input_df['ID'].str.split('_').str[0]
    target_df['ID_main'] = target_df['ID'].str.split('_').str[0]
    
    merged_df = pd.merge(input_df, target_df, on='ID_main')
    unique_ids_main = merged_df['ID_main'].unique()

    # Divisione dati test/train-val
    train_test_ids_main, val_ids_main = train_test_split(
        unique_ids_main, test_size=val_size, random_state=random_state)
    
    train_test_df = merged_df[merged_df['ID_main'].isin(train_test_ids_main)]
    val_df = merged_df[merged_df['ID_main'].isin(val_ids_main)]

    
    x_train_test = train_test_df[feature_columns]
    y_train_test = train_test_df[target_column]

    x_val = val_df[feature_columns]
    y_val = val_df[target_column]

    # KFold su dati test/train
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_indices, test_indices in kf.split(x_train_test):
        x_train_fold = x_train_test.iloc[train_indices]
        y_train_fold = y_train_test.iloc[train_indices]
        x_test_fold = x_train_test.iloc[test_indices]
        y_test_fold = y_train_test.iloc[test_indices]
        
        yield {
            "x_train_fold": x_train_fold.to_numpy(),
            "y_train_fold": y_train_fold.to_numpy(),
            "x_test_fold": x_test_fold.to_numpy(),
            "y_test_fold": y_test_fold.to_numpy(),
            "x_val": x_val.to_numpy(),  
            "y_val": y_val.to_numpy(),  
             }