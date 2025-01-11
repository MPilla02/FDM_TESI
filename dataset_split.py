import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

def datasplit(input_csv, target_csv, feature_columns, target_column, val_size=0.2, random_state=42):
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
    
    return x_train_test, y_train_test, x_val, y_val