import pandas as pd
import argparse
import json
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import wandb
import time
import pathlib
from kfold import keyfold

input_csv = 'C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_parameters_normalized.csv'
target_csv = 'C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_properties_normalized.csv'

feature_columns = ['Layer Height (mm)', 'Extruson Width (mm)', 'Speed Default (mm/s)', 'Speed Solid Fill (mm/s)', 
                   'Acceleration (log10(mm/s2))', 'Jerk (mm/s3)', 'Shells', 'Infill Density (%)', 'Density (%)', 
                   'Base Layers', 'Delta Ext-Melt (°C)', 'Delta Bed-Crist (°C)', 'Delta Bed-Glass (°C)', 
                   'Delta Bed-Chamb (°C)', 'Temperature Filament (°C)', 'X Pos', 'Y Pos', 'Strain 1', 'Strain 2', 
                   'ABS', 'PA12', 'PA66', 'PC', 'PEKK', 'PET', 'PLA', 'TPU', 'PP', 'ASA', 'PURE', 'CF', 'FLAX', 
                   'WOOD', 'Amorphus', 'Crystalline', 'Semi_Crystalline', 'Rectilinear', 'Honeycomb', 'Gyroid', 
                   'Cubic', '9T LABS', 'BambuLab X1E']

target_column = ['Young (N/%)', 'Force UTS (N)', 'Strain UTS (%)', 'Force Yield (N)', 'Strain Yield (%)', 
                 'Strain Break (%)', 'Work(J)', 'b (mm)', 'h (mm)', 'Beauty']

# Parser degli argomenti
parser = argparse.ArgumentParser()
parser.add_argument('--sweep_name', dest='sweep_name', required=True, help='Sweep ID')
parser.add_argument('--project', dest='project', required=True, help='W&B project name')
args = parser.parse_args()
sweep_name = args.sweep_name

# Funzione di training per lo sweep
def train_model():
    wandb.init(
        project="random-forest-hyperparam-sweep-new",   
        entity="pilla-sapienza-universit-di-roma",
        config={}
        )     
    

    print(f"Sweep Name utilizzato: {args.sweep_name}")
    print("\nAltri parametri dello sweep:")

    for key, value in wandb.config.items():
        print(f"{key}: {value}")

    if wandb.run is None:
        print("Errore: wandb.run non è stato inizializzato correttamente.")
        return

    
    config = wandb.config     
    model = RandomForestRegressor (
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        bootstrap=config.bootstrap,
        random_state=42
        )
    
    multioutput_model = MultiOutputRegressor(model)
    
    fold_rmse_values = []

    # Estrazione dati con generatore keyfold
    for fold, fold_data in enumerate(keyfold(input_csv, target_csv, feature_columns, target_column, n_splits=5, random_state=42)):
        print(f"Training fold {fold + 1}...")

        x_train_fold = fold_data["x_train_fold"]
        y_train_fold = fold_data["y_train_fold"]
        x_test_fold = fold_data["x_test_fold"]
        y_test_fold = fold_data["y_test_fold"]
        x_val_fold = fold_data["x_val"]  
        y_val_fold = fold_data["y_val"]  

        multioutput_model.fit(x_train_fold, y_train_fold)
        
        # Testing
        y_test_pred = multioutput_model.predict(x_test_fold)
        test_rmse = np.sqrt(mean_squared_error(y_test_fold, y_test_pred))
        fold_rmse_values.append(test_rmse)
        wandb.log({f'fold_{fold+1}_test_rmse': test_rmse})
    
        # Validation
        y_val_pred = multioutput_model.predict(x_val_fold)
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        wandb.log({f'fold_{fold+1}_val_rmse': val_rmse})
        print(f"Fold {fold + 1} - Test RMSE: {test_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")

    # Media dell'RMSE dei vari fold
    avg_rmse = np.mean(fold_rmse_values)
    print(f"Average RMSE over all folds: {avg_rmse:.4f}")
    wandb.log({'avg_rmse': avg_rmse})
    
    wandb.finish()

    
    
if __name__ == "__main__":
    path_wd = pathlib.Path(os.getcwd())
    path_sweeps_config = path_wd.joinpath('sweep_config')
    path_sweep_config = path_sweeps_config.joinpath(sweep_name).with_suffix('.json')  
    with open(path_sweep_config,'r') as f:
        sweep_config = json.load(f)
        
    sweep_id = args.sweep_name  
    project_name = args.project  

    wandb.agent(sweep_id, 
                function=train_model, 
                project="random-forest-hyperparam-sweep-new",  
                entity="pilla-sapienza-universit-di-roma")
    