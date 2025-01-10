import wandb
from wandb.sdk.internal.internal_api import gql
from wandb import util
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from dataset_split import datasplit
import os
import pickle
import re

def _get_run_query(config_sweep):
    api = wandb.api
    entity = config_sweep['entity_name']
    project_name = config_sweep['project_name']
    sweepID = config_sweep['sweep_id']
        
    query = gql(
            """
        query SweepWithRuns($entity: String, $project: String, $sweep: String!, $cursor: String!) {
            project(name: $project, entityName: $entity) {
                sweep(sweepName: $sweep) {
                    runs(first: 1000, after: $cursor) {
                        edges {
                            node {
                                state
                                displayName
                                summaryMetrics
                                config
                            }
                        }
                        pageInfo {
                            hasNextPage
                            endCursor
                        }    
                    }
                }
            }
        }
        """
    )
    cursor = ""
    hasNextPage = True
    data_flat_tot = []
    while hasNextPage:
        check_retry_fn = util.make_check_retry_fn(check_fn=util.check_retry_conflict_or_gone,
                                                check_timedelta=timedelta(minutes=5),
                                                fallback_retry_fn=util.no_retry_auth)
        response = api.api.gql(
                    query,
                    variable_values={
                        "entity": entity,
                        "project": project_name,
                        "sweep": sweepID,
                        "cursor": cursor
                    },
                    check_retry_fn=check_retry_fn
                )
        
        data = response["project"]["sweep"]["runs"]
        data_flat = api.api._flatten_edges(data)
        data_flat_tot += data_flat
        hasNextPage = data['pageInfo']['hasNextPage']
        cursor = data['pageInfo']['endCursor']
    return data_flat_tot

api = wandb.Api()
sweep_id ='sznu5qxb'  
entity = 'pilla-sapienza-universit-di-roma'
project_name = 'random-forest-hyperparam-sweep-new'

config_sweep = {'entity_name': entity,
                'project_name': project_name,
                'sweep_id': sweep_id}

runs = _get_run_query(config_sweep)
scores = list()

for run in runs:
    if run['state'] == 'finished':  
        _name = run['displayName']
        if 'summaryMetrics' in run:
            summaryMetrics = json.loads(run['summaryMetrics'])
            if 'avg_rmse' in summaryMetrics:
                _metrics_rmse = float(summaryMetrics['avg_rmse'])
            else:
                _metrics_rmse = float('nan')  
                print(f"avg_rmse metric missing for run: {_name}")
                
            _config = json.loads(run['config']) if 'config' in run else None
            scores.append([_metrics_rmse, _name, _config])

best_run = sorted(scores, key=lambda x: x[0])[0]
best_run_config = best_run[2]
cleaned_config = {key: value for key, value in best_run_config.items() if key != "_wandb"}
print("Configurazione della run migliore:")
print(json.dumps(cleaned_config, indent=3))
# Stampa dell'RMSE medio della migliore run
best_rmse = best_run[0]
print(f"RMSE medio della best run: {best_rmse}")

# Allenamento locale con miglior configurazione
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


x_train_test, y_train_test, x_val, y_val = datasplit(input_csv, target_csv, feature_columns, target_column, val_size=0.2, random_state=42)


# Configurazione del modello Random Forest con i migliori iperparametri
best_params_rf = {
    "n_estimators": int(cleaned_config["n_estimators"]["value"]),
    "max_depth": int(cleaned_config["max_depth"]["value"]),
    "min_samples_split": int(cleaned_config["min_samples_split"]["value"]),
    "min_samples_leaf": int(cleaned_config["min_samples_leaf"]["value"]),
    "max_features": cleaned_config["max_features"]["value"],
    "bootstrap": cleaned_config["bootstrap"]["value"],
    "random_state": 42  
    }

# Creazione e allenamento del modello
model_rf = RandomForestRegressor(**best_params_rf)
model_rf.fit(x_train_test, y_train_test)

y_train_test_pred = model_rf.predict(x_train_test)
rmse_train_test = np.sqrt(mean_squared_error(y_train_test, y_train_test_pred))
print(f"RMSE sul dataset di train+test (Random Forest): {rmse_train_test}")

y_val_pred = model_rf.predict(x_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"RMSE sul dataset di validazione (Random Forest): {rmse_val}")

# GRAFICI
# Riscala dati normalizzati nel dominio reale
with open('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/scaler_properties.pkl', 'rb') as f:
    scaler = pickle.load(f)
y_train_test_rescaled = scaler.inverse_transform(y_train_test)
y_train_test_pred_rescaled =  scaler.inverse_transform(y_train_test_pred)
y_val_rescaled =  scaler.inverse_transform(y_val)
y_val_pred_rescaled =  scaler.inverse_transform(y_val_pred)

# Genera grafici per ogni variabile target
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

for i, column in enumerate(target_column):
    title_train_test = f"{column}: Predetto vs Reale (Train+Test) con RF"
    title_validation = f"{column}: Predetto vs Reale (Validation) con RF"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  

    for ax in axes:
        ax.set_aspect('equal')  
    
    # Calcola i limiti minimi e massimi per la variabile specifica
    x_min = min(y_train_test_rescaled[:, i].min(), y_val_rescaled[:, i].min())
    x_max = max(y_train_test_rescaled[:, i].max(), y_val_rescaled[:, i].max())
    y_min = min(y_train_test_pred_rescaled[:, i].min(), y_val_pred_rescaled[:, i].min())
    y_max = max(y_train_test_pred_rescaled[:, i].max(), y_val_pred_rescaled[:, i].max())
    
    # Calcola il minimo e massimo globale con un margine aggiuntivo
    margin = 0.05 * (max(x_max, y_max) - min(x_min, y_min))  # 5% del range come margine
    local_min = min(x_min, y_min) - margin
    local_max = max(x_max, y_max) + margin

    # Subplot per Train+Test
    axes[0].scatter(y_train_test_rescaled[:, i], y_train_test_pred_rescaled[:, i], color='blue', alpha=0.5, label="Predizioni")
    axes[0].plot([local_min, local_max], [local_min, local_max],
                 color='red', linestyle='--', label="Bisettrice (y=x)")
    axes[0].set_title(title_train_test)  
    axes[0].set_xlabel("Reale")
    axes[0].set_ylabel("Predetto")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(local_min, local_max)
    axes[0].set_ylim(local_min, local_max)

    # Subplot per Validation
    axes[1].scatter(y_val_rescaled[:, i], y_val_pred_rescaled[:, i], color='green', alpha=0.5, label="Predizioni")
    axes[1].plot([local_min, local_max], [local_min, local_max],
                 color='red', linestyle='--', label="Bisettrice (y=x)")
    axes[1].set_title(title_validation)  
    axes[1].set_xlabel("Reale")
    axes[1].set_ylabel("Predetto")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlim(local_min, local_max)
    axes[1].set_ylim(local_min, local_max)

    plt.tight_layout()

    sanitized_column = sanitize_filename(column)
    
    # Salvataggio immagini 
    folder_path = 'C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/Grafici_Predizioni_RANDOMFOREST'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f"{folder_path}/{sanitized_column}.png", dpi=300)  
    plt.close
