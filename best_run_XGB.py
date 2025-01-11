import wandb
from wandb.sdk.internal.internal_api import gql
from wandb import util
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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
sweep_id ='d1z95aka'  
entity = 'pilla-sapienza-universit-di-roma'
project_name = 'xgboost-hyperparam-sweep'

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

# Configurazione del modello con i migliori iperparametri
best_params = {
    "learning_rate": cleaned_config["learning_rate"]["value"],
    "max_depth": int(cleaned_config["max_depth"]["value"]),
    "n_estimators": int(cleaned_config["n_estimators"]["value"]),
    "subsample": cleaned_config["subsample"]["value"],
    "colsample_bytree": cleaned_config["colsample_bytree"]["value"],
    "gamma": cleaned_config["gamma"]["value"],
    "alpha": cleaned_config["alpha"]["value"],
    "objective": "reg:squarederror",
    "random_state": 42  
}

model_final = xgb.XGBRegressor(**best_params)
model_final.fit(x_train_test, y_train_test)
y_train_test_pred = model_final.predict(x_train_test)
rmse_val = np.sqrt(mean_squared_error(y_train_test, y_train_test_pred))
print(f"RMSE sul dataset di train+test: {rmse_val}")

y_val_pred = model_final.predict(x_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"RMSE sul dataset di validazione: {rmse_val}")

# Dizionario delle unità di misura per ogni variabile target
units = {
    "Young (N/%)": "N/%",
    "Force UTS (N)": "N",
    "Strain UTS (%)": "%",
    "Force Yield (N)": "N",
    "Strain Yield (%)": "%",
    "Strain Break (%)": "%",
    "Work(J)": "J",
    "b (mm)": "mm",
    "h (mm)": "mm",
    "Beauty": "" 
}

# Rescaling
with open('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/scaler_properties.pkl', 'rb') as f:
    scaler = pickle.load(f)
y_train_test_rescaled = scaler.inverse_transform(y_train_test)
y_train_test_pred_rescaled =  scaler.inverse_transform(y_train_test_pred)
y_val_rescaled =  scaler.inverse_transform(y_val)
y_val_pred_rescaled =  scaler.inverse_transform(y_val_pred)

# Errore assoluto medio
absolute_errors_train_test = np.abs(y_train_test_rescaled - y_train_test_pred_rescaled)
absolute_errors_val = np.abs(y_val_rescaled - y_val_pred_rescaled)

mean_absolute_errors_train_test = np.mean(absolute_errors_train_test, axis=0)
mean_absolute_errors_val = np.mean(absolute_errors_val, axis=0)

print("Errore assoluto medio per variabile (Train+Test):")
for column, error in zip(target_column, mean_absolute_errors_train_test):
    print(f"{column}: {error:.2f}")

print("\nErrore assoluto medio per variabile (Validation):")
for column, error in zip(target_column, mean_absolute_errors_val):
    print(f"{column}: {error:.2f}")

# Salvataggio errori assoluti
error_output_path = 'C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/Errori_Assoluti_XGBoost.txt'
with open(error_output_path, 'w') as f:
    f.write("Errore assoluto medio per variabile (Train+Test):\n")
    for column, error in zip(target_column, mean_absolute_errors_train_test):
        f.write(f"{column}: {error:.2f}\n")
    f.write("\nErrore assoluto medio per variabile (Validation):\n")
    for column, error in zip(target_column, mean_absolute_errors_val):
        f.write(f"{column}: {error:.2f}\n")

# GRAFICI
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

for i, column in enumerate(target_column):
    unit = units.get(column, "")  
    error_train_test = mean_absolute_errors_train_test[i]
    error_val = mean_absolute_errors_val[i]

    title_train_test = f"{column}: Predetto vs Reale (Train+Test) con XGB"
    title_validation = f"{column}: Predetto vs Reale (Validation) con XGB"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
    for ax in axes:
        ax.set_aspect('equal')  
    
    x_min = min(y_train_test_rescaled[:, i].min(), y_val_rescaled[:, i].min())
    x_max = max(y_train_test_rescaled[:, i].max(), y_val_rescaled[:, i].max())
    y_min = min(y_train_test_pred_rescaled[:, i].min(), y_val_pred_rescaled[:, i].min())
    y_max = max(y_train_test_pred_rescaled[:, i].max(), y_val_pred_rescaled[:, i].max())
    
    margin = 0.05 * (max(x_max, y_max) - min(x_min, y_min)) 
    local_min = min(x_min, y_min) - margin
    local_max = max(x_max, y_max) + margin

    # Train+Test Plot
    axes[0].scatter(y_train_test_rescaled[:, i], y_train_test_pred_rescaled[:, i], color='blue', alpha=0.5, label="Predizioni")
    axes[0].plot([local_min, local_max], [local_min, local_max], color='red', linestyle='--', label="Bisettrice (y=x)")
    axes[0].set_title(title_train_test)
    axes[0].set_xlabel("Reale")
    axes[0].set_ylabel("Predetto")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(local_min, local_max)
    axes[0].set_ylim(local_min, local_max)
    axes[0].text(0.05, 0.95, f"Errore Assoluto Medio:\n{error_train_test:.2f} {unit}", transform=axes[0].transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    # Validation Plot
    axes[1].scatter(y_val_rescaled[:, i], y_val_pred_rescaled[:, i], color='green', alpha=0.5, label="Predizioni")
    axes[1].plot([local_min, local_max], [local_min, local_max], color='red', linestyle='--', label="Bisettrice (y=x)")
    axes[1].set_title(title_validation)
    axes[1].set_xlabel("Reale")
    axes[1].set_ylabel("Predetto")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlim(local_min, local_max)
    axes[1].set_ylim(local_min, local_max)
    axes[1].text(0.05, 0.95, f"Errore Assoluto Medio:\n{error_val:.2f} {unit}", transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    plt.tight_layout()

    sanitized_column = sanitize_filename(column)
    folder_path = 'C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/Grafici_Predizioni_XGBOOST'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f"{folder_path}/{sanitized_column}.png", dpi=300)  
    plt.close()
