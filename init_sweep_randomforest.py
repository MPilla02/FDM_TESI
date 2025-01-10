import wandb
import os
from os import path
import pathlib
import json

project = "random-forest-hyperparam-sweep-new"  
entity = 'pilla-sapienza-universit-di-roma'
name = 'prova'

sweep_config = {
    'method': 'random',  
    'metric': {
        'name': 'rmse',  
        'goal': 'minimize'  
    },
    'parameters': {
        'n_estimators':{ 'values': [50, 100, 150, 200, 300, 500, 700]}, 
        'max_depth': {'values': [5, 10, 20, 30]}, 
        'min_samples_split': {'values': [2, 5, 10, 20]}, 
        'min_samples_leaf': {'values': [1, 2, 5, 10, 20]}, 
        'max_features': {'values': ['sqrt', 'log2', 0.5]}, 
        'bootstrap': {'values': [True, False]}  
    }
}

sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)

sweep_config_data = {'entity_name': entity,
                     'project_name': project,
                     'sweep_ID': sweep_id }

path_wd=pathlib.Path(os.getcwd())
path_sweep_config= path_wd.joinpath('sweep_config')
os.makedirs(path_sweep_config, exist_ok=True)

path_sweep_config_file = path_sweep_config.joinpath(f"{sweep_id}.json")
with open(path_sweep_config_file, 'w') as f:
    f.write(json.dumps(sweep_config, indent=4))
print(f"File di configurazione creato: {path_sweep_config}")
