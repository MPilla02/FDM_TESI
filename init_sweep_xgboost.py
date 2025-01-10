import wandb
import os
import pathlib
import json
os.environ["WANDB_SYMLINK"] = "false"

project="xgboost-hyperparam-sweep"
entity='pilla-sapienza-universit-di-roma'
name='prova'

# Parametri per sweep
sweep_config = {
    'method': 'random', 
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'
    },
    'parameters': {
        'max_depth': {'values': [ 4, 6, 8, 10 ]},
        'learning_rate': {'min':0.01 , 'max': 0.2 },
        'n_estimators': {'values': [50, 100, 150, 200, 300, 500, 700]},
        'subsample': {'min': 0.3, 'max': 1.0},  
        'colsample_bytree': {'min': 0.3, 'max': 0.8},  
        'gamma': {'values': [0, 0.1, 0.3, 0.5, 1.0]},
        'alpha': {'min': 0.0, 'max': 5.0}  
        
    }
    }
    
# Inizializzazione sweep
sweep_id = wandb.sweep(sweep_config, project="xgboost-hyperparam-sweep")

sweep_config = {'entity_name': entity,
                'project_name': project,
                'sweep_ID': sweep_id }

path_wd=pathlib.Path(os.getcwd())
path_sweep_config= path_wd.joinpath('sweep_config')
os.makedirs(path_sweep_config, exist_ok=True)

path_sweep_config_file = path_sweep_config.joinpath(f"{sweep_id}.json")
with open(path_sweep_config_file, 'w') as f:
    f.write(json.dumps(sweep_config, indent=4))
print(f"File di configurazione creato: {path_sweep_config}")

