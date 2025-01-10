import subprocess
import argparse
import wandb

wandb.login(key='ef2d782ee105ad7fad8b3c2fef30944e7f0aeef4')

parser = argparse.ArgumentParser()
parser.add_argument('--sweep_name', dest='sweep_name', required=True, help='Sweep ID')
parser.add_argument('--num_workers', dest='num_workers', required=True, help='Number of workers')
parser.add_argument('--project', dest='project', required=True, help='Name of the project')
args = parser.parse_args()

if __name__ == '__main__':
    sweep_name = args.sweep_name
    num_workers = int(args.num_workers)
    project_name = args.project  

    processes = []  

    while True:
        
        cmd = f"python model_xgboost.py --sweep_name {sweep_name} --project {project_name}"

        # Avvia nuovi processi fino a raggiungere il numero di workers desiderato
        while len(processes) < num_workers:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            processes.append(process)

        # Monitoriamo i processi già avviati
        for process in processes[:]:
            
            stdout, stderr = process.communicate()
            print(f"Output del worker (PID {process.pid}):")
            if process.returncode == 0:
                print("Successo:")
                print(stdout)  # Mostra output del worker
            else:
                print("Errore:")
                print(stderr)  # Mostra errore del worker

            # Rimuoviamo il processo dalla lista dei processi
            processes.remove(process)

     
