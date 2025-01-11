# Analisi dei parametri di stampa e previsione delle proprietà meccaniche dei compositi fibrorinforzati mediante Machine Learning: ottimizzazione della stampa FDM
L’obiettivo principale di questo studio è prevedere le proprietà dei componenti stampati già in fase di progettazione, ovvero prima della loro effettiva produzione, 
utilizzando metodi di Machine Learning. In particolare, sono stati utilizzati due metodi: XGBoost, scelto per le sue elevate prestazioni e Random Forest, 
utilizzato come termine di confronto. La repositary contiene i codici necessari per effettuare il preprocessing dei dati e implementare i due modelli.

## Contenuti
- [Lettura file csv](./read_parameters) - sono stati estratti i dati necessari dal dataset disponibile a seguito delle prove sperimentali
- [PCA](./variance) - calcolo varianza e matrice di correlazione dei dati in entrata
- [Pre-processing](./preprocess_parameters) - normalizzazione del dataset
- [Pre-processing](./kfold) - suddivisione dataset in validation e train+test e suddivisione tramite KFold-Cross Validation dei dati di training da quelli di testing
- [Suddivisione dati per allenamento finale](./dataset_split)
  
- [Inizializzazione sweep XGB](./init_sweep_xgboost)- inizializzazione sweep su wandb per ottimizzazione parametri
- [Modello XGB](./model_xgbost)
- [Parallelizzazione XGB](./parallelW_XGB) - parallelizzazione di più processi paralleli
- [Allenamento finale XGB](./best_run_XGB) - è stata trovata la configurazione di parametri ottimale e si è allenato su questa il modello

- [Inizializzazione sweep RF](./init_sweep_randomforest)- inizializzazione sweep su wandb per ottimizzazione parametri
- [Modello RF](./random_forest_new)
- [Parallelizzazione RF](./parallelW_RF) - parallelizzazione di più processi paralleli
- [Allenamento finale RF](./best_run_RF) - è stata trovata la configurazione di parametri ottimale e si è allenato su questa il modello
