import pandas as pd
import os

cartella = 'C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI'
if not os.path.exists(cartella):
    os.makedirs(cartella)  

parameters_file_path = "C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/data/parameters_data.csv" #percorso del file parametri CSV
properties_file_path = 'C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/data/properties_data.csv'      #percorso del file proprietà CSV

data = pd.read_csv(parameters_file_path) #lettura file parametri csv

# Filtro dati PLA, PA12, PA66, PA12/PA66
PLA_PA12_PA66_data = data[(data['PLA'] == 1.0) | (data['PA12'] == 1.0) | (data['PA66'] == 1.0)]
PA66_PA12_combined_data = data[(data['PA12'] == 0.5) & (data['PA66'] == 0.5)]
total_data = pd.concat([PLA_PA12_PA66_data, PA66_PA12_combined_data]).drop_duplicates()

# Salvo indici dati PLA, PA12, PA66, PA12/PA66
combined_indices = total_data.index
combined_indices_df = pd.DataFrame(combined_indices)
combined_indices_df.to_csv('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_indices.csv', index=False, header=False)
print("Indici PLA, PA12, PA66, PA12/PA66 salvati nel file total_indices.csv")

# Salvo dati estratti PLA, PA12, PA66, PA12/PA66
total_extracted_data = data.loc[combined_indices]
total_extracted_data.to_csv('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_parameters.csv', index=False)
print("Parametri del PLA, PA12, PA66, PA12/PA66 estratti e salvati nel file total_parameters.csv")    

# Aggiungo proprietà estratte
properties_data = pd.read_csv(properties_file_path)       #lettura file proprietà csv
new_properties_data = properties_data.iloc[:, :-2]        #rimuovo le ultime due colonne (non mi interessano)
total_properties_data = new_properties_data.iloc[combined_indices]
total_properties_data.to_csv('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_properties.csv', index=False)
print("Proprietà del PLA, PA12, PA66, PA12/PA66 estratte e salvate nel file extracted_properties.csv")



