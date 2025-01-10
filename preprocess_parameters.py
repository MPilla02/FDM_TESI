import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle 

# Caricamento dati estratti 
total_parameters = pd.read_csv('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_parameters.csv')
total_properties= pd.read_csv('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_properties.csv')

# Normalizzazione parametri

columns_to_exclude = total_parameters.columns[[0]]    # Escludo le colonne 0 e 21-43 
columns_to_exclude = columns_to_exclude.append(total_parameters.columns[20:43])  
columns_to_normalize = total_parameters.drop(columns=columns_to_exclude)  
excluded_columns = total_parameters[columns_to_exclude]

scaler = StandardScaler()

normalized_data = scaler.fit_transform(columns_to_normalize)

normalized_data_df = pd.DataFrame(normalized_data, columns=columns_to_normalize.columns)

final_data = pd.concat([excluded_columns, normalized_data_df], axis=1)

final_data = final_data[total_parameters.columns]

# Salva lo scaler per i parametri
with open('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/scaler_parameters.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Salva dati normalizzati 
final_data.to_csv('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_parameters_normalized.csv', index=False)
print("Parametri normalizzati e salvati nel file total_data_normalized.csv")


# Normalizzazione proprietà
columns_to_exclude_properties = total_properties.columns[[0]]  
columns_to_normalize_properties = total_properties.drop(columns=columns_to_exclude_properties)  
excluded_columns_properties = total_properties[columns_to_exclude_properties]

normalized_properties_data = scaler.fit_transform(columns_to_normalize_properties)

normalized_properties_data_df = pd.DataFrame(normalized_properties_data, columns=columns_to_normalize_properties.columns)

final_properties_data = pd.concat([excluded_columns_properties, normalized_properties_data_df], axis=1)
final_properties_data = final_properties_data[total_properties.columns]

# Salva lo scaler per le proprietà
with open('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/scaler_properties.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
# Salva i dati normalizzati delle proprietà
final_properties_data.to_csv('C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_properties_normalized.csv', index=False)
print("Proprietà normalizzate e salvate nel file total_properties_normalized.csv")

