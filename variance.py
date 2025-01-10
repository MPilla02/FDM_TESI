import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento del dataset
file_path = "C:/Users/maria/OneDrive/Desktop/Python/FDM_TESI/DATI_ESTRATTI/total_parameters_normalized.csv"
data = pd.read_csv(file_path)

# Rimozione di colonne specifiche
columns_to_remove = ['ID', 'ABS', 'PA12', 'PA66', 'PC', 'PEKK', 'PET', 'PLA', 'TPU', 'PP', 'ASA', 'PURE', 'CF', 'FLAX', 'WOOD', 
                     'Amorphus', 'Crystalline', 'Semi_Crystalline', '9T LABS', 'BambuLab X1E', 'Strain 1', 'Strain 2','Density (%)']  
data_new = data.drop(columns=columns_to_remove)

# Calcolo percentuale varianza spiegata
cov_matrix = np.cov(data_new.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("Autovalori (varianza associata):")
print(eigenvalues)
print("\nPercentuale di varianza spiegata:")
print(explained_variance_ratio)

# Grafico della varianza spiegata cumulativa
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Varianza Cumulativa (Autovalori)')
plt.xlabel('Numero componenti principali')
plt.ylabel('Rapporto di varianza cumulativa spiegata')
plt.grid()
plt.show()

# Calcolo della matrice di correlazione dai tuoi dati originali
correlation_matrix = data_new.corr()
plt.figure(figsize=(16, 14))  
ax = sns.heatmap(
    correlation_matrix, 
    cmap='coolwarm', 
    annot=False, 
    xticklabels=correlation_matrix.columns, 
    yticklabels=correlation_matrix.columns
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
plt.tight_layout(pad=4.0, h_pad=2.0, w_pad=2.0)  
plt.savefig("correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# Identificazione delle coppie con forte correlazione 
threshold = 0.8  
strong_correlations = correlation_matrix[
    (correlation_matrix > threshold) | (correlation_matrix < -threshold)
]

np.fill_diagonal(strong_correlations.values, np.nan)

strong_corr_pairs = strong_correlations.stack().reset_index()
strong_corr_pairs.columns = ["Parametro 1", "Parametro 2", "Valore Correlazione"]
strong_corr_pairs = strong_corr_pairs.sort_values(by="Valore Correlazione", ascending=False)
print("\nCoppie di parametri con forte correlazione:")
print(strong_corr_pairs)
