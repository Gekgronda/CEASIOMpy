import pandas as pd
import numpy as np
from prova_modello_surrogato2 import MultiOutputKriging  # Assicurati di importare la classe

# Supponiamo di avere il file CSV già caricato
file_path = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/training_data.csv"

# Carica il modello addestrato
loaded_model = MultiOutputKriging.load("multi_output_kriging_model.pkl")

# Crea un'istanza della classe senza ricaricare il dataset
model = MultiOutputKriging(file_path)  # Carica i dati per la nuova istanza se necessario
model.model_cl = loaded_model.model_cl  # Usa il modello CL caricato
model.model_cd = loaded_model.model_cd  # Usa il modello CD caricato

# Nuovi dati di test (ad esempio)
new_data = np.array(
    [
        [10000, 0.8, 5, 2],  # Esempio 1: Altitude, Mach number, AoA, AoS
        [15000, 0.9, 10, 3],  # Esempio 2
        [20000, 1.0, 15, 4],  # Esempio 3
    ]
)

# Effettua le previsioni
cl_predictions, cd_predictions = model.predict(new_data)

# Visualizza i risultati
for i, (cl, cd) in enumerate(zip(cl_predictions, cd_predictions)):
    print(f"Prediction {i+1}:")
    print(f"Total CL: {cl}")
    print(f"Total CD: {cd}\n")
