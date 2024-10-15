import pandas as pd
import numpy as np
from gg_sm2 import MultiOutputKriging  # Assicurati di importare la classe

# Supponiamo di avere il file CSV già caricato
# name = input("Insert database name (with .csv extention): ")
file_path = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/gg_td.csv"
df = pd.read_csv(file_path)

# Definisci gli input e output
X = df[["Altitude", "Mach number", "Angle of attack (AoA)", "Angle of sideslip (AoS)"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values

# Crea un'istanza della classe
model = MultiOutputKriging()

# Imposta i valori di theta, corr e poly
theta_values = [1e-2] * X.shape[1]  # Esempio di valori di theta
corr_value = "matern32"  # Esempio di tipo di correlazione
poly_value = "linear"  # Esempio di grado del polinomio

# Addestra il modello
model.fit(X, y_cl, y_cd, theta=theta_values, corr=corr_value, poly=poly_value)

# Valuta il modello
model.evaluate(model.X_val, model.y_cl_val, model.y_cd_val)

# Visualizza le predizioni
model.plot_predictions(X, y_cl, y_cd)


# Nuovi dati di test (ad esempio)
# new_data = np.array(
#     [
#         [10000, 0.8, 5, 2],  # Esempio 1: Altitude, Mach number, AoA, AoS
#         [15000, 0.9, 10, 3],  # Esempio 2
#         [20000, 1.0, 15, 4],  # Esempio 3
#     ]
# )

# # Effettua le previsioni
# cl_predictions, cd_predictions = model.predict(new_data)

# # Visualizza i risultati
# for i, (cl, cd) in enumerate(zip(cl_predictions, cd_predictions)):
#     print(f"Prediction {i+1}:")
#     print(f"Total CL: {cl}")
#     print(f"Total CD: {cd}\n")
