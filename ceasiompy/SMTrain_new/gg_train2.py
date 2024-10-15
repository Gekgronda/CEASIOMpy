import pandas as pd
import numpy as np
from gg_sm2 import MultiOutputKriging  # Assicurati di importare la classe

# Supponiamo di avere il file CSV già caricato
name = input("Insert database name (with .csv extention): ")
file_path = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/{name}"
df = pd.read_csv(file_path)

# Definisci gli input e output
X = df[["Altitude", "Mach number", "Angle of attack (AoA)", "Angle of sideslip (AoS)"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values

# Crea un'istanza della classe
model = MultiOutputKriging()

# Imposta i valori di theta, corr e poly
# Chiedi all'utente di inserire i valori di theta, corr e poly
theta_input = input("Insert theta value (single value, e.g., 0.01): ")
theta_values = [float(theta_input)] * X.shape[1]  # Estende theta al numero di dimensioni
corr_value = input("Insert correlation type (e.g., squar_exp, abs_exp, matern32, matern52): ")
poly_value = input("Insert polynomial type (e.g., constant, linear, quadratic): ")

# Addestra il modello
model.fit(X, y_cl, y_cd, theta=theta_values, corr=corr_value, poly=poly_value)

# Valuta il modello
model.evaluate(model.X_val, model.y_cl_val, model.y_cd_val)

# Visualizza le predizioni
model.plot_predictions(X, y_cl, y_cd)

# Richiedi all'utente di inserire nuovi dati per fare previsioni
n_predictions = int(input("How many predictions do you want to make? "))

new_data = []
for i in range(n_predictions):
    altitude = float(input(f"Insert Altitude for prediction {i+1}: "))
    mach = float(input(f"Insert Mach number for prediction {i+1}: "))
    aoa = float(input(f"Insert Angle of attack (AoA) for prediction {i+1}: "))
    aos = float(input(f"Insert Angle of sideslip (AoS) for prediction {i+1}: "))
    new_data.append([altitude, mach, aoa, aos])

new_data = np.array(new_data)
# Effettua le previsioni
cl_predictions, cd_predictions = model.predict(new_data)

# Visualizza i risultati
for i, (cl, cd) in enumerate(zip(cl_predictions, cd_predictions)):
    print(f"Prediction {i+1}:")
    print(f"Total CL: {cl}")
    print(f"Total CD: {cd}\n")
