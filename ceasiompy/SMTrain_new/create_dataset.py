import numpy as np
import pandas as pd

# Generazione dei dati
altitude = np.random.randint(0, 15000, 500)
mach = np.round(np.random.uniform(0.1, 0.6, 500), 1)
aoa = np.random.randint(-4, 15, 500)
aos = np.random.randint(-15, 15, 500)

# Creazione di un DataFrame con i dati
df = pd.DataFrame({"Altitude": altitude, "Mach": mach, "AoA": aoa, "AoS": aos})

# Salvataggio del DataFrame in un file CSV
filename = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/dataset.csv"
df.to_csv(filename, index=False)

print(f"File saved to {filename}")

filename = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/dataset.csv"
