import numpy as np
import pandas as pd

# Generazione dei dati
altitude = np.random.randint(0, 15000, 15)
mach = np.round(np.random.uniform(0.1, 0.6, 15), 1)
aoa = np.random.randint(-4, 15, 15)
aos = np.random.randint(-15, 15, 15)

# Creazione di un DataFrame con i dati
df = pd.DataFrame(
    {"altitude": altitude, "machNumber": mach, "angleOfAttack": aoa, "angleOfSideslip": aos}
)

# Salvataggio del DataFrame in un file CSV
filename = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/aeromap1.csv"
df.to_csv(filename, index=False)

print(f"File saved to {filename}")
