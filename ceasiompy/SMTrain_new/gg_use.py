import pickle
import smt
import numpy as np
from smt.surrogate_models import KRG

sm2 = None
filename = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/model.pkl"
with open(filename, "rb") as f:
    loaded_data = pickle.load(f)

# Se il file contiene un dizionario, estrai il modello
if isinstance(loaded_data, dict):
    sm2 = loaded_data.get("model")  # Assumi che il modello sia sotto la chiave 'model'
else:
    sm2 = loaded_data

# Verifica che il modello sia un'istanza di KRG
if sm2 is None or not hasattr(sm2, "predict_values"):
    raise ValueError("Il modello Kriging non è stato trovato o non è valido!")

# Prepara i dati come un array bidimensionale
data = np.array([[100, 0.5, 1, 1]])  # Dati di input come array 2D

# Effettua la predizione
prediction = sm2.predict_values(data)
print(prediction)
