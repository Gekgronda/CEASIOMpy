import pickle
import numpy as np
import os
import csv
import pandas as pd
from smt.surrogate_models import KRG
from smt.applications import EGO, MFK, MFKPLS, MFKPLSK


# Funzione per caricare il modello salvato
def load_model(path):
    """Carica il modello salvato da file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Il file {path} non esiste.")
    with open(path, "rb") as p:
        model = pickle.load(p)
    return model


# Imposta i percorsi
directory_path = "/wrk/Gronda/validazione/mengmeng"
model_name = "surrogate_model_CL.pkl"  # Il nome del modello
dataset_path = "/wrk/Gronda/validazione/mengmeng/dataset.csv"  # Inserisci il percorso corretto

# Carica il modello
model_path = os.path.join(directory_path, model_name)
model = load_model(model_path)

# Carica il dataset
df = pd.read_csv(dataset_path)
X_pred = df.to_numpy()

# Esegui le previsioni
y_pred = model.predict_values(X_pred)

# Recupera il nome del coefficiente dal nome del modello
coeff_name = model_name.replace("surrogate_model_", "").replace(".pkl", "").upper()
prediction_col_name = f"{coeff_name}_predictions"

# Aggiungi la colonna delle previsioni al DataFrame
df[prediction_col_name] = y_pred

# Salva il nuovo file CSV con "_PRED" aggiunto al nome
predicted_csv_path = dataset_path.replace(".csv", "_PRED.csv")
df.to_csv(predicted_csv_path, index=False)

print(f"File salvato: {predicted_csv_path}")
print(f"Colonna delle previsioni aggiunta: {prediction_col_name}")
