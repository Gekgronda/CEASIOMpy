import pickle
import numpy as np
import os


# Funzione per caricare il modello salvato
def load_model(filename):
    """Carica il modello salvato da file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Il file {filename} non esiste.")
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


# Funzione per fare previsioni
def make_predictions(model, input_data):
    """Fai previsioni per CL e CD utilizzando il modello caricato."""
    model_cl = model["model_cl"]
    model_cd = model["model_cd"]

    cl_pred = model_cl.predict_values(input_data)
    cd_pred = model_cd.predict_values(input_data)

    return cl_pred, cd_pred


# Carica il modello
model_filename = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/model.pkl"
model = load_model(model_filename)

# Dati di input per la previsione
input_data = np.array([[9025, 0.4, 6, 3]])  # Cambia i valori in base alle tue necessità

# Esegui la previsione
cl_prediction, cd_prediction = make_predictions(model, input_data)

# Stampa i risultati
print(f"Predizione di CL: {cl_prediction}")
print(f"Predizione di CD: {cd_prediction}")
