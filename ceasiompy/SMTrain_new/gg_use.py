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
model_filename = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/surrogate_model.pkl"
model = load_model(model_filename)

# Dati di input per la previsione
input_data = np.array(
    [
        [1345, 0.3, 11, -8],
        [12567, 0.3, 10, 5],
        [2451, 0.4, 13, -11],
        [9726, 0.4, -2, 12],
        [14805, 0.4, 5, 12],
        [11244, 0.1, 4, 11],
        [1343, 0.3, 11, -7],
        [9632, 0.4, 8, 2],
        [11220, 0.5, 6, 10],
        [12632, 0.4, 7, -13],
        [4891, 0.4, 4, 13],
        [4023, 0.3, 14, -3],
        [12912, 0.3, 6, -1],
        [6311, 0.5, 2, -11],
        [9922, 0.4, -2, -2],
    ]
)

print(input_data)
# Cambia i valori in base alle tue necessità

# Esegui la previsione
cl_prediction, cd_prediction = make_predictions(model, input_data)

# Stampa i risultati
print(f"Predizione di CL: {cl_prediction}")
print(f"Predizione di CD: {cd_prediction}")
