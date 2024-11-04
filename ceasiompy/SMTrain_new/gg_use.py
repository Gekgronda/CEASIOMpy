import pickle
import numpy as np
import os
import csv


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


# Funzione per salvare i risultati in un file CSV
def save_to_csv(cl_pred, cd_pred, filename):
    """Salva le previsioni di CL e CD in un file CSV."""
    # Creazione del dizionario per i dati predetti
    predicted_data = [{"Cl predicted": cl, "Cd predicted": cd} for cl, cd in zip(cl_pred, cd_pred)]

    # Scrittura dei dati nel file CSV
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Cl predicted", "Cd predicted"])
        writer.writeheader()  # Scrive l'intestazione
        writer.writerows(predicted_data)  # Scrive le righe dei dati

    print(f"Dati salvati in {filename}")


# Carica il modello
model_filename = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/surrogate_model.pkl"
model = load_model(model_filename)

# Dati di input per la previsione
input_data = np.array(
    [
        [5.86e02, 2.00e-01, 1.40e01, -2.00e00],
        [6.87e02, 2.00e-01, 1.00e00, 1.00e00],
        [2.20e02, 2.00e-01, 1.30e01, -2.00e00],
        [7.51e02, 3.00e-01, 1.00e00, 1.00e00],
        # [1345, 0.3, 11, -8],
        # [12567, 0.3, 10, 5],
        # [2451, 0.4, 13, -11],
        # [9726, 0.4, -2, 12],
        # [14805, 0.4, 5, 12],
        # [11244, 0.1, 4, 11],
        # [1343, 0.3, 11, -7],
        # [9632, 0.4, 8, 2],
        # [11220, 0.5, 6, 10],
        # [12632, 0.4, 7, -13],
        # [4891, 0.4, 4, 13],
        # [4023, 0.3, 14, -3],
        # [12912, 0.3, 6, -1],
        # [6311, 0.5, 2, -11],
        # [9922, 0.4, -2, -2],
    ]
)

# Esegui la previsione
cl_prediction, cd_prediction = make_predictions(model, input_data)

# Definisci il percorso di output per il file CSV
output_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/"
output_csv = "predicted_values.csv"

# Salva i risultati nel file CSV
save_to_csv(cl_prediction, cd_prediction, os.path.join(output_directory, output_csv))

# Stampa i risultati
print(f"Predizione di CL: {cl_prediction}")
print(f"Predizione di CD: {cd_prediction}")
