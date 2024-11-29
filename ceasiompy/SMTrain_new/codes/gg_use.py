import pickle
import numpy as np
import os
import csv
import pandas as pd


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
    """Fai previsioni per CL, CD, CS, CMX, CMY e CMZ utilizzando il modello caricato."""
    model_cl = model["model_cl"]
    model_cd = model["model_cd"]
    model_cs = model["model_cs"]
    model_cmx = model["model_cmx"]
    model_cmy = model["model_cmy"]
    model_cmz = model["model_cmz"]

    # Previsioni per ogni coefficiente
    cl_pred = model_cl.predict_values(input_data)
    cd_pred = model_cd.predict_values(input_data)
    cs_pred = model_cs.predict_values(input_data)
    cmx_pred = model_cmx.predict_values(input_data)
    cmy_pred = model_cmy.predict_values(input_data)
    cmz_pred = model_cmz.predict_values(input_data)

    return cl_pred, cd_pred, cs_pred, cmx_pred, cmy_pred, cmz_pred


# Funzione per salvare i risultati in un file CSV
def save_to_csv(
    cl_prediction,
    cd_prediction,
    cs_predictions,
    cmx_predictions,
    cmy_predictions,
    cmz_predictions,
    filename,
):
    """Salva le previsioni di CL, CD, CS, CMX, CMY e CMZ in un file CSV."""
    # Creazione del dizionario per i dati predetti
    predicted_data = [
        {
            "Cl predicted": cl,
            "Cd predicted": cd,
            "Cs predicted": cs,
            "Cmx predicted": cmx,
            "Cmy predicted": cmy,
            "Cmz predicted": cmz,
        }
        for cl, cd, cs, cmx, cmy, cmz in zip(
            cl_prediction,
            cd_prediction,
            cs_predictions,
            cmx_predictions,
            cmy_predictions,
            cmz_predictions,
        )
    ]

    # Scrittura dei dati nel file CSV
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Cl predicted",
                "Cd predicted",
                "Cs predicted",
                "Cmx predicted",
                "Cmy predicted",
                "Cmz predicted",
            ],
        )
        writer.writeheader()  # Scrive l'intestazione
        writer.writerows(predicted_data)  # Scrive le righe dei dati

    print(f"Dati salvati in {filename}")


# Carica il modello
model_filename = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/z_surrogate_models/surrogate_model_6coeff.pkl"
model = load_model(model_filename)

# Carica i nuovi dati per la previsione
new_data_file = "/home/cfse/Stage_Gronda/datasets/prediction_takeoff.csv"  # Modifica con il percorso del tuo file CSV
new_data = pd.read_csv(new_data_file, header=0)
X_new = new_data.values

# Esegui la previsione
cl_prediction, cd_prediction, cs_predictions, cmx_predictions, cmy_predictions, cmz_predictions = (
    make_predictions(model, X_new)
)

# Definisci il percorso di output per il file CSV
output_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/z_predictions/"
output_csv = "predicted_values_sm.csv"

# Salva i risultati nel file CSV
save_to_csv(
    cl_prediction,
    cd_prediction,
    cs_predictions,
    cmx_predictions,
    cmy_predictions,
    cmz_predictions,
    os.path.join(output_directory, output_csv),
)

# Stampa i risultati
print(f"Predizione di CL: {cl_prediction}")
print(f"Predizione di CD: {cd_prediction}")
print(f"Predizione di CS: {cs_predictions}")
print(f"Predizione di CMX: {cmx_predictions}")
print(f"Predizione di CMY: {cmy_predictions}")
print(f"Predizione di CMZ: {cmz_predictions}")
