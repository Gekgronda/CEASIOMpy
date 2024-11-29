# from smt.applications.mfck import MFCK
from smt.applications.mfk import MFK
import numpy as np
from matplotlib import pyplot as plt
from smt.sampling_methods import LHS
from smt.applications.mfk import NestedLHS
from smt.surrogate_models import KRG
from matplotlib import cm
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("TkAgg")


# Funzione per caricare e pre-processare i dati
def load_data(df_lf, df_hf):
    """
    Carica i dati dal dataframe fornito.
    Args:
        df_lf: Dataframe per dati a bassa fedeltà.
        df_hf: Dataframe per dati ad alta fedeltà.
    Returns:
        x_lf, y_lf, x_hf, y_hf: input/output per LF e HF.
    """
    # Input: Altitude, MachNumber, AngleOfAttack, AngleOfSideslip
    # Output: Total CL, Total CD
    x_lf = df_lf[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
    y_lf = df_lf[["Total CL", "Total CD"]].values

    x_hf = df_hf[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
    y_hf = df_hf[["Total CL", "Total CD"]].values

    return x_lf, y_lf, x_hf, y_hf


def preprocess_data(x_hf, y_hf):
    """
    Divide i dati HF in training e test set.
    Args:
        x_hf: Input dati HF.
        y_hf: Output dati HF.
    Returns:
        x_train, y_train, x_test, y_test: Dati divisi in training e test.
    """
    x_train, x_test, y_train, y_test = train_test_split(x_hf, y_hf, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test


# Esempio: Supponiamo che df_lf e df_hf siano già forniti
# Includi il caricamento reale dei dataframe `df_lf` e `df_hf` qui sotto
df_lf = pd.read_csv("/wrk/Gronda/labAR/EULER/takeoff0/takeoff_train.csv")
df_hf = pd.read_csv("/wrk/Gronda/labAR/RANS/RANS_train_dataset.csv")

# Carica i dati
x_lf, y_lf, x_hf, y_hf = load_data(df_lf, df_hf)

# Prepara i dati HF (divisione train/test)
x_train, y_train, x_test, y_test = preprocess_data(x_hf, y_hf)

# Creazione e addestramento dei modelli per Cd e Cl
models = {}
outputs = ["cl", "cd"]
for i, output in enumerate(outputs):

    print(f"Training MFK model for {output.upper()}...")

    # Creazione del modello MFK
    sm = MFK(theta0=[1.0, 1.0, 1.0, 1.0], corr="squar_exp", poly="constant")

    # Impostazione dei dati di training
    sm.set_training_values(x_lf, y_lf[:, i], name=0)  # Dati di bassa fedeltà (LF)
    sm.set_training_values(x_train, y_train[:, i])  # Dati di alta fedeltà (HF)

    # Addestramento
    sm.train()

    # Salva il modello
    models[output] = sm

    # Predizioni sul set di test
    y_pred = sm.predict_values(x_test)
    print(f"Predictions for {output.upper()} on test set: {y_pred.flatten()}")
    print(x_test, y_test)

# Nota: Modifica `df_lf` e `df_hf` in base al caricamento dei tuoi dataset.


# ndim = 4
# nlvl = 2
# ndoe_HF = 10

# xlimits = np.array([[0.0, 1.0]])
# xdoes = NestedLHS(nlevel=nlvl, xlimits=xlimits, random_state=2)
# Xt_c, Xt_e = xdoes(ndoe_HF)
# ndoe_LF = np.shape(Xt_c)[0]

# yt_e = HF_function(Xt_e)
# yt_c = LF_function(Xt_c)

# sm = MFK(theta0=Xt_e.shape[1] * [1.0])

# sm.set_training_values(Xt_c, yt_c, name=0)
# sm.set_training_values(Xt_e, yt_e)

# sm.train()

# # test training
# x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

# # query the outputs
# y = sm.predict_values(x)

# print(y)

# n_start=100
# opti='Cobyla'

# # Funzioni per gestire i dati e simulazioni
# def load_data():
#     """
#     Funzione per caricare i dataset.
#     - dataset_lf: punti di simulazione Euleriana (bassa fedeltà)
#     - dataset_hf: punti di simulazione RANS (alta fedeltà)
#     """
#     # Supponiamo che i dati siano già disponibili in due array separati
#     # Dati di esempio: [Mach, Altitude, AoA, AoS]
#     dataset_lf = np.loadtxt("/wrk/Gronda/labAR/EULER/takeoff0/takeoff_train.csv", delimiter=",")  # [N_points, 6] (4 input + 2 output)
#     dataset_hf = np.loadtxt("/wrk/Gronda/labAR/RANS/RANS_train_dataset.csv", delimiter=",")  # [20, 6] (4 input + 2 output)

#     X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
# y_cl = df["Total CL"].values
# y_cd = df["Total CD"].values
#     x_lf, y_lf = dataset_lf[:, :4], dataset_lf[:, 4:]  # Input e Output per Euler
#     x_hf, y_hf = dataset_hf[:, :4], dataset_hf[:, 4:]  # Input e Output per RANS

#     return x_lf, y_lf, x_hf, y_hf


# def preprocess_data(x_lf, y_lf, x_hf, y_hf):
#     """
#     Prepara i dati per il modello multifidelity. Opzionale: suddivisione training/test.
#     """
#     x_test, y_test = train_test_split(np.hstack((x_hf, y_hf)), test_size=0.2, random_state=42)
#     return x_test[:, :4], x_test[:, 4:]


# # Carica i dati
# x_lf, y_lf, x_hf, y_hf = load_data()
# x_test, y_test = preprocess_data(x_lf, y_lf, x_hf, y_hf)

# # Creazione e addestramento dei modelli per Cd e Cl
# models = {}
# outputs = ["cd", "cl"]
# for i, output in enumerate(outputs):
#     print(f"Training MFK model for {output.upper()}...")

#     # Creazione del modello MFK
#     sm = MFK(theta0=[1., 1., 1., 1.], corr="squar_exp", poly="quadratic")

#     # Impostazione dei dati di training
#     sm.set_training_values(x_lf, y_lf[:, i], name=0)  # Dati di bassa fedeltà (Euler)
#     sm.set_training_values(x_hf, y_hf[:, i])         # Dati di alta fedeltà (RANS)

#     # Addestramento
#     sm.train()

#     # Salva il modello
#     models[output] = sm

#     # Predizioni sul set di test
#     y_pred = sm.predict_values(x_test)
#     print(f"Predictions for {output.upper()}: {y_pred.flatten()}")
