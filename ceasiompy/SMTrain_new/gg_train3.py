import pandas as pd
import numpy as np
import os
from ceasiompy.SMTrain_new.gg_sm2 import (
    fit_model,
    predict_model,
    evaluate_model,
    plot_predictions,
    combine_models,
    save_model,
)

# Carica il database
name = input("Insert database name (with .csv extention): ") or "gg_td.csv"
file_path = f"/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/{name}"
df = pd.read_csv(file_path)
# Definisci gli input e output
X = df[["Altitude", "Mach number", "Angle of attack (AoA)", "Angle of sideslip (AoS)"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values

# Chiedi all'utente di inserire i valori di theta, corr e poly, con valori di default
theta_input = (
    input("Insert theta value (single value, e.g., 0.1, 0.01, 0.001, 0.0001) [default=0.1]: ")
    or "0.1"
)
theta_values = [float(theta_input)]  # Estende theta al numero di dimensioni

corr_value = (
    input(
        "Insert correlation type (e.g., squar_exp, abs_exp, matern32, matern52) [default=matern32]: "
    )
    or "matern32"
)

poly_value = (
    input("Insert polynomial type (e.g., constant, linear, quadratic) [default=linear]: ")
    or "linear"
)

# Nome del modello
# model_name = input("Insert new model filename: ")


model_cl, model_cd, X_test, y_test_cl, y_test_cd = fit_model(
    X, y_cl, y_cd, theta_values, corr_value, poly_value
)
cl_pred, cd_pred = predict_model(model_cl, model_cd, X_test)
errors = evaluate_model(y_test_cl, y_test_cd, cl_pred, cd_pred)
plot = plot_predictions(y_test_cl, y_test_cd, cl_pred, cd_pred)
# Esempio di utilizzo
model = combine_models(model_cl, model_cd)
model_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/"
model_name = "model.pkl"
save = save_model(model, f"{model_directory}{model_name}")
