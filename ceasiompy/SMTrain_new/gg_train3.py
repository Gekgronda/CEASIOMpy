import pandas as pd
import numpy as np
from ceasiompy.SMTrain_new.gg_sm2 import (
    latin_hypercube_sampling,
    match_outputs,
    fit_model,
    predict_model,
    evaluate_model,
    plot_predictions,
    combine_models,
    save_model,
)

# Carica il database
name = input("Insert database name (with .csv extention): ") or "dataset_500_points.csv"
file_path = f"/home/cfse/Stage_Gronda/datasets/{name}"
df = pd.read_csv(file_path)
# Definisci gli input e output
X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values

# Chiedi all'utente di inserire i valori di theta, corr e poly, con valori di default
theta_input = (
    input("Insert theta value (single value, e.g., 0.1, 0.01, 0.001, 0.0001) [default=0.01]: ")
    or "0.01"
)
theta_values = [float(theta_input)]  # Estende theta al numero di dimensioni

corr_value = (
    input(
        "Insert correlation type (e.g., squar_exp, abs_exp, matern32, matern52) [default=matern32]: "
    )
    or "matern32"
)

poly_value = (
    input("Insert polynomial type (e.g., constant, linear, quadratic) [default=quadratic]: ")
    or "quadratic"
)

use_sampling = (
    input("Do you want to apply Latin Hypercube Sampling (yes/no)? [default=no]: ") or "no"
)

if use_sampling.lower() == "yes":
    num_samples = input("Insert number of samples [default=100]: ") or "100"
    X_lhs = latin_hypercube_sampling(X, num_samples)
    y_cl_lhs, y_cd_lhs = match_outputs(X_lhs, X, y_cl, y_cd)
else:
    X_lhs, y_cl_lhs, y_cd_lhs = X, y_cl, y_cd


# Nome del modello
# model_name = input("Insert new model filename: ")

model_cl, model_cd, X_test, y_test_cl, y_test_cd = fit_model(
    X_lhs, y_cl_lhs, y_cd_lhs, theta_values, corr_value, poly_value
)
cl_pred, cd_pred = predict_model(model_cl, model_cd, X_test)
errors = evaluate_model(model_cl, model_cd, X_test, y_test_cl, y_test_cd, cl_pred, cd_pred)
plot = plot_predictions(y_test_cl, y_test_cd, cl_pred, cd_pred)
# Esempio di utilizzo
model = combine_models(model_cl, model_cd)
model_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/"
model_name = "surrogate_model.pkl"
save = save_model(model, f"{model_directory}{model_name}")
