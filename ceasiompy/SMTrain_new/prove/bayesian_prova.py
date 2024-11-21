import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import os
from mango import Tuner
from smt.surrogate_models import KRG
from ceasiompy.SMTrain_new.gg_sm2 import split_data, train_surrogate_model

# Carica il database
name = input("Insert database name (with .csv extention): ") or "takeoff_totale.csv"
file_path = os.path.join("/home/cfse/Stage_Gronda/datasets", name)
df = pd.read_csv(file_path)

# Definisci gli input e output
X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values

# Suddividi i dati per addestramento, validazione e test

X_train, X_val, X_test, y_cl_train, y_cl_val, y_cl_test, y_cd_train, y_cd_val, y_cd_test = (
    split_data(X, y_cl, y_cd)
)

param_space = {
    "theta_values": range(1, 2),
    "poly_value": ["constant", "linear"],
    "corr_value": ["pow_exp", "abs_exp"],
}


def objective(list_parameters):
    global X_train, y_cl_train, X_val, y_cl_val

    results = []
    for hyper_params in list_parameters:
        model = Kriging(X_train, y_cl_train, "Kriging", list_parameters)
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        error = np.sqrt(mean_squared_error(y_validation, prediction))
        results.append(error)
    return results


start_time = time.time()
tuner = Tuner(param_space, objective, dict(num_iteration=2, initial_random=10))  # Initialize Tuner
optimisation_results = tuner.minimize()
print(f"The optimisation in series takes {(time.time()-start_time)/60.} minutes.")

# Inspect the results
print("best parameters:", optimisation_results["best_params"])
print("best accuracy (RMSE):", optimisation_results["best_objective"])

best_theta = [
    0.001,
    0.01,
][best_params["theta"]]
best_poly = ["constant", "linear"][best_params["poly"]]
best_corr = ["pow_exp", "abs_exp"][best_params["corr"]]
best_model = Kriging(X_train, y_cl_train, best_theta, best_corr, best_poly)
best_model.fit(X_train, y_cl_train)
y_pred = best_model.predict(X_test)  # to get the real value not in log scale
print("rmse on test:", np.sqrt(mean_squared_error(y_test, y_pred)))
