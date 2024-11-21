import pandas as pd
import numpy as np
import os
from ceasiompy.SMTrain_new.gg_sm2 import (
    latin_hypercube_sampling,
    get_model_params,
    # noise da implementare
    predict_model,
    split_data,
    evaluate_model,
    plot_predictions,
    combine_models,
    optimize_hyperparameters_ego,
    save_model,
    compare_models,
    train_surrogate_model,
)
import matplotlib.pyplot as plt
import numpy as np

from smt.surrogate_models import RMTB

# xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
# yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

# xlimits = np.array([[0.0, 4.0]])


# Carica il database
name = input("Insert database name (with .csv extention): ") or "takeoff_totale.csv"
file_path = os.path.join("/home/cfse/Stage_Gronda/datasets", name)
df = pd.read_csv(file_path)

# Definisci gli input e output
X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
y_cl = df["Total CL"].values
xlimits = np.array([[0, 1000], [0.1, 0.3], [0, 15], [-2, 2]])

"""Train Regularized minimal-energy tensor-product B-splines model"""
model = RMTB(
    xlimits=xlimits,
    smoothness=1.0,
    order=2,
    grad_weight=0.5,
    approx_order=4,
    solver_tolerance=1e-14,
    num_ctrl_pts=15,
    energy_weight=0.0001,
    regularization_weight=1e-14,
    min_energy=True,
    nonlinear_maxiter=10,
    # print_global=False,
)
model.set_training_values(X, y_cl)
model.train()

# sm = RMTB(
#     xlimits=xlimits,
#     order=4,
#     num_ctrl_pts=20,
#     energy_weight=1e-15,
#     regularization_weight=0.0,
# )
# sm.set_training_values(xt, yt)
# sm.train()

# num = 100
# x = np.linspace(0.0, 4.0, num)
# y = sm.predict_values(x)

# plt.plot(xt, yt, "o")
# plt.plot(x, y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(["Training data", "Prediction"])
# plt.show()
# par = get_model_params("RMTB")
# print(par)
