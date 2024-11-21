import pandas as pd
import numpy as np
from ceasiompy.SMTrain_new.gg_sm2 import compare_models, split_data

# Carica il database
name = input("Insert database name (with .csv extention): ") or "dataset_500_points.csv"
file_path = f"/home/cfse/Stage_Gronda/datasets/{name}"
df = pd.read_csv(file_path)

# Definisci gli input e output
X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values
(
    X_train,
    y_cl_train,
    X_val,
    y_cl_val,
    X_test,
    y_cl_test,
    X_temp,
    y_cd_train,
    X_val,
    y_cd_val,
    X_test,
    y_cd_test,
) = split_data(X, y_cl, y_cd)
compare_models(X_test, y_cl_test, X_train, y_cl_train)
