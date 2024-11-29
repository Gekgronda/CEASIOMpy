import pandas as pd
import numpy as np
from ceasiompy.SMTrain_new.codes.gg_sm2 import compare_models, split_data

# Carica il database
name = input("Insert database name (with .csv extention): ") or "takeoff_tot_d150.csv"
file_path = f"/home/cfse/Stage_Gronda/datasets/{name}"
df = pd.read_csv(file_path)

# Definisci gli input e output
X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values
y_cs = df["Total CL"].values
y_cmx = df["Total CL"].values
y_cmy = df["Total CL"].values
y_cmz = df["Total CL"].values
(
    X_train,
    X_val,
    X_test,
    y_cl_train,
    y_cl_val,
    y_cl_test,
    y_cd_train,
    y_cd_val,
    y_cd_test,
    y_cs_train,
    y_cs_val,
    y_cs_test,
    y_cmx_train,
    y_cmx_val,
    y_cmx_test,
    y_cmy_train,
    y_cmy_val,
    y_cmy_test,
    y_cmz_train,
    y_cmz_val,
    y_cmz_test,
) = split_data(X, y_cl, y_cd, y_cs, y_cmx, y_cmy, y_cmz)

print("X_test shape:", X_test.shape)
print("y_cl_test shape:", y_cl_test.shape)
print("X_train shape:", X_train.shape)
print("y_cl_train shape:", y_cl_train.shape)

compare_models(X_test, y_cl_test, X_train, y_cl_train)
