import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

RANDOM_STATE = 55  ## We will pass it to every sklearn call so we ensure reproducibility

from ceasiompy.SMTrain_new.gg_sm2 import split_data

# Load the dataset
name = input("Insert database name (with .csv extension): ") or "takeoff0.csv"
file_path = f"/home/cfse/Stage_Gronda/datasets/{name}"
df = pd.read_csv(file_path)

# Define inputs and outputs
X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values

# Split data using the custom split_data function
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

xgb_model = XGBRegressor(
    n_estimators=500, learning_rate=0.1, verbosity=1, random_state=RANDOM_STATE
)
xgb_model.fit(X_train, y_cl_train, eval_set=[(X_val, y_cl_val)])

input_data = np.array(
    [
        [5.86e02, 2.00e-01, 1.40e01, -2.00e00],
        [6.87e02, 2.00e-01, 1.00e00, 1.00e00],
        [2.20e02, 2.00e-01, 1.30e01, -2.00e00],
        [7.51e02, 3.00e-01, 1.00e00, 1.00e00],
    ]
)

predicted_cl = xgb_model.predict(input_data)
print(predicted_cl)
print(y_cl_test[:4])
