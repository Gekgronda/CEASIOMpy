import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, make_scorer
from ceasiompy.SMTrain_new.gg_sm2 import split_data


# Carica il database
name = input("Insert database name (with .csv extention): ") or "takeoff_totale.csv"
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
print(X.shape, y_cd.shape)

# # Definisci i parametri per la grid search
# param_grid = {
#     "n_estimators": [10, 50, 100, 150, 200],
#     "max_depth": [None, 10, 20, 30, 40, 50],
#     "max_features": ["sqrt", "log2"],
#     "bootstrap": [True, False],
# }

# # Configura e esegui la grid search
# rf = RandomForestClassifier()
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")
# grid_search.fit(X_train, y_train)


# BAYESIAN OPTIMIZATION
def rf_cv(n_estimators, max_depth, max_features):
    estimator = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_features=min(max_features, 0.999),
        random_state=42,
        n_jobs=-1,
    )
    cval = cross_val_score(
        estimator,
        X_train,
        y_cd_train,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=5,
    )
    return cval.mean()


params = {"n_estimators": (10, 200), "max_depth": (1, 50), "max_features": (0.1, 0.999)}

optimizer = BayesianOptimization(f=rf_cv, pbounds=params, random_state=42)
optimizer.maximize(init_points=20, n_iter=50)

best_params = optimizer.max["params"]
best_rf = RandomForestRegressor(
    n_estimators=int(best_params["n_estimators"]),
    max_depth=int(best_params["max_depth"]),
    max_features=min(best_params["max_features"], 0.999),
    n_jobs=-1,
)

# Fit the model with the best parameters
best_rf.fit(X_train, y_cd_train)

# Evaluate the model on the test set
predictions = best_rf.predict(X_test)
mse = mean_squared_error(y_cd_test, predictions)
print(f"Best MSE after Bayesian Optimization: {mse:.4f}")

# ---- Prediction on New CSV Data ----

# Load new data for prediction
new_data_file = (
    "/home/cfse/Stage_Gronda/datasets/takeoff4_calc.csv"  # Update with the path to your CSV file
)
new_data = pd.read_csv(new_data_file, header=0)
X_new = new_data.values  # Assuming the new CSV has only features without target values

# Make predictions on the new data
new_predictions = best_rf.predict(X_new)

# Save predictions to a new CSV file
output_df = pd.DataFrame(new_predictions, columns=["Predicted Values"])
output_df.to_csv(
    "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/predicted_values_randomfor.csv",
    index=False,
)
print(
    "Predictions saved to /home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/predicted_values_randomfor.csv"
)
