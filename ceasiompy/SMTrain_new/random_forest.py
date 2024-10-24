import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Carica il database
file_path = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/dataset_prova_libro.csv"
df = pd.read_csv(file_path)
# Definisci gli input e output
X = df[["Altitude", "Mach", "AoA", "AoS"]].values
y_cl = df["Total CD"].values

X_train, X_test, y_cl_train, y_cl_test = train_test_split(X, y_cl, test_size=0.3, random_state=10)

regressor = RandomForestRegressor(n_estimators=100, random_state=41)
regressor.fit(X_train, y_cl_train)
y_pred = regressor.predict(X_test)

importances = regressor.feature_importances_
features = ["Altitude", "Mach", "AoA", "AoS"]
plt.barh(features, importances)
plt.show()
