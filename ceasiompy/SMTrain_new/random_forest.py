import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import export_graphviz
import graphviz

# Carica il database
file_path = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/dataset_prova_libro.csv"
df = pd.read_csv(file_path)
# Definisci gli input e output
X = df[["Altitude", "Mach", "AoA", "AoS"]].values
y_cl = df["Total CD"].values

X_train, X_test, y_cl_train, y_cl_test = train_test_split(X, y_cl, test_size=0.3, random_state=10)


# il modello random forest e molto semplice e al contempo molto efficace
regressor = RandomForestRegressor(n_estimators=100, random_state=41)
regressor.fit(X_train, y_cl_train)
y_pred = regressor.predict(X_test)


# possiamo definire l'importanza delle diverse features!
importances = regressor.feature_importances_
features = ["Altitude", "Mach", "AoA", "AoS"]
plt.barh(features, importances)


# visualizziamo come l cambiamento di una feature influenza l'output
Display1 = PartialDependenceDisplay.from_estimator(regressor, X, [2])


# Esportare l'albero come file .dot
# tree = regressor.estimators_[1]  # Prendi il primo albero
# export_graphviz(tree, out_file="tree.dot", feature_names=features, filled=True, rounded=True)

# # Convertire il file .dot in un file .png usando Graphviz
# os.system("dot -Tpng tree.dot -o tree.png")

# # Visualizzare l'immagine generata
# from PIL import Image

# img = Image.open("tree.png")
# img.show()



