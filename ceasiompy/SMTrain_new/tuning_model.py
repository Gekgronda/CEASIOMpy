import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Carica i dati
data = "/home/cfse/Stage_Gronda/sonar.csv"
dataframe = pd.read_csv(data, header=0)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
y = y.astype(int)

# GRID SEARCH NON FUNZIONA
# # Converte la variabile target (y) in valori numerici
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)


# Suddividi i dati in training e test set
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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
    estimator = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_features=min(max_features, 0.999),
        random_state=42,
        n_jobs=-1,
    )
    cval = cross_val_score(estimator, X_train, y_train, scoring="accuracy", cv=5)
    return cval.mean()


params = {"n_estimators": (10, 200), "max_depth": (1, 50), "max_features": (0.1, 0.999)}

optimizer = BayesianOptimization(f=rf_cv, pbounds=params, random_state=42)
optimizer.maximize(init_points=20, n_iter=50)

best_params = optimizer.max["params"]
best_rf = RandomForestClassifier(
    n_estimators=int(best_params["n_estimators"]),
    max_depth=int(best_params["max_depth"]),
    max_features=min(best_params["max_features"], 0.999),
    n_jobs=-1,
)

best_rf.fit(X_train, y_train)
best_accuracy = cross_val_score(best_rf , X_train, y_train, scoring="accuracy", cv=5).mean()
print(f"Best Acc after BayOpt: {best_accuracy:.4f}")
