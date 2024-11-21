# from skopt import BayesSearchCV
# # parameter ranges are specified by one of below
# from skopt.space import Real, Categorical, Integer

# from sklearn.datasets import load_iris
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split


# def Kriging(X_train, y_train, theta, corr, poly):
#     """Train Kriging model."""
#     model = KRG(theta0=[theta], corr=corr, poly=poly, print_global=False)
#     model.set_training_values(X_train, y_train)
#     model.train()
#     return model


# def ego_optimize(X_train, y_train, X_val, y_val, n_calls=50):
#     opt = BayesSearchCV(
#         SVC(),
#     {
#         'C': (1e-6, 1e+6, 'log-uniform'),
#         'gamma': (1e-6, 1e+1, 'log-uniform'),
#         'degree': (1, 8),  # integer valued parameter
#         'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
#     },
#     n_iter=32,
#     cv=3)


# def optimize_hyperparameters_ego_bayes(X_train, y_train, X_val, y_val, n_calls=50):
#     """
#     Ottimizzazione iperparametri con ottimizzazione bayesiana.
#     """
#     # Definisci lo spazio dei parametri
#     space = [
#         Categorical([0.0001, 0.001, 0.01, 0.1, 1], name="theta"),  # intervallo continuo per theta
#         Categorical(["constant", "linear", "quadratic"], name="poly"),
#         Categorical(
#             ["pow_exp", "abs_exp", "squar_exp", "squar_sin_exp", "matern52", "matern32"],
#             name="corr",
#         ),
#     ]

#     # Funzione obiettivo per skopt
#     def objective(params):
#         theta, poly, corr = params
#         model = Kriging(X_train, y_train, theta, corr, poly)
#         y_pred = model.predict(X_val)[0]
#         mse = mean_squared_error(y_val, y_pred)
#         return mse

#     # Ottimizzazione bayesiana con skopt
#     result = gp_minimize(objective, space, n_calls=15, random_state=42)

#     # Ritorna i risultati migliori
#     best_params = {"theta": result.x[0], "poly": result.x[1], "corr": result.x[2]}
#     best_model = Kriging(
#         X_train, y_train, best_params["theta"], best_params["corr"], best_params["poly"]
#     )
#     best_model.train()
#     best_mse = result.fun

#     return best_model, best_params, best_mse

from hyperopt import fmin, tpe, hp, Trials, rand
from hyperopt.pyll.base import scope
from sklearn.metrics import mean_squared_error
from smt.surrogate_models import KRG
import numpy as np


def Kriging(X_train, y_train, theta, corr, poly):
    """Funzione per addestrare il modello Kriging."""
    model = KRG(theta0=[theta], corr=corr, poly=poly, print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def optimize_hyperparameters_hyperopt(X_train, y_train, X_val, y_val, max_evals=50):
    """
    Ottimizza gli iperparametri di un modello Kriging utilizzando Hyperopt.

    Parametri:
    - X_train: array delle feature di addestramento.
    - y_train: array delle etichette di addestramento.
    - X_val: array delle feature di validazione.
    - y_val: array delle etichette di validazione.
    - max_evals: numero massimo di valutazioni di iperparametri.

    Ritorna:
    - Miglior modello addestrato.
    - Migliori parametri trovati.
    - Errore quadratico medio sui dati di validazione.
    """

    # Definisci lo spazio di ricerca per gli iperparametri
    space = {
        "theta": hp.choice("theta", [0.001, 0.01]),
        "poly": hp.choice("poly", ["constant", "linear"]),
        "corr": hp.choice("corr", ["pow_exp", "abs_exp"]),
    }

    def objective(params):
        """Funzione obiettivo per Hyperopt."""
        theta = params["theta"]
        poly = params["poly"]
        corr = params["corr"]

        try:
            # Addestra il modello Kriging
            model = Kriging(X_train, y_train, theta, corr, poly)
            y_pred = model.predict_values(X_val)
            mse = mean_squared_error(y_val, y_pred)
        except Exception as e:
            print(f"Errore per parametri {params}: {e}")
            return float("inf")  # Restituisce un valore alto in caso di errore
        return mse

    # Ottimizza gli iperparametri con TPE (Tree of Parzen Estimators)
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=rand.suggest,
        max_evals=10,
        trials=trials,
        rstate=np.random.default_rng(42),  # random state per risultati riproducibili
    )

    # Addestra il modello finale con i migliori parametri
    best_theta = [
        0.001,
        0.01,
    ][best_params["theta"]]
    best_poly = ["constant", "linear"][best_params["poly"]]
    best_corr = ["pow_exp", "abs_exp"]
    [best_params["corr"]]
    best_model = Kriging(X_train, y_train, best_theta, best_corr, best_poly)

    # Calcola il miglior MSE sui dati di validazione
    best_model.train()
    best_y_pred = best_model.predict_values(X_val)
    best_mse = mean_squared_error(y_val, best_y_pred)

    return best_model, {"theta": best_theta, "poly": best_poly, "corr": best_corr}, best_mse
