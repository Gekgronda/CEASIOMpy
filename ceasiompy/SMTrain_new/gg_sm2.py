import pandas as pd
import numpy as np
import os
import ast
from smt.surrogate_models import KRG, LS, QP, KPLSK, KPLS, IDW, RBF, RMTB, RMTC
from smt.applications import MOE, EGO
from smt.applications.mixed_integer import MixedIntegerContext
from smt.utils.design_space import (
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
)
from smt.utils.misc import compute_rms_error
from smt.sampling_methods import LHS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import joblib
import matplotlib.pyplot as plt


# Funzione per eseguire il Latin Hypercube Sampling sugli input
def latin_hypercube_sampling(X, num_samples):
    """Perform Latin Hypercube Sampling on the input data."""
    # Definisci i limiti del campione
    xlimits = np.array([[np.min(X[:, i]), np.max(X[:, i])] for i in range(X.shape[1])])

    # Crea un oggetto LHS
    lhs = LHS(xlimits=xlimits)

    # Genera il campionamento
    X_lhs = lhs(num_samples)
    return X_lhs


# Funzione per abbinare gli output con i campioni LHS
def match_outputs(X_lhs, X, y_cl, y_cd):
    """Match the outputs for the sampled inputs."""
    y_cl_lhs = []
    y_cd_lhs = []

    for sample in X_lhs:
        # Trova il valore più vicino nel dataset originale
        index = np.argmin(np.linalg.norm(X - sample, axis=1))  # norma euclidea
        y_cl_lhs.append(y_cl[index])
        y_cd_lhs.append(y_cd[index])

    return np.array(y_cl_lhs), np.array(y_cd_lhs)


import numpy as np


# Funzione per aggiungere rumore ai dati di input
def add_noise(X, y_cl, y_cd, noise_level=0.05, num_samples=None):
    """
    Add noise to the input data to create new samples.

    Parameters:
    - X: array di input originale.
    - y_cl, y_cd: array degli output originali.
    - noise_level: livello di rumore (percentuale della deviazione standard).
    - num_samples: numero di campioni da generare (default: uguale al numero di campioni originali).

    Returns:
    - X_noisy: dati di input con rumore.
    - y_cl_noisy, y_cd_noisy: output corrispondenti senza modifiche.
    """
    if num_samples is None:
        num_samples = X.shape[0]  # default: genera lo stesso numero di campioni

    # Calcola la deviazione standard di ciascun input
    std_dev = np.std(X, axis=0)

    # Genera il rumore in base al livello specificato
    noise = np.random.normal(loc=0, scale=noise_level * std_dev, size=(num_samples, X.shape[1]))

    # Crea nuovi campioni aggiungendo il rumore ai dati di input
    X_noisy = X[:num_samples] + noise

    # Mantieni gli stessi valori di output
    y_cl_noisy = y_cl[:num_samples]
    y_cd_noisy = y_cd[:num_samples]

    return X_noisy, y_cl_noisy, y_cd_noisy


def match_noisy_outputs(X_noisy, X, y_cl, y_cd):
    """Match the outputs for noisy inputs."""
    y_cl_noisy = []
    y_cd_noisy = []

    for sample in X_noisy:
        # Trova il valore più vicino nel dataset originale
        index = np.argmin(np.linalg.norm(X - sample, axis=1))
        y_cl_noisy.append(y_cl[index])
        y_cd_noisy.append(y_cd[index])

    return np.array(y_cl_noisy), np.array(y_cd_noisy)


def validate_inputs(data):
    """Convalida i dati di input per assicurarsi che non ci siano valori mancanti o anomali."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Gli input devono essere in formato numpy array.")

    if np.isnan(data).any():
        raise ValueError("Gli input contengono valori mancanti.")

    # aggiungere la possibilita di sostituire i Nan con valori interpolati?

    if np.isinf(data).any():
        raise ValueError("Gli input contengono valori infiniti.")

    # aggiungere un range in cui i valori devono trovarsi?

    return True


def split_data(X, y_cl, y_cd, test_size=0.3, random_state=42):
    """Divide the data into training and testing sets."""

    validate_inputs(X)
    validate_inputs(y_cl)
    validate_inputs(y_cd)

    X_train, X_temp, y_cl_train, y_cl_temp = train_test_split(
        X, y_cl, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_cl_val, y_cl_test = train_test_split(
        X_temp, y_cl_temp, test_size=0.9, random_state=random_state
    )

    _, X_temp, y_cd_train, y_cd_temp = train_test_split(
        X, y_cd, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_cd_val, y_cd_test = train_test_split(
        X_temp, y_cd_temp, test_size=0.9, random_state=random_state
    )

    return (
        X_train,
        X_val,
        X_test,
        y_cl_train,
        y_cl_val,
        y_cl_test,
        y_cd_train,
        y_cd_val,
        y_cd_test,
    )


def Linear(X_train, y_train):
    """Train LS model"""
    model = LS(print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def Quadratic(X_train, y_train):
    """Train QP model"""
    model = QP(print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def Kriging(X_train, y_train, theta, corr, poly):
    """Train Kriging model."""
    model = KRG(theta0=theta, corr=corr, poly=poly, print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def Kriging_PLS(X_train, y_train, theta, corr, poly):
    """Train Kriging model using partial least squares."""
    model = KPLS(theta0=theta, corr=corr, poly=poly, print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def Kriging_PLSK(X_train, y_train, theta, corr, poly):
    """Train KrigingPLS model in two steps."""
    model = KPLSK(theta0=theta, corr=corr, poly=poly, print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def Inverse_DW(X_train, y_train):
    """Train IDW model"""
    model = IDW(print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def Radial_Basis_function(X_train, y_train, d0, poly_degree, reg):
    """Train Radial Basis function model"""
    model = RBF(poly_degree=poly_degree, d0=d0, reg=reg, print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def Regularized_MTB(
    X_train,
    y_train,
    approx_order,
    num_ctrl_pts,
    energy_weight,
    regularization_weight,
    nonlinear_maxiter,
    xlimits,
    smoothness,
    order,
    grad_weight,
    solver_tolerance,
):

    # xlimits = np.array([[0, 1000], [0.1, 0.3], [0, 15], [-2, 2]])
    """Train Regularized minimal-energy tensor-product B-splines model"""
    model = RMTB(
        xlimits=xlimits,
        smoothness=smoothness,
        order=order,
        grad_weight=grad_weight,
        approx_order=approx_order,
        solver_tolerance=solver_tolerance,
        num_ctrl_pts=num_ctrl_pts,
        energy_weight=energy_weight,
        regularization_weight=regularization_weight,
        min_energy=True,
        nonlinear_maxiter=nonlinear_maxiter,
        # print_global=False,
    )
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def Regularized_MTC(
    X_train,
    y_train,
    xlimits,
    approx_order,
    grad_weight,
    num_elements,
    energy_weight,
    regularization_weight,
    nonlinear_maxiter,
):
    """Train Regularized minimal-energy tensor-product Cubic Hermite splines model"""
    model = RMTC(
        xlimits=xlimits,
        approx_order=approx_order,
        grad_weight=grad_weight,
        num_elements=num_elements,
        energy_weight=energy_weight,
        regularization_weight=regularization_weight,
        min_energy=True,
        nonlinear_maxiter=nonlinear_maxiter,
        print_global=False,
    )
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def get_model_params(model_type):
    """Raccoglie i parametri specifici in base al modello selezionato."""
    params = {
        "theta_values": None,
        "corr_value": None,
        "poly_value": None,
        "d0": None,
        "poly_degree": None,
        "reg": None,
        "approx_order": None,
        "num_ctrl_pts": None,
        "energy_weight": None,
        "regularization_weight": None,
        "nonlinear_maxiter": None,
        "num_elements": None,
        "xlimits": None,
        "smoothness": None,
        "order": None,
        "grad_weight": None,
        "solver_tolerance": None,
    }
    if model_type in ["Linear", "Quadratic", "IDW"]:
        params: None

    elif model_type in ["Kriging", "KPLS", "KPLSK"]:
        params["theta_values"] = [
            float(
                input("Insert theta value (e.g., 0.1, 0.01, 0.001, 0.0001) [default=0.01]: ")
                or 0.01
            )
        ]  # THETA DEVE ESSERE UNA LISTA!!!
        params["poly_value"] = (
            input("Insert polynomial type (constant, linear, quadratic) [default=quadratic]: ")
            or "quadratic"
        )

        if model_type == "Kriging":
            params["corr_value"] = (
                input(
                    "Insert correlation type (pow_exp, abs_exp, squar_exp, squar_sin_exp, matern52, matern32) [default=squar_exp]: "
                )
                or "squar_exp"
            )
        elif model_type == "KPLS":
            params["corr_value"] = (
                input(
                    "Insert correlation type (abs_exp, squar_exp, pow_exp) [default=squar_exp]: "
                )
                or "squar_exp"
            )
        elif model_type == "KPLSK":
            params["corr_value"] = "squar_exp"

    elif model_type == "RBF":
        params["d0"] = float(input("Insert d0 value (e.g., 1) [default=1]: ") or 1)
        params["poly_degree"] = int(
            input("Insert polynomial degree (acceptable values = [-1, 0, 1]) [default=-1]: ") or -1
        )
        params["reg"] = float(input("Insert regularization value (e.g., 5) [default=5]: ") or 5)

    elif model_type == "RMTB":
        params["approx_order"] = int(
            input("Insert approximation order (e.g., 2) [default=4]: ") or 4
        )
        params["num_ctrl_pts"] = int(
            input("Insert number of control points (e.g., 15) [default=15]: ") or 15
        )
        params["nonlinear_maxiter"] = int(
            input("Insert nonlinear max iterations (e.g., 100) [default=100]: ") or 100
        )

        params["xlimits"] = np.array(
            input(
                "Insert lower/upper bounds in each dimension as ndarray [nx, 2] (e.g., [[0,1],[0,1]]) [default=None]: "
            )
            or [[0, 1000], [0.1, 0.3], [0, 15], [-2, 2]]
        )
        # if necessario perche essendo un array se uno digita male si sbagascia tutto e crasha
        # if xlimits_input:
        #     try:
        #         # Usa ast.literal_eval invece di eval per un parsing più sicuro
        #         xlimits = ast.literal_eval(xlimits_input)

        #         # Converte in ndarray e controlla che sia di forma [nx, 2]
        #         xlimits_array = np.array(
        #             xlimits, dtype=np.float64
        #         )  # Converti direttamente in float64

        #         if xlimits_array.ndim == 2 and xlimits_array.shape[1] == 2:
        #             params["xlimits"] = xlimits_array
        #             # Converte in ndarray e controlla che sia di forma [nx, 2]
        #         else:
        #             print(
        #                 "Invalid shape for xlimits. Expected [nx, 2]. Setting to empty 2D array."
        #             )
        #             params["xlimits"] = np.array(
        #                 [[0, 1], [0, 1]], dtype=np.float64
        #             )  # Array bidimensionale vuoto come fallback
        #     except (ValueError, SyntaxError):
        #         print("Invalid xlimits input. Setting to empty 2D array.")
        #         params["xlimits"] = np.array(
        #             [[0, 1], [0, 1]], dtype=np.float64
        #         )  # Array bidimensionale vuoto come fallback
        # else:
        #     params["xlimits"] = np.array(
        #         [[0, 1], [0, 1]], dtype=np.float64
        #     )  # Se l'utente non inserisce nulla, assegna un array bidimensionale vuoto

        params["smoothness"] = float(
            input("Insert smoothness parameter (e.g., 1.0) [default=1.0]: ") or 1.0
        )
        params["order"] = int(
            input("Insert B-spline order in each dimension (e.g., 3) [default=3]: ") or 3
        )
        params["grad_weight"] = float(
            input("Insert weight on gradient training data (e.g., 0.5) [default=0.5]: ") or 0.5
        )
        params["solver_tolerance"] = float(
            input(
                "Insert convergence tolerance for the nonlinear solver (e.g., 1e-12) [default=1e-12]: "
            )
            or 1e-12
        )
        params["energy_weight"] = float(
            input("Insert energy weight (e.g., 0.0001) [default=0.0001]: ") or 0.0001
        )
        params["regularization_weight"] = float(
            input("Insert regularization weight (e.g., 1e-14) [default=1e-14]: ") or 1e-14
        )

    elif model_type == "RMTC":
        params["num_elements"] = int(
            input("Insert number of elements (e.g., 4) [default=4]: ") or 4
        )
        params["nonlinear_maxiter"] = int(
            input("Insert nonlinear max iterations (e.g., 10) [default=10]: ") or 10
        )

        params["xlimits"] = np.array(
            input(
                "Insert lower/upper bounds in each dimension as ndarray [nx, 2] (e.g., [[0,1],[0,1]]) [default=None]: "
            )
            or [[0, 1000], [0.1, 0.3], [0, 15], [-2, 2]]
        )
        params["approx_order"] = int(
            input("Insert exponent in the approximation term (e.g., 4) [default=4]: ") or 4
        )
        params["grad_weight"] = float(
            input("Insert weight on gradient training data (e.g., 0.5) [default=0.5]: ") or 0.5
        )
        params["energy_weight"] = float(
            input("Insert energy weight (e.g., 0.0001) [default=0.0001]: ") or 0.0001
        )
        params["regularization_weight"] = float(
            input("Insert regularization weight (e.g., 1e-14) [default=1e-14]: ") or 1e-14
        )

    return params


# Funzione per il training una volta scelto il modello
def train_surrogate_model(X_train, y_train, model_type, params):
    """
    Adatta un modello surrogato per i dati di input.

    Args:
        X_train (array): dati di input per l'addestramento.
        y_train (array): dati di output per l'addestramento.
        model_type (str): tipo di modello (es. Kriging, KPLS, RBF).
        params (dict): dizionario di iperparametri specifici per il modello.

    Returns:
        model: modello addestrato.
    """
    if model_type == "Linear":
        model = Linear(X_train, y_train)

    elif model_type == "Quadratic":
        model = Quadratic(X_train, y_train)

    elif model_type == "IDW":
        model = Inverse_DW(X_train, y_train)

    elif model_type == "Kriging":
        model = Kriging(
            X_train,
            y_train,
            theta=params["theta_values"],
            corr=params["corr_value"],
            poly=params["poly_value"],
        )

    elif model_type == "KPLS":
        model = Kriging_PLS(
            X_train,
            y_train,
            theta=params["theta_values"],
            corr=params["corr_value"],
            poly=params["poly_value"],
        )

    elif model_type == "KPLSK":
        model = Kriging_PLS(
            X_train,
            y_train,
            theta=params["theta_values"],
            corr=params["corr_value"],
            poly=params["poly_value"],
        )

    elif model_type == "RBF":
        model = Radial_Basis_function(
            X_train, y_train, d0=params["d0"], poly_degree=params["poly_degree"], reg=params["reg"]
        )

    elif model_type == "RMTB":
        model = Regularized_MTB(
            X_train,
            y_train,
            approx_order=params["approx_order"],
            num_ctrl_pts=params["num_ctrl_pts"],
            energy_weight=params["energy_weight"],
            regularization_weight=params["regularization_weight"],
            nonlinear_maxiter=params["nonlinear_maxiter"],
            xlimits=params["xlimits"],
            smoothness=params["smoothness"],
            order=params["order"],
            grad_weight=params["grad_weight"],
            solver_tolerance=params["solver_tolerance"],
        )

    elif model_type == "RMTC":
        model = Regularized_MTC(
            X_train,
            y_train,
            num_elements=params["num_elements"],
            energy_weight=params["energy_weight"],
            regularization_weight=params["regularization_weight"],
            nonlinear_maxiter=params["nonlinear_maxiter"],
            xlimits=params["xlimits"],
            approx_order=params["approx_order"],
            grad_weight=params["grad_weight"],
        )
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")

    return model


# Funzione per fare previsioni sui dati di test
def predict_model(model, X_test):
    """Make predictions for CL and CD."""
    y_pred = model.predict_values(X_test)

    return y_pred


# Funzione per valutare i modelli
def evaluate_model(model_cl, model_cd, X_test, y_cl_test, y_cd_test, cl_pred, cd_pred):
    """Evaluate the model and compare predictions with test data."""
    # Calcolo MSE e MAE per CL
    rms_cl = compute_rms_error(model_cl, X_test, y_cl_test)
    mse_cl = mean_squared_error(y_cl_test, cl_pred)
    mae_cl = mean_absolute_error(y_cl_test, cl_pred)

    # Calcolo MSE e MAE per CD
    rms_cd = compute_rms_error(model_cd, X_test, y_cd_test)
    mse_cd = mean_squared_error(y_cd_test, cd_pred)
    mae_cd = mean_absolute_error(y_cd_test, cd_pred)

    # Print results
    print("Errors for CL:")
    print(f"Root Mean Squared Error (CL): {rms_cl}")
    print(f"Mean Squared Error (CL): {mse_cl}")
    print(f"Mean Absolute Error (CL): {mae_cl}")

    print("\nErrors for CD:")
    print(f"Root Mean Squared Error (CD): {rms_cd}")
    print(f"Mean Squared Error (CD): {mse_cd}")
    print(f"Mean Absolute Error (CD): {mae_cd}")

    return {
        "rms_cl": rms_cl,
        "mse_cl": mse_cl,
        "mae_cl": mae_cl,
        "mse_cd": mse_cd,
        "mae_cd": mae_cd,
    }


# Funzione per plottare i risultati
def plot_predictions(y_cl_test, y_cd_test, cl_pred, cd_pred):
    """Plot the predicted vs actual values for CL and CD."""

    # Creazione dei grafici
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Grafico per CL
    axs[0].scatter(y_cl_test, cl_pred, color="blue", alpha=0.5)
    axs[0].plot(
        [y_cl_test.min(), y_cl_test.max()],
        [y_cl_test.min(), y_cl_test.max()],
        "r--",
        lw=2,
    )
    axs[0].set_title("Predicted vs Actual CL")
    axs[0].set_xlabel("Actual CL")
    axs[0].set_ylabel("Predicted CL")
    axs[0].grid()

    # Grafico per CD
    axs[1].scatter(y_cd_test, cd_pred, color="green", alpha=0.5)
    axs[1].plot(
        [y_cd_test.min(), y_cd_test.max()],
        [y_cd_test.min(), y_cd_test.max()],
        "r--",
        lw=2,
    )
    axs[1].set_title("Predicted vs Actual CD")
    axs[1].set_xlabel("Actual CD")
    axs[1].set_ylabel("Predicted CD")
    axs[1].grid()

    plt.tight_layout()
    plt.show()


def combine_models(model_cl, model_cd):
    """Combine CL and CD models into a single dictionary."""
    return {"model_cl": model_cl, "model_cd": model_cd}


# Funzione per salvare il modello, con pickle (+ gestione errori)
def save_model(model, model_directory, base_model_name, model_extension):
    """
    Salva il modello in un file unico nella directory specificata.

    Parametri:
    - model: il modello da salvare.
    - model_directory: directory in cui salvare il modello.
    - base_model_name: nome base per il modello (default "surrogate_model").
    - model_extension: estensione del file (default ".pkl").
    """
    # Verifica che la directory esista, altrimenti la crea
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Crea il percorso del modello
    model_path = os.path.join(model_directory, base_model_name + model_extension)

    # Trova un nome di file che non esista già
    counter = 1
    while os.path.exists(model_path):
        model_name = f"{base_model_name}_{counter}{model_extension}"
        model_path = os.path.join(model_directory, model_name)
        counter += 1

    # Salva il modello
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Model saved to {model_path}")


# Bayesian optimizarion of the model, for the moment only for kriging and derivatives
# def optimize_hyperparameters_ego(model_type, X_train, y_train, X_val, y_val):
#     """Ottimizza i parametri di un modello tramite ottimizzazione bayesiana usando SMT EGO."""

#     # Mapping of the categorical values of Kriging, KPLS e KPLSK
#     corr_options = {
#         "Kriging": ["pow_exp", "abs_exp", "squar_exp", "squar_sin_exp", "matern52", "matern32"],
#         "KPLS": ["pow_exp", "abs_exp", "squar_exp"],
#         "KPLSK": ["squar_exp"],
#     }
#     poly_options = ["constant", "linear", "quadratic"]
#     theta_values = [0.0001, 0.001, 0.01, 0.1, 1]

#     print(corr_options[model_type])

#     # Set up of the search space for each model type (da implementare anche per altri modelli)
#     if model_type in corr_options:
#         design_space = DesignSpace(
#             [
#                 IntegerVariable(0, len(theta_values) - 1),  # indice di theta
#                 CategoricalVariable(corr_options[model_type]),  # kernel di correlazione
#                 CategoricalVariable(poly_options),  # tipo di polinomio
#             ],
#             random_state=42,
#         )
#     else:
#         raise ValueError("Model type not supported.")
#     # elif model_type == "RBF":
#     #     bounds = np.array([[0.1, 10], [-1, 1], [0.1, 10]])
#     # elif model_type == "RMTB":
#     #     bounds = np.array([[1, 5], [5, 30], [0.0001, 0.1], [1e-14, 1e-10], [10, 200]])
#     # elif model_type == "RMTC":
#     #     bounds = np.array([[1, 10], [1, 10], [0.0001, 0.1], [1e-14, 1e-10], [10, 200]])
#     # else:
#     #     raise ValueError("Model type not supported.")

#     # Objective function for Bayesian optimization (for KRIGING)
#     def objective_function(x):
#         # Assicuriamoci che x sia un array monodimensionale per evitare errori
#         # Estrai i parametri ottimali da x
#         theta_value = theta_values[int(x[0])]
#         corr_value = str(x[1])
#         poly_value = str(x[2])

#         # Definizione dei parametri usando i valori di x
#         params = {
#             "theta_values": [theta_value],
#             "corr_value": corr_value,
#             "poly_value": poly_value,
#         }

#         # Addestra e valuta il modello su X_train e y_train
#         print("Training with params:", params)
#         # model = train_surrogate_model(X_train, y_train, model_type, params)
#         model = Kriging(X_train, y_train, theta_value, corr_value, poly_value)
#         y_pred = predict_model(model, X_val)
#         error = mean_squared_error(y_val, y_pred)

#         return error

#     # Imposta il contesto per le variabili miste
#     mixint = MixedIntegerContext(design_space)
#     sampling = mixint.build_sampling_method(random_state=42)
#     xdoe = sampling(5)  # DOE con 5 punti casuali
#     ydoe = np.array([objective_function(x) for x in xdoe])

#     sm = KRG(
#         design_space=design_space,
#         categorical_kernel=MixIntKernelType.GOWER,  # kernel per variabili categoriche
#         hyper_opt="Cobyla",
#         print_global=False,
#     )

#     ego = EGO(
#         n_iter=5,  # da cambiare
#         criterion="EI",
#         xdoe=xdoe,
#         ydoe=ydoe,
#         surrogate=sm,
#         qEI="KBRand",
#         random_state=42,
#     )

#     x_opt, y_opt, _, _, y_data = ego.optimize(fun=objective_function)

#     # Salva i migliori parametri trovati
#     best_params = {
#         "theta": theta_values[int(x_opt[0])],
#         "corr_value": x_opt[1],
#         "poly_value": x_opt[2],
#     }
#     print("Best hyperparameters found:", best_params)
#     return best_params


# MoE
def compare_models(X_test, y_test, X_train, y_train):

    moe = MOE(n_clusters=5, xtest=X_test, ytest=y_test)
    print("MOE enabled experts: ", moe.enabled_experts)

    moe.set_training_values(X_train, y_train)
    moe.train()

    print("MOE + 1 cluster, error: " + str(compute_rms_error(moe, X_test, y_test)))

    return moe
    # QUESTO IF SUGGERISCE LA MIGLIOR COMBINAZIONE DI IPERPARAMETRI
    if (
        (moe._experts[0].name == "Kriging")
        or (moe._experts[0].name == "KPLS")
        or (moe._experts[0].name == "KPLSK")
    ):
        print("Correlation parameter of this model:", moe._experts[0].options["corr"])
        print("Regression parameter of this model:", moe._experts[0].options["poly"])

    # Prediction of the validation points
    y = moe.predict_values(X_test)


# # Specifica il percorso del file CSV
# file_path = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/training_data.csv"

# # Crea un'istanza della classe
# model = MultiOutputKriging(file_path)

# # Imposta i valori di theta, corr e poly
# theta_values = [1e-2] * model.ndim  # Esempio di valori di theta
# corr_value = "matern32"  # Esempio di tipo di correlazione
# poly_value = "linear"  # Esempio di grado del polinomio

# # Addestra il modello
# model.fit(model.X, model.y_cl, model.y_cd, theta=theta_values, corr=corr_value, poly=poly_value)

# # Valuta il modello
# model.evaluate(model.X_val, model.y_cl_val, model.y_cd_val)

# # Visualizza le predizioni
# model.plot_predictions(model.X, model.y_cl, model.y_cd)


# model_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/"
# model_filename = "multi_output_kriging_model.pkl"
# model.save(f"{model_directory}{model_filename}")
