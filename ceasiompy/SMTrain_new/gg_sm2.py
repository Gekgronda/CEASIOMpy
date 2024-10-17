import pandas as pd
import numpy as np
import os
from smt.surrogate_models import KRG
from smt.utils.misc import compute_rms_error
from smt.sampling_methods import LHS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import matplotlib.pyplot as plt


# Funzione per eseguire il Latin Hypercube Sampling sugli input
def latin_hypercube_sampling(X, num_samples):
    """Perform Latin Hypercube Sampling on the input data."""
    # Definisci i limiti del campione
    xlimits = np.array([[np.min(X[:, i]), np.max(X[:, i])] for i in range(X.shape[1])])

    # Crea un oggetto LHS
    lhs = LHS(xlimits=xlimits)

    num_samples = int(num_samples)

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
        index = np.argmin(np.linalg.norm(X - sample, axis=1))
        y_cl_lhs.append(y_cl[index])
        y_cd_lhs.append(y_cd[index])

    return np.array(y_cl_lhs), np.array(y_cd_lhs)


# Funzione per addestrare i modelli per CL e CD
def fit_model(X, y_cl, y_cd, theta, corr, poly, test_size=0.3, random_state=42):
    """Train models for CL and CD."""
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

    ndim = X_train.shape[1]

    # Inizializzazione dei modelli per CL e CD
    model_cl = KRG(theta0=theta, corr=corr, poly=poly, print_global=False)
    model_cd = KRG(theta0=theta, corr=corr, poly=poly, print_global=False)

    # Imposta i valori di addestramento
    model_cl.set_training_values(X_train, y_cl_train)
    model_cd.set_training_values(X_train, y_cd_train)

    # Addestra i modelli
    model_cl.train()
    model_cd.train()

    return model_cl, model_cd, X_test, y_cl_test, y_cd_test


# Funzione per fare previsioni sui dati di test
def predict_model(model_cl, model_cd, X_test):
    """Make predictions for CL and CD."""
    cl_pred = model_cl.predict_values(X_test)
    cd_pred = model_cd.predict_values(X_test)
    return cl_pred, cd_pred


# Funzione per valutare i modelli
def evaluate_model(model_cl, model_cd, X_test, y_test_cl, y_test_cd, cl_pred, cd_pred):
    """Evaluate the model and compare predictions with test data."""
    # Calcolo MSE e MAE per CL
    rms_cl = compute_rms_error(model_cl, X_test, y_test_cl)
    mse_cl = mean_squared_error(y_test_cl, cl_pred)
    mae_cl = mean_absolute_error(y_test_cl, cl_pred)

    # Calcolo MSE e MAE per CD
    rms_cd = compute_rms_error(model_cd, X_test, y_test_cd)
    mse_cd = mean_squared_error(y_test_cd, cd_pred)
    mae_cd = mean_absolute_error(y_test_cd, cd_pred)

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
def plot_predictions(y_test_cl, y_test_cd, cl_pred, cd_pred):
    """Plot the predicted vs actual values for CL and CD."""

    # Creazione dei grafici
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Grafico per CL
    axs[0].scatter(y_test_cl, cl_pred, color="blue", alpha=0.5)
    axs[0].plot(
        [y_test_cl.min(), y_test_cl.max()],
        [y_test_cl.min(), y_test_cl.max()],
        "r--",
        lw=2,
    )
    axs[0].set_title("Predicted vs Actual CL")
    axs[0].set_xlabel("Actual CL")
    axs[0].set_ylabel("Predicted CL")
    axs[0].grid()

    # Grafico per CD
    axs[1].scatter(y_test_cd, cd_pred, color="green", alpha=0.5)
    axs[1].plot(
        [y_test_cd.min(), y_test_cd.max()],
        [y_test_cd.min(), y_test_cd.max()],
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


# Funzione per salvare il modello
def save_model(model, filename):
    # Verifica che la directory esista, altrimenti la crea
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved succesfully into {filename}")


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
