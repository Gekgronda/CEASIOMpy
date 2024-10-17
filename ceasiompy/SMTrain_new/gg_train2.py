import pandas as pd
import numpy as np

def load_data():
    """Carica il file CSV contenente i dati di training."""
    name = input("Insert database name (with .csv extension): ")
    file_path = f"/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/{name}"
    df = pd.read_csv(file_path)
    X = df[["Altitude", "Mach number", "Angle of attack (AoA)", "Angle of sideslip (AoS)"]].values
    y_cl = df["Total CL"].values
    y_cd = df["Total CD"].values
    return X, y_cl, y_cd


def get_model_parameters():
    """Richiedi all'utente i parametri per il modello surrogato."""
    theta_input = input("Insert theta value (single value, e.g., 0.1, 0.01, 0.001, 0.0001): ")
    theta_values = [
        float(theta_input)
    ] * 4 #Occhio qui che moltiplico per 4 perche so che ci sono 4 dimensioni
    corr_value = input("Insert correlation type (e.g., squar_exp, abs_exp, matern32, matern52): ")
    poly_value = input("Insert polynomial type (e.g., constant, linear, quadratic): ")
    return theta_values, corr_value, poly_value


def train_model(X, y_cl, y_cd, theta, corr, poly):
    """Addestra il modello utilizzando i dati forniti."""
    model = fit(X, y_cl, y_cd, theta, corr, poly)
    return model


def evaluate_model(model, X_test, y_cl_test, y_cd_test):
    """Valuta le prestazioni del modello su un set di validazione."""
    eval = evaluate(X_test, y_cl_test, y_cd_test)
    """Plotta i valori predetti vs valori reali"""
    plot = (X_test, y_cl_test, y_cd_test)
    return eval, plot


def get_prediction_data():
    """Richiedi i nuovi dati per cui effettuare previsioni."""
    n_predictions = int(input("How many predictions do you want to make? "))
    new_data = []
    for i in range(n_predictions):
        altitude = float(input(f"Insert Altitude for prediction {i+1}: "))
        mach = float(input(f"Insert Mach number for prediction {i+1}: "))
        aoa = float(input(f"Insert Angle of attack (AoA) for prediction {i+1}: "))
        aos = float(input(f"Insert Angle of sideslip (AoS) for prediction {i+1}: "))
        new_data.append([altitude, mach, aoa, aos])
    return np.array(new_data)


def make_predictions(model, new_data):
    """Effettua le previsioni e visualizza i risultati."""
    cl_predictions, cd_predictions = model.predict(new_data)
    for i, (cl, cd) in enumerate(zip(cl_predictions, cd_predictions)):
        print(f"Prediction {i+1}:")
        print(f"Total CL: {cl}")
        print(f"Total CD: {cd}\n")


# Programma principale
def main():
    X, y_cl, y_cd = load_data()  # Carica i dati
    theta, corr, poly = get_model_parameters()  # Richiedi i parametri del modello
    model = train_model(X, y_cl, y_cd, theta, corr, poly)  # Addestra il modello
    # Valutazione del modello se necessario

    # Effettua previsioni su nuovi dati
    new_data = get_prediction_data()
    make_predictions(model, new_data)


if __name__ == "__main__":
    main()
