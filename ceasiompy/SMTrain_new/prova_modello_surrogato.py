import pandas as pd
import numpy as np
from smt.surrogate_models import KRG
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import matplotlib.pyplot as plt


class MultiOutputKriging:
    def __init__(self, df, test_size=0.3, random_state=10):

        self.X = df[
            ["Altitude", "Mach number", "Angle of attack (AoA)", "Angle of sideslip (AoS)"]
        ].values
        self.y_cl = df["Total CL"].values
        self.y_cd = df["Total CD"].values

        # Divisione dei dati in set di addestramento, validazione e test
        X_train, X_temp, y_cl_train, y_cl_temp = train_test_split(
            self.X, self.y_cl, test_size=test_size, random_state=random_state
        )
        self.X_val, self.X_test, self.y_cl_val, self.y_cl_test = train_test_split(
            X_temp, y_cl_temp, test_size=0.5, random_state=random_state
        )

        X_train, X_temp, y_cd_train, y_cd_temp = train_test_split(
            self.X, self.y_cd, test_size=test_size, random_state=random_state
        )
        self.X_val, self.X_test, self.y_cd_val, self.y_cd_test = train_test_split(
            X_temp, y_cd_temp, test_size=0.5, random_state=random_state
        )

        self.ndim = X_train.shape[1]

        self.model_cl = KRG(theta0=[1e-2] * self.ndim, print_global=False)
        self.model_cd = KRG(theta0=[1e-2] * self.ndim, print_global=False)

    def fit(self, X_train, y_cl_train, y_cd_train, theta=None, corr=None, poly=None):
        """Train models for CL and CD."""
        if theta is not None:
            # Utilizza theta0 al momento della creazione del modello
            self.model_cl = KRG(theta0=theta, print_global=False)
            self.model_cd = KRG(theta0=theta, print_global=False)
        else:
            # Se theta è None, non ricreare i modelli
            self.model_cl.set_training_values(X_train, y_cl_train)
            self.model_cd.set_training_values(X_train, y_cd_train)

        # Se corr è specificato, ricrea i modelli con il nuovo valore di correlazione
        if corr is not None:
            self.model_cl = KRG(
                theta0=self.model_cl.options["theta0"], corr=corr, print_global=False
            )
            self.model_cd = KRG(
                theta0=self.model_cd.options["theta0"], corr=corr, print_global=False
            )
            self.model_cl.set_training_values(X_train, y_cl_train)
            self.model_cd.set_training_values(X_train, y_cd_train)

        # Se poly è specificato, ricrea i modelli con il nuovo valore di polinomio
        if poly is not None:
            self.model_cl = KRG(
                theta0=self.model_cl.options["theta0"], poly=poly, print_global=False
            )
            self.model_cd = KRG(
                theta0=self.model_cd.options["theta0"], poly=poly, print_global=False
            )
            self.model_cl.set_training_values(X_train, y_cl_train)
            self.model_cd.set_training_values(X_train, y_cd_train)

        # Addestra i modelli
        self.model_cl.train()
        self.model_cd.train()

    def predict(self, X_test):
        """Make predictions for CL and CD."""
        cl_pred = self.model_cl.predict_values(X_test)
        cd_pred = self.model_cd.predict_values(X_test)
        return cl_pred, cd_pred

    def save(self, filename):
        """Save the entire model to file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Carica il modello salvato da file."""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def evaluate(self, X_test, y_test_cl, y_test_cd):
        """Evaluate the model and compare predictions with test data."""
        cl_pred, cd_pred = self.predict(X_test)

        # Calcolo MSE e MAE per CL
        mse_cl = mean_squared_error(y_test_cl, cl_pred)
        mae_cl = mean_absolute_error(y_test_cl, cl_pred)

        # Calcolo MSE e MAE per CD
        mse_cd = mean_squared_error(y_test_cd, cd_pred)
        mae_cd = mean_absolute_error(y_test_cd, cd_pred)

        # Print results
        print("Errors for CL:")
        print(f"Mean Squared Error (CL): {mse_cl}")
        print(f"Mean Absolute Error (CL): {mae_cl}")

        print("\nErrors for CD:")
        print(f"Mean Squared Error (CD): {mse_cd}")
        print(f"Mean Absolute Error (CD): {mae_cd}")

        return {"mse_cl": mse_cl, "mae_cl": mae_cl, "mse_cd": mse_cd, "mae_cd": mae_cd}

    def tune_hyperparameters(self, X_train, y_cl_train, y_cd_train, X_val, y_cl_val, y_cd_val):
        """Tunes hyperparameters using validation data."""
        best_score = float("inf")
        best_params = {}
        results = []  # Initialize results here

        # Define hyperparameter grid
        theta_values = [
            [theta] * self.ndim for theta in [0.1, 0.01, 0.001, 0.0001]
        ]  # Lista di liste
        corr_values = ["squar_exp", "abs_exp", "matern32", "matern52"]
        poly_values = ["constant", "linear", "quadratic"]

        # Create parameter grid
        param_grid = ParameterGrid(
            {
                "theta": theta_values,
                "corr": corr_values,
                "poly": poly_values,
            }
        )

        # Evaluate each combination of hyperparameters
        for params in param_grid:
            self.fit(
                X_train,
                y_cl_train,
                y_cd_train,
                params["theta"],
                params["corr"],
                params["poly"],
            )
            evaluation_results = self.evaluate(X_val, y_cl_val, y_cd_val)
            total_error = evaluation_results["mse_cl"] + evaluation_results["mse_cd"]
            results.append((params, total_error))

            # Check for best score
            if total_error < best_score:
                best_score = total_error
                best_params = params

        print(f"Best hyperparameters: {best_params} with total error: {best_score}")

        # Visualize errors
        self.visualize_errors(results)

    def visualize_errors(self, results):
        """Visualizes errors for different hyperparameter combinations."""
        theta_values = [result[0]["theta"][0] for result in results]  # Prendi solo il primo valore
        total_errors = [result[1] for result in results]

        plt.figure(figsize=(10, 6))
        plt.plot(theta_values, total_errors, marker="o")
        plt.xscale("log")
        plt.xlabel("Theta Values")
        plt.ylabel("Total Error (MSE)")
        plt.title("Total Error for Different Theta Values")
        plt.grid()
        plt.show()


file_path = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/training_data.csv"
df = pd.read_csv(file_path)


# Crea e addestra il modello multi-output
multi_output_model = MultiOutputKriging(df)

# Chiama il tuning degli iperparametri
multi_output_model.tune_hyperparameters(
    X_train=multi_output_model.X_val,
    y_cl_train=multi_output_model.y_cl_val,
    y_cd_train=multi_output_model.y_cd_val,
    X_val=multi_output_model.X_val,
    y_cl_val=multi_output_model.y_cl_val,
    y_cd_val=multi_output_model.y_cd_val,
)

# Specify the directory where you want to save the model
# model_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/"
# model_filename = "surrogate_model.pkl"

# # Save the trained model to the specified directory
# multi_output_model.save(f"{model_directory}{model_filename}")


# # Definisci due modelli Kriging, uno per CL e uno per CD
# kriging_cl = KRG(print_global=False)
# kriging_cd = KRG(print_global=False)

# # Addestra il modello per CL
# kriging_cl.set_training_values(X_train, y_train[:, 0])  # CL come primo output
# kriging_cl.train()

# # Addestra il modello per CD
# kriging_cd.set_training_values(X_train, y_train[:, 1])  # CD come secondo output
# kriging_cd.train()

# # Valutazione dell'errore sui dati di test
# cl_test_pred = kriging_cl.predict_values(X_test)
# cd_test_pred = kriging_cd.predict_values(X_test)

# cl_error = np.sqrt(np.mean((cl_test_pred - y_test[:, 0])**2))  # RMSE CL
# cd_error = np.sqrt(np.mean((cd_test_pred - y_test[:, 1])**2))  # RMSE CD

# print(f"RMSE CL: {cl_error}")
# print(f"RMSE CD: {cd_error}")

# cl_pred = kriging_cl.predict_values(X_val)
# cd_pred = kriging_cd.predict_values(X_val)
