import pandas as pd
import numpy as np
from smt.surrogate_models import KRG
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import matplotlib.pyplot as plt


class MultiOutputKriging:
    def __init__(self, file_path, test_size=0.3, random_state=42):
        # Carica il dataset dal file CSV
        self.df = pd.read_csv(file_path)

        self.X = self.df[
            ["Altitude", "Mach number", "Angle of attack (AoA)", "Angle of sideslip (AoS)"]
        ].values
        self.y_cl = self.df["Total CL"].values
        self.y_cd = self.df["Total CD"].values

        self.X_train, X_temp, self.y_cl_train, y_cl_temp = train_test_split(
            self.X, self.y_cl, test_size=test_size, random_state=random_state
        )
        self.X_val, self.X_test, self.y_cl_val, self.y_cl_test = train_test_split(
            X_temp, y_cl_temp, test_size=0.9, random_state=random_state
        )

        self.X_train, X_temp, self.y_cd_train, y_cd_temp = train_test_split(
            self.X, self.y_cd, test_size=test_size, random_state=random_state
        )
        self.X_val, self.X_test, self.y_cd_val, self.y_cd_test = train_test_split(
            X_temp, y_cd_temp, test_size=0.9, random_state=random_state
        )

        self.ndim = self.X_train.shape[1]

        # Modelli iniziali
        self.model_cl = KRG(
            theta0=[1e-2] * self.ndim, corr="matern32", poly="linear", print_global=False
        )
        self.model_cd = KRG(
            theta0=[1e-2] * self.ndim, corr="matern32", poly="linear", print_global=False
        )

    def fit(self, X_train, y_cl_train, y_cd_train, theta=None, corr=None, poly=None):
        """Train models for CL and CD."""
        # Inizializzazione dei modelli se theta è specificato
        if theta is not None:
            self.model_cl = KRG(theta0=theta, print_global=False)
            self.model_cd = KRG(theta0=theta, print_global=False)

        # Se corr o poly sono specificati, ricrea i modelli con i nuovi parametri
        if corr is not None or poly is not None:
            self.model_cl = KRG(
                theta0=self.model_cl.options["theta0"],
                corr=corr if corr is not None else self.model_cl.options.get("corr"),
                poly=poly if poly is not None else self.model_cl.options.get("poly"),
                print_global=False,
            )
            self.model_cd = KRG(
                theta0=self.model_cd.options["theta0"],
                corr=corr if corr is not None else self.model_cd.options.get("corr"),
                poly=poly if poly is not None else self.model_cd.options.get("poly"),
                print_global=False,
            )

        # Imposta i valori di addestramento
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

    # @staticmethod
    # def load(filename):
    #     """Carica il modello salvato da file."""
    #     with open(filename, "rb") as f:
    #         return pickle.load(f)

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

    def plot_predictions(self, X_train, y_train_cl, y_train_cd):
        """Plot the predicted vs actual values for CL and CD."""
        # Predizioni per i dati di addestramento
        cl_pred = self.model_cl.predict_values(X_train)
        cd_pred = self.model_cd.predict_values(X_train)

        # Creazione dei grafici
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Grafico per CL
        axs[0].scatter(y_train_cl, cl_pred, color="blue", alpha=0.5)
        axs[0].plot(
            [y_train_cl.min(), y_train_cl.max()],
            [y_train_cl.min(), y_train_cl.max()],
            "r--",
            lw=2,
        )
        axs[0].set_title("Predicted vs Actual CL")
        axs[0].set_xlabel("Actual CL")
        axs[0].set_ylabel("Predicted CL")
        axs[0].grid()

        # Grafico per CD
        axs[1].scatter(y_train_cd, cd_pred, color="green", alpha=0.5)
        axs[1].plot(
            [y_train_cd.min(), y_train_cd.max()],
            [y_train_cd.min(), y_train_cd.max()],
            "r--",
            lw=2,
        )
        axs[1].set_title("Predicted vs Actual CD")
        axs[1].set_xlabel("Actual CD")
        axs[1].set_ylabel("Predicted CD")
        axs[1].grid()

        plt.tight_layout()
        plt.show()

    def sensitivity_analysis(self, X_base, y_base_cl, y_base_cd, parameter_indices, delta=0.1):
        """
        Esegui un'analisi della sensibilità utilizzando il metodo OAT.

        Args:
            X_base: Input base per le predizioni.
            y_base_cl: Valori di CL corrispondenti.
            y_base_cd: Valori di CD corrispondenti.
            parameter_indices: Indici dei parametri da analizzare.
            delta: Variazione percentuale da applicare ai parametri (default=0.1 per il 10%).

        Returns:
            results: Un dizionario con i risultati delle predizioni e degli errori.
        """
        results = {}

        # Esegui predizioni per i valori base
        base_cl_pred, base_cd_pred = self.predict(X_base)

        for index in parameter_indices:
            # Copia X_base per modificare il parametro specificato
            X_test = np.copy(X_base)

            # Calcola il nuovo valore del parametro
            perturbed_value = X_test[:, index] * (1 + delta)  # Aumenta del delta
            X_test[:, index] = perturbed_value

            # Esegui predizioni con il nuovo valore del parametro
            cl_pred_perturbed, cd_pred_perturbed = self.predict(X_test)

            # Calcola l'errore
            mse_cl = mean_squared_error(y_base_cl, cl_pred_perturbed)
            mse_cd = mean_squared_error(y_base_cd, cd_pred_perturbed)

            results[f"Parameter {index} MSE CL (perturbed)"] = mse_cl
            results[f"Parameter {index} MSE CD (perturbed)"] = mse_cd

            # Stampa i risultati
            print(f"Parameter {index} perturbed (delta={delta*100}%):")
            print(f"MSE CL: {mse_cl}, MSE CD: {mse_cd}\n")

        return results


# Specifica il percorso del file CSV
file_path = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/gg_td2.csv"

# Crea un'istanza della classe
model = MultiOutputKriging(file_path)

# Imposta i valori di theta, corr e poly
theta_values = [1e-2] * model.ndim  # Esempio di valori di theta
corr_value = "matern32"  # Esempio di tipo di correlazione
poly_value = "linear"  # Esempio di grado del polinomio

# Addestra il modello
model.fit(
    model.X_train,
    model.y_cl_train,
    model.y_cd_train,
    theta=theta_values,
    corr=corr_value,
    poly=poly_value,
)

# Valuta il modello
model.evaluate(model.X_val, model.y_cl_val, model.y_cd_val)

# Visualizza le predizioni
model.plot_predictions(model.X, model.y_cl, model.y_cd)

# Esegui l'analisi della sensibilità
parameter_indices = [0, 1, 2, 3]  # Indici dei parametri: Altitude, Mach number, AoA, AoS
model.sensitivity_analysis(model.X_val, model.y_cl_val, model.y_cd_val, parameter_indices)

model_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/"
model_filename = "multi_output_kriging_model.pkl"
model.save(f"{model_directory}{model_filename}")
