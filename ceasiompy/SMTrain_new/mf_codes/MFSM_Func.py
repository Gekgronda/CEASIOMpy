import pandas as pd
import numpy as np
import os
import csv
import re
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler
from smt.surrogate_models import KRG
from smt.applications import EGO, MFK
from smt.sampling_methods import LHS
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from smt.utils.misc import compute_rms_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

# ========= FIRST LATIN HYPERCUBE SAMPLING ==========


def get_user_inputs():
    """
    Ask to enter number of samples and range of domain.
    Otherwise select default parameters
    """
    # Default values
    default_samples = 100
    default_ranges = {
        "altitude": [0, 1000],
        "machNumber": [0.1, 0.5],
        "angleOfAttack": [0, 15],
        "angleOfSideslip": [0, 0],
    }

    # Prompt user for the number of samples
    n_samples_input = input(f"Enter the number of samples (default: {default_samples}): ")
    n_samples = int(n_samples_input) if n_samples_input.strip() else default_samples

    print("Enter the range for each dimension (press Enter to use default values):")
    ranges = {}
    for key, default_range in default_ranges.items():
        min_input = input(f"Minimum {key} (default: {default_range[0]}): ")
        max_input = input(f"Maximum {key} (default: {default_range[1]}): ")

        # Use defaults if input is empty
        min_value = float(min_input) if min_input.strip() else default_range[0]
        max_value = float(max_input) if max_input.strip() else default_range[1]
        ranges[key] = [min_value, max_value]

    return n_samples, ranges


def lh_sampling(
    ranges, n_samples, plot_dim1="altitude", plot_dim2="angleOfAttack", random_state=None
):
    """
    Perform lhsampling of given domain (range and number of samples).
    Apply desired precision (decimal number)
    Visualization of DoE with variables on axes (selected of default)

    """
    # Perform Latin Hypercube Sampling
    sampling = LHS(
        xlimits=np.array(list(ranges.values())), criterion="ese", random_state=random_state
    )
    samples = sampling(n_samples)

    # Map sampled values back to variable names
    sampled_dict = {key: samples[:, idx] for idx, key in enumerate(ranges.keys())}

    # Post-process sampled data to apply desired precision
    for key in sampled_dict:
        if key in ["altitude", "machNumber"]:
            sampled_dict[key] = np.round(sampled_dict[key], 2)  # Two decimal places
        elif key in ["angleOfAttack", "angleOfSideslip"]:
            sampled_dict[key] = np.round(sampled_dict[key]).astype(int)  # Integers

    # Visualization of the results
    print("Visualization of DOE: ")
    xlimits = np.array(list(ranges.values()))
    num = n_samples
    intervals = []
    for i in range(len(xlimits)):
        intervals.append(np.linspace(xlimits[i][0], xlimits[i][1], num + 1))

    # Map plot dimensions to their indices
    dim_keys = list(ranges.keys())
    PLOT1 = dim_keys.index(plot_dim1)
    PLOT2 = dim_keys.index(plot_dim2)

    fig, ax = plt.subplots(1)
    ax.plot(samples[:, PLOT1], samples[:, PLOT2], ".")
    for i in range(len(intervals[0])):
        ax.plot(
            [intervals[PLOT1][i], intervals[PLOT1][i]],
            [intervals[PLOT2][0], intervals[PLOT2][-1]],
            linewidth=0.5,
            color="k",
        )
        ax.plot(
            [intervals[PLOT1][0], intervals[PLOT1][-1]],
            [intervals[PLOT2][i], intervals[PLOT2][i]],
            linewidth=0.5,
            color="k",
        )
    ax.set_xlabel(f"Dimension {PLOT1 + 1}: {plot_dim1}")
    ax.set_ylabel(f"Dimension {PLOT2 + 1}: {plot_dim2}")
    ax.legend(["Initial LHS"], bbox_to_anchor=(1.05, 0.6))

    plt.show()

    return sampled_dict


def save_to_csv(samples, filename):
    """
    Save data to a CSV file.

    Args:
        samples (dict): Data to save, with keys as column names and values as lists/arrays of data.
        filename (str): Path of the output file.
    """
    # Check if all lists has same lenght (maybe is not useful)
    n_rows = len(next(iter(samples.values())))
    for key, values in samples.items():
        if len(values) != n_rows:
            raise ValueError(
                f"All columns must have the same number of rows. Column '{key}' has {len(values)} rows."
            )

    # Write csv
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=samples.keys())
        writer.writeheader()  # Scrive l'intestazione

        # Combine values
        rows = [dict(zip(samples.keys(), row)) for row in zip(*samples.values())]
        writer.writerows(rows)  # Scrive le righe

    print(f"File saved to: {filename}")


# ============= COLLECT DATA FROM AVL SIMULATIONS AND UPDATE  ================


def get_latest_workflow(base_directory):
    # List all items in the base directory
    entries = os.listdir(base_directory)
    # Filter items that start with "Workflow_" and are directories
    workflows = [
        entry
        for entry in entries
        if entry.startswith("Workflow_") and os.path.isdir(os.path.join(base_directory, entry))
    ]
    # Sort directories based on the numeric part at the end (e.g., Workflow_006, Workflow_007, ...)
    workflows.sort(key=lambda x: int(x.split("_")[-1]))
    # Return the full path of the latest Workflow, if any
    return os.path.join(base_directory, workflows[-1]) if workflows else None


def sort_natural(s):
    """Sort string in natural order such as: 'Case01', 'Case10', ecc."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_coefficients_from_AVL(base_path):
    """Extract CLtot, CDtot, Cmtot from ft.txt files inside subdirectories, natual sorted."""
    results = []

    # Natural sort directories
    directories = sorted(
        [d for d in os.listdir(base_path) if d.startswith("Case")], key=sort_natural
    )

    for i, directory in enumerate(directories):
        directory_path = os.path.join(base_path, directory)
        file_path = os.path.join(directory_path, "ft.txt")

        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                content = file.read()

            # Use regex to find values
            matches = {
                "CLtot": re.search(r"CLtot\s*=\s*([-+]?\d*\.\d+|\d+)", content),
                "CDtot": re.search(r"CDtot\s*=\s*([-+]?\d*\.\d+|\d+)", content),
                "Cmtot": re.search(r"Cmtot\s*=\s*([-+]?\d*\.\d+|\d+)", content),
            }

            # Verify if all values are finded
            if all(matches.values()):
                results.append(
                    {
                        "Index": i,
                        "Total CL": float(matches["CLtot"].group(1)),
                        "Total CD": float(matches["CDtot"].group(1)),
                        "Total CM": float(matches["Cmtot"].group(1)),
                    }
                )
            else:
                # print a message if some values are missing
                print("Missing values into file:")
                for key, match in matches.items():
                    if not match:
                        print(f" - {key}")

            # Log for every directory
            print(f"Directory: {directory}, Matches: {matches}")
        else:
            print(f"File not found: {file_path}")

    return results


def append_to_new_csv(data, original_filename):
    """Create a new dataset with extract data and save it with '_TRAIN' extention"""
    if not os.path.isfile(original_filename):
        print(f"Error: {original_filename} is missing.")
        return

    # Read existing CSV filename adn add sequential index
    df = pd.read_csv(original_filename)
    if "Index" not in df.columns:
        df["Index"] = range(len(df))

    # Create a Dataframe with new data and set index
    new_data = pd.DataFrame(data)
    new_filename = os.path.splitext(original_filename)[0] + "_TRAIN.csv"  # Define filename early
    if not new_data.empty:
        new_data.set_index("Index", inplace=True)

        # Unite datasets on equivalent index
        df = df.set_index("Index").join(new_data, how="left", rsuffix="_new")

        # Save new file with '_TRAIN' extention
        df.to_csv(new_filename, index=False, float_format="%.6f")
        print(f"Data saved in: {new_filename} succesfully.")
    else:
        print("Missing data to add to CSV file")

    return new_filename


# ================= FIRST KRIGING MODEL =======================================


def load_and_split_data(path, default_path):
    """
    Load database, split inputs and outputs, handle issues.

    Parameters:
        path (str): Database path provided by the user.
        default_path (str): Database default path.

    Returns:
        dict: Dictionary containing inputs, outputs, and the full dataframe.
    """

    def load_database(file_path, default_path):
        """
        Load the database from the given path, or fallback to default.

        Parameters:
            file_path (str): Path to the file to load.
            default_path (str): Fallback path if the first one fails.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        if not file_path:  # Use default path if no path is provided
            print(f"No path given. Using default path: {default_path}")
            file_path = default_path
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded database from: {file_path}")
            return df
        except Exception as e:
            print(f"Error loading database from {file_path}: {e}")
            print(f"Falling back to default path: {default_path}")
            try:
                return pd.read_csv(default_path)
            except Exception as e_default:
                raise FileNotFoundError(f"Failed to load default database: {e_default}")

    # Load the database
    df = load_database(path, default_path)

    def split_data(df):
        """
        Split dataframe into inputs and outputs.

        Parameters:
            df (pd.DataFrame): The full dataframe.

        Returns:
            tuple: Inputs (X) as a numpy array and outputs (y) as a dictionary.
        """
        input_columns = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]
        output_columns = {
            "CL": "Total CL",
            "CD": "Total CD",
            "CSF": "Total CSF",
            "CMx": "Total CMx",
            "CMy": "Total CMy",
            "CMz": "Total CMz",
        }

        # Check input columns
        missing_inputs = [col for col in input_columns if col not in df.columns]
        if missing_inputs:
            raise KeyError(f"Missing input columns: {missing_inputs}")

        X = df[input_columns].values

        # Check and collect output columns
        y = {}
        for key, col in output_columns.items():
            if col in df.columns:
                y[key] = df[col].values
            else:
                print(f"Warning: Column '{col}' is missing and will be excluded from outputs.")

        return X, y

    # Split the data into inputs and outputs
    X, y = split_data(df)

    return {"dataset": {"X": X, "y": y, "df": df}}


def plot_distributions(df, title):
    """
    Make hystogram for each DataFrame column.

    Parameters:
        df (pd.DataFrame): DataFrame containing data.
        title (str): Plot title.
    """
    df.hist(bins=30, figsize=(15, 10), color="skyblue", edgecolor="black")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def normalize_data(data):
    """
    Normalize inputs and outputs using MinMaxScaler.

    Parameters:
        data (dict): Dictionary containing "X", "y", and "df" keys, as returned by load_and_split_data.

    Returns:
        dict: Dictionary containing normalized X, normalized y, and the original DataFrame.
    """
    df = data["dataset"]["df"]
    X = data["dataset"]["X"]
    y = data["dataset"]["y"]

    # Normalize inputs
    scaler_X = MinMaxScaler()
    X_normalized = scaler_X.fit_transform(X)

    # Normalize outputs
    y_normalized = {}
    for key, values in y.items():
        scaler_y = MinMaxScaler()
        y_normalized[key] = scaler_y.fit_transform(values.reshape(-1, 1)).flatten()

    return {
        "dataset": {
            "df": df,
            "X_normalized": X_normalized,
            "y_normalized": y_normalized,
        }
    }


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


def test_training_data(X, y, test_size=0.3, random_state=42):
    """Divide the data into training, validation, and testing sets"""

    # Validazione degli input
    validate_inputs(X)
    validate_inputs(y)

    # Suddivisione dei dati
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def Kriging(X_train, y_train, theta, corr, poly):
    """Train Kriging model."""
    model = KRG(theta0=theta, corr=corr, poly=poly, print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


# STUDIA LA DOCUMENTAZIONE (hyperparametri)
def MF_Kriging(Xt_hf, yt_hf, Xt_mf, yt_mf, Xt_lf, yt_lf):
    # Build the MFK object with 3 levels
    sm = MFK(theta0=[1e-2], theta_bounds=[1e-06, 100.0], hyper_opt="TNC")
    # low-fidelity dataset names being integers from 0 to level-1
    sm.set_training_values(Xt_lf, yt_lf, name=0)
    sm.set_training_values(Xt_mf, yt_mf, name=1)
    # high-fidelity dataset without name
    sm.set_training_values(Xt_hf, yt_hf)
    # train the model
    sm.train()
    return


def MF_CoKriging(X_hf_train, X_mf, X_lf, y_hf_train, y_mf, y_lf):

    n_start = 100
    opti = "Cobyla"

    # Build the MFCK model with 3 levels
    sm = MFCK(theta0=[1e-2], theta_bounds=[1e-06, 100.0], hyper_opt=opti, n_start=n_start)
    # low-fidelity dataset names being integers from 0 to level-1
    sm.set_training_values(X_lf, y_lf, name=0)
    sm.set_training_values(X_mf, y_mf, name=1)
    # high-fidelity dataset without name
    sm.set_training_values(X_hf_train, y_hf_train)
    # train the model
    sm.train()

    return sm


# Funzione per fare previsioni con Kriging
def predict_model(model, X_test, y_test):
    """Make predictions"""
    y_pred = model.predict_values(X_test)
    var = model.predict_variances(X_test)
    print("Kriging,  err: " + str(compute_rms_error(model, X_test, y_test)))

    # Plot the function and the prediction
    fig = plt.figure()
    plt.plot(y_test, y_test, "-", label="$y_{true}$")
    plt.plot(y_test, y_pred, "r.", label=r"$\hat{y}$")

    plt.xlabel("$y_{true}$")
    plt.ylabel(r"$\hat{y}$")

    plt.legend(loc="upper left")
    plt.title("Kriging model: validation of the prediction model")

    plt.show()

    # Value of theta
    print("theta values", model.optimal_theta)

    return {"y_pred": y_pred, "variance": var}


# Funzione per fare previsioni sui dati di test con mf-kriging
def predict_mf_model(model, X_test):
    """Make prediction"""
    y_pred_hf = sm.predict_values(X_test)
    y_pred_lf = sm._predict_intermediate_values(X_test, 1)
    y_pred_mf = sm._predict_intermediate_values(X_test, 2)
    var_hf = sm.predict_variances(X_test)
    varAll, _ = sm.predict_variances_all_levels(X_test)
    var_lf = varAll[:, 0].reshape(-1, 1)
    var_mf = varAll[:, 1].reshape(-1, 1)

    return y_pred_hf, y_pred_mf, y_pred_lf, var_hf, var_mf, var_lf


# Funzione per fare previsioni sui dati di test con mf-co-kriging
def predict_mfco_model(model, X_test):
    """Make predictions"""
    mn, cn = model.predict_all_levels(X_test)
    pred, cov = model._predict(X_test)

    return mn, cn, pred, cov


def plot_results(y_hf_test, hf_pred, label):

    # Calculate rmse
    rmse = np.sqrt(np.mean((y_hf_test.flatten() - hf_pred.flatten()) ** 2))

    plt.ion()
    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_hf_test, hf_pred, color="blue", alpha=0.5, label="Data points")
    plt.plot(
        [y_hf_test.min(), y_hf_test.max()],
        [y_hf_test.min(), y_hf_test.max()],
        "r--",
        lw=2,
        label=label,
    )

    # Add RMSE to the plot
    plt.text(
        0.05,
        0.95,
        f"RMSE: {rmse:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.title(f"Predicted vs Actual {label}")
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label}")
    plt.grid()
    plt.show()
