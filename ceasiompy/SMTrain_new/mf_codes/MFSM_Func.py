import pandas as pd
import numpy as np
import os
import shutil
import csv
import re
import sys
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
from scipy.interpolate import griddata
import pickle
from tixi3.tixi3wrapper import Tixi3
import subprocess
from sklearn.preprocessing import MinMaxScaler
from smt.utils.misc import compute_rms_error

from CPACS_Func import (
    add_new_aeromap,
    avl_update,
    change_reference_value,
    SU2_update,
)


def choose_fidelity_workflow():
    """
    Allows the user to select the number of fidelity levels for the surrogate model
    and define the workflow for each fidelity level.
    """

    # TO DO
    # - aggiungere controllo input
    # - controllare logica

    # Input validation for fidelity level
    while True:
        fidelity_level = (
            input(
                "Choose how many fidelity levels to train the surrogate model (1, 2, 3) [default: 3]: "
            )
            or "3"
        )
        if fidelity_level in {"1", "2", "3"}:
            fidelity_level = int(fidelity_level)
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    print("Now select your workflow:")
    print(
        "You can choose to perform Low Fidelity simulation with PyAvl Module, or High Fidelity simulations with SU2 Module"
    )
    print(
        "SU2 module can perform both Euleran or RANS simulations, you can choose Euleran simulations, RANS or both"
    )
    print(
        "In this module results from Euleran simulations are called Medium Fidelity (MF) and High Fidelity (HF) from RANS "
    )

    print("\nRecap workflow options:")
    print("  - LF: Low Fidelity (PyAVL)")
    print("  - MF: Medium Fidelity (SU2 Euleran)")
    print("  - HF: High Fidelity (SU2 RANS)")
    print("")

    # Fidelity workflow selection based on fidelity level
    if fidelity_level == 1:
        print("One level of fidelity selected.")
        while True:
            fidelity_workflow = (
                input("Select one fidelity level (LF, MF, or HF): ").strip().upper()
            )
            if fidelity_workflow in {"LF", "MF", "HF"}:
                break
            else:
                print("Invalid choice. Please choose LF, MF, or HF.")
    elif fidelity_level == 2:
        print("Two levels of fidelity selected.")
        while True:
            fidelity_workflow = (
                input(
                    "Select two fidelity levels (e.g., 'LF and MF', 'LF and HF', or 'MF and HF'): "
                )
                .strip()
                .upper()
            )
            if fidelity_workflow in {"LF AND MF", "LF AND HF", "MF AND HF"}:
                break
            else:
                print("Invalid choice. Please choose 'LF and MF', 'LF and HF', or 'MF and HF'.")
    else:
        print("All levels of fidelity selected.")
        fidelity_workflow = "LF, MF, and HF"

    print(f"\nFidelity level: {fidelity_level}")
    print(f"Selected workflow: {fidelity_workflow}")

    return fidelity_level, fidelity_workflow


# ========= FIRST LATIN HYPERCUBE SAMPLING ==========


def find_and_save_file(file_name, search_path, destination_folder):
    """
    Search for a file in the specified path and save it to the destination folder.

    Parameters:
        file_name (str): Name of the file to search for.
        search_path (str): Path to search in.
        destination_folder (str): Folder to copy the file to.
    """
    for root, dirs, files in os.walk(search_path):
        if file_name in files:
            source_path = os.path.join(root, file_name)
            os.makedirs(destination_folder, exist_ok=True)  # Create folder if not exists
            shutil.copy(source_path, destination_folder)
            print(f"File '{file_name}' copied to '{destination_folder}'.")
            return

    print(f"File '{file_name}' not found in '{search_path}'.")


def get_user_inputs():
    """
    Ask to enter number of samples and range of domain.
    Otherwise select default parameters
    """
    # Default values
    default_samples = 30
    default_ranges = {
        "altitude": [10000, 10000],
        "machNumber": [0.1, 0.7],
        "angleOfAttack": [-2, 10],
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


def lh_sampling(ranges, n_samples, random_state=None):
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
            sampled_dict[key] = np.round(sampled_dict[key], 1)  # One decimal places
        elif key in ["angleOfAttack", "angleOfSideslip"]:
            sampled_dict[key] = np.round(sampled_dict[key]).astype(int)  # Integers

    # Convert post-processed dictionary back to array for plotting
    processed_samples = np.column_stack([sampled_dict[key] for key in ranges.keys()])

    return sampled_dict, processed_samples


def plot_doe(
    processed_samples,
    ranges,
    n_samples,
    plot_dim1="altitude",
    plot_dim2="angleOfAttack",
    highlight_points=None,
):

    # Visualization of the results
    print("Visualization of DOE: ")
    xlimits = np.array(list(ranges.values()))
    intervals = []
    for i in range(len(xlimits)):
        intervals.append(np.linspace(xlimits[i][0], xlimits[i][1], n_samples + 1))

    # Map plot dimensions to their indices
    dim_keys = list(ranges.keys())
    PLOT1 = dim_keys.index(plot_dim1)
    PLOT2 = dim_keys.index(plot_dim2)

    fig, ax = plt.subplots(1)

    ax.plot(processed_samples[:, PLOT1], processed_samples[:, PLOT2], ".")

    # Highlight specific points if provided
    if highlight_points is not None:
        ax.plot(
            highlight_points[:, PLOT1],
            highlight_points[:, PLOT2],
            "ro",
            label="High Variance Points",
        )

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


def doe_workflow(default_doe_path, directory_path, output_filename):
    """
    Workflow to handle Design of Experiment (DoE).

    Parameters:
        default_doe_path (str): Path to the default DoE file.
        directory_path (str): Directory to save new DoE files.

    Returns:
        tuple: A DataFrame containing the DoE, ranges, and processed samples.
    """
    print("Do you want to select manually DOE points (YES, NO)? [default: NO]")
    print("If YES: select manually number of points and ranges.")
    print("If NO: insert a file.csv with DoE.")
    define_doe_or_default = input(": ") or "NO"

    if define_doe_or_default.upper() == "YES":
        # Insert a domain of interest
        n_samples, ranges = get_user_inputs()

        # Latin Hypercube Sampling (DoE)
        print("Sampling...")
        samples, processed_samples = lh_sampling(ranges, n_samples)

        # Save DoE to CSV
        full_path = os.path.join(directory_path, output_filename)
        save_to_csv(samples, full_path)
        print(f"New DoE saved in {full_path}")

    # AGGIUNGERE VERIFICHE

    else:
        doe_path = input("Insert file.csv path: ")
        if not doe_path:  # Use default path if no path is provided
            print(f"No path given. Using default path: {default_doe_path}")
            doe_path = default_doe_path
        try:
            # Load the DoE from the file
            df = pd.read_csv(doe_path)
            print(f"Loaded database from: {doe_path}")

            # Convert DataFrame to dictionary
            samples = df.to_dict(orient="list")

            # Compute ranges and processed samples
            ranges = {col: [df[col].min(), df[col].max()] for col in df.columns}
            processed_samples = df.to_numpy()
            n_samples = len(df)
            full_path = doe_path

        except Exception as e:
            print(f"Error loading database from {doe_path}: {e}")
            print(f"Falling back to default path: {default_doe_path}")
            try:
                df = pd.read_csv(default_doe_path)
                print(f"Loaded database from default path: {default_doe_path}")

                # Convert DataFrame to dictionary
                samples = df.to_dict(orient="list")

                # Compute ranges and processed samples
                ranges = {col: [df[col].min(), df[col].max()] for col in df.columns}
                processed_samples = df.to_numpy()
                n_samples = len(df)
                full_path = default_doe_path

            except Exception as e_default:
                raise FileNotFoundError(f"Failed to load default database: {e_default}")

    plot_doe(processed_samples, ranges, n_samples, "angleOfAttack", "machNumber")

    return samples, ranges, processed_samples, n_samples, full_path


# ============= AVL WORKFLOW  ================


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


def launch_avl_simulations(
    default_kriging_dataset_path, directory_path, input_cpacs_path, full_path
):

    kriging_dataset_path = None  # Initialize with None

    # LAUNCH AVL COMMAND
    print("CPACS updated, running PyAVL Module in CEASIOMpy...")
    command = (
        f"cd {os.path.abspath(directory_path)} && "
        f"ceasiompy_run -m {os.path.abspath(input_cpacs_path)} PyAVL"
    )

    try:
        # Run the command with subprocess.run()
        print("PyAVL simulation started. Press Ctrl+C to interrupt manually.")
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=sys.stdout,  # Forward standard output to the terminal
            stderr=sys.stderr,  # Forward standard error to the terminal
        )

        # Check if the process completed successfully
        if result.returncode == 0:
            print("Simulations completed successfully!")
        else:
            print("An error occurred during the simulations!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during the simulation: {e}")
        # Use the default dataset if there was an error
        kriging_dataset_path = default_kriging_dataset_path
        result = None  # Ensure result is defined
    except KeyboardInterrupt:
        print("\nSimulation manually interrupted.")
        # Manually set to default dataset if interrupted
        kriging_dataset_path = default_kriging_dataset_path
        result = None  # Ensure result is defined

    # PROCESS SIMULATION RESULTS OR USE DEFAULT DATASET
    if result and result.returncode == 0:
        # If the process completed, analyze the results
        latest_workflow_path = get_latest_workflow(directory_path)
        if latest_workflow_path:
            results_path = os.path.join(latest_workflow_path, "Results", "PyAVL")
            print("Latest Workflow:", latest_workflow_path)
            print("Results Path:", results_path)

            if os.path.isdir(results_path):
                data1 = extract_coefficients_from_AVL(results_path)
                print("Coefficents got from AVL simulations:")
                print(data1)
                kriging_dataset_path = append_to_new_csv(data1, full_path)
            else:
                print(f"Error: The directory {results_path} does not exist.")
        else:
            print("No workflow found.")
    else:
        # Use the default dataset if the process was interrupted
        print(f"Using the default dataset: {default_kriging_dataset_path}")
        kriging_dataset_path = default_kriging_dataset_path

    return kriging_dataset_path


def avl_workflow(
    input_cpacs_path,
    directory_path,
    default_kriging_dataset_path,
    full_path,
    samples,
    aeromap_uid,
    aeromap_name,
    avl_parameters,
):

    print("Do you want to proceed with AVL simulations? [default: NO]")
    print("If YES: proceed with AVL configuration")
    print("If NO: insert file.csv to train first kriging")
    avl_yes_or_not = input(": ") or "NO"

    if avl_yes_or_not.upper() == "YES":
        print("Updating aeromap and reference value for AVL simulations")
        input("Press ENTER to continue....")

        # Aeromap updating on CPACS
        tixi = Tixi3()
        tixi.open(input_cpacs_path)

        try:
            # add the new aeroMap
            add_new_aeromap(tixi, samples, aeromap_uid, aeromap_name)
            # change_reference_value(tixi, reference_values)
            # save the updated CPACS file
            tixi.save(input_cpacs_path)
            print("New aeroMap added successfully!")
        except Exception as e:
            print(f"Error adding aeroMap: {e}")
        finally:
            tixi.close()

        # AVL updating on CPACS
        print("Updating parameters for AVL simulations")
        input("Press ENTER to continue....")

        tixi = Tixi3()
        tixi.open(input_cpacs_path)

        try:
            # avl update
            avl_update(tixi, aeromap_name, avl_parameters)
            # save the updated CPACS file
            tixi.save(input_cpacs_path)
            print("AVL parameters updated successfully!")
        except Exception as e:
            print(f"Error updating parameters: {e}")
        finally:
            tixi.close()

        input("Press ENTER to continue....")

        # Obtain path of train dataset
        kriging_dataset_path = launch_avl_simulations(
            default_kriging_dataset_path, directory_path, input_cpacs_path, full_path
        )

    else:

        kriging_dataset_path = input("Insert file.csv path: ")

        if not kriging_dataset_path:  # Use default path if no path is provided
            print(f"No path given. Using default path: {default_kriging_dataset_path}")
            kriging_dataset_path = default_kriging_dataset_path

    return kriging_dataset_path


# ============== non in ordine: COLLECT DATA FROM SU2 ========


def extract_coefficients_from_SU2(base_path):
    """Estrae CL, CD, CSF, CMx, CMy, e CMz dai file .dat nelle sottocartelle, in ordine naturale."""
    results = []

    # Ordina le directory in ordine naturale
    directories = sorted(
        [d for d in os.listdir(base_path) if d.startswith("Case")], key=sort_natural
    )

    for i, directory in enumerate(directories):
        directory_path = os.path.join(base_path, directory)
        file_path = os.path.join(directory_path, "forces_breakdown.dat")

        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                content = file.read()

            # Cerca i valori dei coefficienti
            matches = {
                "Total CL": re.search(r"Total CL:\s+([-+]?\d*\.\d+|\d+)", content),
                "Total CD": re.search(r"Total CD:\s+([-+]?\d*\.\d+|\d+)", content),
                "Total CSF": re.search(r"Total CSF:\s+([-+]?\d*\.\d+|\d+)", content),
                "Total CMx": re.search(r"Total CMx:\s+([-+]?\d*\.\d+|\d+)", content),
                "Total CMy": re.search(r"Total CMy:\s+([-+]?\d*\.\d+|\d+)", content),
                "Total CMz": re.search(r"Total CMz:\s+([-+]?\d*\.\d+|\d+)", content),
            }

            # Aggiunge i risultati se tutti i valori sono trovati
            if all(matches.values()):
                results.append(
                    {
                        "Index": i,
                        "Total CL": float(matches["Total CL"].group(1)),
                        "Total CD": float(matches["Total CD"].group(1)),
                        "Total CSF": float(matches["Total CSF"].group(1)),
                        "Total CMx": float(matches["Total CMx"].group(1)),
                        "Total CMy": float(matches["Total CMy"].group(1)),
                        "Total CMz": float(matches["Total CMz"].group(1)),
                    }
                )
            else:
                print(f"Valori mancanti in {file_path}")

            # Log per ogni directory elaborata
            print(f"Directory: {directory}, Matches: {matches}")
        else:
            print(f"File non trovato: {file_path}")

    return results


# ================= FIRST KRIGING MODEL =======================================


def load_and_split_data(path):
    """
    Load database from the given path, split inputs and outputs, and handle errors.
    Handles cases where 'Total CMy' is labeled as 'Total CM'.

    Parameters:
        path (str): Path to the database file.

    Returns:
        dict: Dictionary containing inputs, outputs, and the full dataframe.
    """
    try:
        # Load the database
        df = pd.read_csv(path)
        print(f"Loaded database from: {path}")
    except Exception as e:
        raise FileNotFoundError(f"Error loading database from {path}: {e}")

    # Handle special case: 'Total CM' as 'Total CMy'
    if "Total CM" in df.columns and "Total CMy" not in df.columns:
        df.rename(columns={"Total CM": "Total CMy"}, inplace=True)
        print("Notice: Column 'Total CM' found and renamed to 'Total CMy'.")

    # Define input and output columns
    input_columns = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]
    output_columns = {
        "CL": "Total CL",
        "CD": "Total CD",
        "CSF": "Total CSF",
        "CMx": "Total CMx",
        "CMy": "Total CMy",
        "CMz": "Total CMz",
    }

    # Check for missing input columns
    missing_inputs = [col for col in input_columns if col not in df.columns]
    if missing_inputs:
        raise KeyError(f"Missing input columns: {missing_inputs}")

    # Extract inputs
    X = df[input_columns].values

    # Extract outputs
    y = {}
    for key, col in output_columns.items():
        if col in df.columns:
            y[key] = df[col].values
        else:
            print(f"Warning: Column '{col}' is missing and will be excluded from outputs.")

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

    # Normalize outputs (cycle bec every output has diff scale)
    y_normalized = {}
    scalers_y = {}
    for key, values in y.items():
        scaler_y = MinMaxScaler()
        y_normalized[key] = scaler_y.fit_transform(values.reshape(-1, 1)).flatten()
        scalers_y[key] = scaler_y  # in modo da de-scalare i diversi output

    return {
        "dataset": {
            "df": df,
            "X_normalized": X_normalized,
            "y_normalized": y_normalized,
        },
        "scalers": {
            "scaler_X": scaler_X,
            "scaler_y": scalers_y,  # Dictionary of output scalers
        },
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
def MF_Kriging(Xt_lf, yt_lf, Xt_mf, yt_mf, theta, corr, poly, Xt_hf=None, yt_hf=None):
    """
    Create a multi-fidelity Kriging model with 2 or 3 levels.

    Args:
        Xt_lf (array): Input for low-fidelity data.
        yt_lf (array): Output for low-fidelity data.
        Xt_mf (array): Input for mid-fidelity data.
        yt_mf (array): Output for mid-fidelity data.
        Xt_hf (array, optional): Input for high-fidelity data. Defaults to None.
        yt_hf (array, optional): Output for high-fidelity data. Defaults to None.

    Returns:
        MFK: Trained Kriging model.
    """
    # Create the Kriging model
    model = MFK(theta0=theta, theta_bounds=[1e-06, 100.0], corr=corr, poly=poly, hyper_opt="TNC")

    # Set training values for the fidelity levels
    model.set_training_values(Xt_lf, yt_lf, name=0)  # Low-fidelity

    # Add high-fidelity data if provided
    if Xt_hf is None and yt_hf is None:
        model.set_training_values(Xt_mf, yt_mf)  # Mid-fidelity
    else:
        model.set_training_values(Xt_mf, yt_mf, name=1)  # Mid-fidelity
        model.set_training_values(Xt_hf, yt_hf)  # High-fidelity without a name

    # Train the model
    model.train()

    return model


def MF_CoKriging(X_lf, y_lf, X_mf, y_mf, X_hf_train=None, y_hf_train=None):
    """
    Create a multi-fidelity Co-Kriging model with 2 or 3 levels.

    Args:
        X_lf (array): Input for low-fidelity data.
        y_lf (array): Output for low-fidelity data.
        X_mf (array): Input for mid-fidelity data.
        y_mf (array): Output for mid-fidelity data.
        X_hf_train (array, optional): Input for high-fidelity data. Defaults to None.
        y_hf_train (array, optional): Output for high-fidelity data. Defaults to None.

    Returns:
        MFCK: Trained Co-Kriging model.
    """

    n_start = 100
    opti = "Cobyla"

    # Create the Co-Kriging model
    model = MFCK(theta0=[1e-2], theta_bounds=[1e-06, 100.0], hyper_opt=opti, n_start=n_start)

    # Set training values for the fidelity levels
    model.set_training_values(X_lf, y_lf, name=0)  # Low-fidelity
    model.set_training_values(X_mf, y_mf, name=1)  # Mid-fidelity

    # Add high-fidelity data if provided
    if X_hf_train is not None and y_hf_train is not None:
        model.set_training_values(X_hf_train, y_hf_train)  # High-fidelity without a name

    # Train the model
    model.train()

    return model


# Funzione per fare previsioni con Kriging
def predict_model(model, X_test, y_test):
    """Make predictions"""
    y_pred = model.predict_values(X_test)
    var = model.predict_variances(X_test)
    print("Kriging,  err: " + str(compute_rms_error(model, X_test, y_test)))

    # Value of theta
    print("theta values", model.optimal_theta)

    return {"y_pred": y_pred, "variance": var}


# Funzione per fare previsioni sui dati di test con mf-kriging
def predict_mf_model(model, X_test, y_test):
    """Make prediction"""
    y_pred = model.predict_values(X_test)  # cosi fa la previsione sul livello piu alto
    var = model.predict_variances(X_test)

    # Value of theta
    print("theta values", model.optimal_theta)

    return {"y_pred": y_pred, "variance": var}


# Funzione per fare previsioni sui dati di test con mf-co-kriging
def predict_mfco_model(model, X_test):
    """Make predictions"""
    mn, cn = model.predict_all_levels(X_test)
    pred, cov = model._predict(X_test)

    # Value of theta
    print("theta values", model.optimal_theta)

    return mn, cn, pred, cov


def plot_validation(y_test, predictions, label):
    """Crea un grafico Predicted vs Actual"""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, color="blue", alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
    )
    plt.title(f"Predicted vs Actual {label}")
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label}")
    plt.grid()
    plt.show()


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


def prediction_metrics_plots(
    model,
    X_test,
    y_test,
    which_coefficent,
    altitude,
    aos,
    X_train,
    y_train,
    mach,
    aoa,
    selected_mach,
):

    # Prediction and metrics
    rms = compute_rms_error(model, X_test, y_test)
    predictions = predict_model(model, X_test, y_test)
    y_pred = predictions["y_pred"]
    var = predictions["variance"]

    print(f"RMS Error: {rms}")

    # Plot validation and response surfaces
    plot_validation(y_test, y_pred, which_coefficent)
    plot_response_surface(altitude, aos, X_train, y_train, model, which_coefficent, mach, aoa)
    plot_coefficent_alpha_for_mach(X_train, y_train, model, selected_mach, which_coefficent)

    return rms, predictions, y_pred, var


def high_variance_new_doe(
    var,
    n_samples,
    fraction_of_new_samples,
    X_test,
    processed_samples,
    ranges,
    output_filename,
    directory_path,
):

    print("Selecting DOE points with highest variance...")
    var_flat = var.flatten()
    sorted_indices = np.argsort(var_flat)[::-1]
    n_new_samples = n_samples // fraction_of_new_samples
    top_n_indices = sorted_indices[:n_new_samples]
    top_n_X_test = X_test[top_n_indices]

    # Print results
    print(f"Top {n_new_samples} variances: {var_flat[top_n_indices]}")
    print(f"Top {n_new_samples} X_test samples: {top_n_X_test}")

    # Plot DOE highlighting new points
    plot_doe(
        processed_samples,
        ranges,
        n_samples=n_new_samples,
        plot_dim1="angleOfAttack",
        plot_dim2="machNumber",
        highlight_points=top_n_X_test,
    )

    input("Press ENTER to continue: ")

    new_aeromap = {key: top_n_X_test[:, idx] for idx, key in enumerate(ranges.keys())}

    print("New aeromap with high variance points:")

    for key, value in new_aeromap.items():
        print(f"{key}: {value}")

    full_path = os.path.join(directory_path, output_filename)
    save_to_csv(new_aeromap, full_path)

    return new_aeromap, full_path


def sm_workflow(
    kriging_dataset_path,
    directory_path,
    output_filename,
    theta,
    corr,
    poly,
    selected_mach,
    altitude,
    aos,
    n_samples,
    fidelity_level,
    fraction_of_new_samples=None,
    ranges=None,
    processed_samples=None,
    coefficent_to_predict=None,
    X_train_LF=None,
    y_train_LF=None,
    X_train_MF=None,
    y_train_MF=None,
    base_model_name=None,
    model_extension=None,
):
    """
    Workflow for training surrogate models with support for multiple fidelity levels.

    Parameters:
    - fidelity_level: int (1, 2, 3) - The number of fidelity levels the model should have.
    """

    # Validate input
    if not isinstance(fidelity_level, int) or fidelity_level not in [1, 2, 3]:
        raise ValueError("fidelity_level must be an integer (1, 2, or 3).")

    # Load and prepare the dataset (only for LF)
    df = load_and_split_data(kriging_dataset_path)
    X = df["dataset"]["X"]
    y = df["dataset"]["y"]

    # Select coefficient to predict
    if coefficent_to_predict is None:
        which_coefficent = (
            input("Insert which coefficient to predict (CL, CD, CM) [default: CL]: ") or "CL"
        )
    else:
        which_coefficent = coefficent_to_predict

    if which_coefficent not in y:
        raise KeyError(
            f"The specified coefficient '{which_coefficent}' is not available in the dataset."
        )

    coefficent = y[which_coefficent]

    # Split data into training and test sets (LF data)
    train_test_values = test_training_data(X, coefficent)
    X_train = train_test_values["X_train"]
    X_test = train_test_values["X_test"]
    y_train = train_test_values["y_train"]
    y_test = train_test_values["y_test"]

    print("Data splitted into training and test sets.")

    # Initialize variables for iteration
    model = None
    top_n_X_test = None

    # Training based on fidelity_level
    if fidelity_level == 1:
        print(f"Training surrogate model...")
        model = Kriging(X_train, y_train, theta, corr, poly)
        # Prediction metrics and graphs
        rms, prediction, y_pred, var = prediction_metrics_plots(
            model,
            X_test,
            y_test,
            which_coefficent,
            altitude,
            aos,
            X_train,
            y_train,
            ranges["machNumber"],
            ranges["angleOfAttack"],
            selected_mach,
        )
        input("Press ENTER to continue: ")

        # Saving of the model
        print("Saving model...")
        save_model(model, directory_path, base_model_name, model_extension)

    elif fidelity_level == 2:
        if X_train_LF is None and y_train_LF is None:
            # First iteration
            print(f"Training first surrogate model...")
            model = Kriging(X_train, y_train, theta, corr, poly)
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            new_aeromap, full_path = high_variance_new_doe(
                var,
                n_samples,
                fraction_of_new_samples,
                X_test,
                processed_samples,
                ranges,
                output_filename,
                directory_path,
            )

        else:
            # Second iteration
            print("Training final multi fidelity surrogate model...")
            model = MF_Kriging(X_train_LF, y_train_LF, X_train, y_train, theta, corr, poly)
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            input("Press ENTER to continue: ")

            # Saving of the model
            print("Saving model...")
            save_model(model, directory_path, base_model_name, model_extension)

    else:
        if X_train_LF is None and y_train_LF is None and X_train_MF is None and y_train_MF is None:
            # First iteration
            print(f"Training first surrogate model...")
            model = Kriging(X_train, y_train, theta, corr, poly)
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            new_aeromap, full_path = high_variance_new_doe(
                var,
                n_samples,
                fraction_of_new_samples,
                X_test,
                processed_samples,
                ranges,
                output_filename,
                directory_path,
            )

        elif X_train_MF is None and y_train_MF is None:
            # Second iteration
            print(f"Training first multy fidelity surrogate model...")
            model = MF_Kriging(X_train_LF, y_train_LF, X_train, y_train, theta, corr, poly)
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            # Fraction for RANS should be different
            new_aeromap, full_path = high_variance_new_doe(
                var,
                n_samples,
                fraction_of_new_samples,
                X_test,
                processed_samples,
                ranges,
                output_filename,
                directory_path,
            )

        else:
            # Third iteration
            print("Training final multi fidelity surrogate model...")
            model = MF_Kriging(
                X_train_LF, y_train_LF, X_train_MF, y_train_MF, theta, corr, poly, X_train, y_train
            )
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            input("Press ENTER to continue: ")
            # Saving of the model
            print("Saving model...")
            save_model(model, directory_path, base_model_name, model_extension)

    return new_aeromap, full_path, which_coefficent, top_n_X_test, model, rms, X_train, y_train


# =================== SAVE MODEL ===============


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


# ===================== VALIDATION ======================


def plot_response_surface(
    altitude, aos, X_train, Y_train, model, coeff, mach_range=None, aoa_range=None
):
    """
    Plot the response surface for a Kriging surrogate model.

    Args:
        altitude (float): Altitude to generate the response surface at.
        X_train (np.ndarray): DoE points used for training (Mach, AoA, Altitude).
        Y_train (np.ndarray): DoE values used for training (Coeff).
        model (object): Trained surrogate model.
        mach_range (list or None): Range of Mach numbers [min, max] (default: DoE range).
        aoa_range (list or None): Range of AoA [min, max] (default: DoE range).
    """
    # Determine Mach and AoA ranges
    if mach_range is None:
        mach_range = [X_train[:, 1].min(), X_train[:, 1].max()]
    if aoa_range is None:
        aoa_range = [X_train[:, 2].min(), X_train[:, 2].max()]

    # Create grid for Mach and AoA
    mach_values = np.linspace(mach_range[0], mach_range[1], 70)
    aoa_values = np.linspace(aoa_range[0], aoa_range[1], 70)
    Mach, AoA = np.meshgrid(mach_values, aoa_values)

    # Generate prediction points
    X_pred = np.column_stack(
        [np.full(Mach.size, altitude), Mach.ravel(), AoA.ravel(), np.full(Mach.size, aos)]
    )

    # Predict on the grid
    predictions = predict_model(model, X_pred, y_test=None)
    y_pred = predictions["y_pred"]
    coeff_pred_surface = y_pred.reshape(Mach.shape)

    # Filter DoE points at the selected altitude
    doe_idx = np.isclose(X_train[:, 0], altitude)
    doe_points = X_train[doe_idx]
    doe_coeff = Y_train[doe_idx]

    coeff_doe_surface = griddata(
        points=doe_points[:, 1:3],  # Use Mach and AoA as the grid dimensions
        values=doe_coeff,  # Corresponding coefficent values
        xi=(Mach, AoA),  # The mesh grid
        method="linear",  # Interpolation method
        fill_value=np.nan,  # Fill missing values with NaN
    )

    # Plot the response surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot predicted surface
    surf_pred = ax.plot_surface(
        Mach,
        AoA,
        coeff_pred_surface,
        cmap="viridis",
        alpha=0.8,
        edgecolor="none",
        label="Predicted Surface",
    )

    # Plot DoE surface
    surf_doe = ax.plot_surface(
        Mach,
        AoA,
        coeff_doe_surface,
        cmap="plasma",
        alpha=0.5,
        edgecolor="none",
        label="DoE Surface",
    )

    # Scatter DoE points
    scatter = ax.scatter(
        doe_points[:, 1],  # Mach of DoE points
        doe_points[:, 2],  # AoA of DoE points
        doe_coeff,  # values of DoE points
        color="red",
        label="DoE Points",
        depthshade=False,
    )

    # Plot details
    ax.set_xlabel("Mach Number")
    ax.set_ylabel("Angle of Attack (AoA)")
    ax.set_zlabel(f"{coeff}")
    ax.set_title(f"Response Surface of {coeff} at Altitude = {altitude} m, AoS = {aos}°")

    ax.invert_xaxis()

    # Aggiungi la legenda solo per gli scatter
    ax.legend(handles=[scatter], loc="upper right")  # Usa solo lo scatter nella legenda

    plt.colorbar(surf_doe, ax=ax, shrink=0.5, aspect=10, label=f"Predicted {coeff}")
    plt.show()


def plot_coefficent_alpha_for_mach(
    X_train, y_train, model, selected_mach_numbers, which_coefficent
):
    """
    Plot the Coefficent vs AoA graph for three selected Mach numbers.

    Args:
        X_train (np.ndarray): DoE points used for training (Mach, AoA, Altitude).
        Y_train (np.ndarray): DoE values used for training (Coefficent).
        model (object): Trained surrogate model.
        selected_mach_numbers (list): Three Mach numbers to visualize.
    """

    plt.figure(figsize=(10, 6))

    # Plot DoE points and lines for reference
    for mach in selected_mach_numbers:
        mach_idx = np.isclose(X_train[:, 1], mach)
        aoa_values = X_train[mach_idx, 2]  # AoA for this Mach
        coef_values = y_train[mach_idx]  # Coefficent values for this Mach

        # Scatter plot of DoE points
        plt.scatter(
            aoa_values,
            coef_values,
            label=f"DoE Points Mach = {mach:.2f}",
            alpha=0.7,
        )

        # Sort AoA and corresponding coefficent values for a smooth line
        sorted_indices = np.argsort(aoa_values)
        sorted_aoa = aoa_values[sorted_indices]
        sorted_coef = coef_values[sorted_indices]

        # Line passing through the points
        plt.plot(
            sorted_aoa,
            sorted_coef,
            linestyle="-",
            alpha=0.8,
            label=f"Line Mach = {mach:.2f}",
        )

    # Plot details
    plt.xlabel("Angle of Attack (AoA)")
    plt.ylabel(f"{which_coefficent}")
    plt.title(f"{which_coefficent} vs AoA for Selected Mach Numbers")
    plt.legend()
    plt.grid(True)
    plt.show()
