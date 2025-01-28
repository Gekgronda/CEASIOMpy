import pandas as pd
import numpy as np
import os
import shutil
import csv
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler
from smt.surrogate_models import KRG
from smt.applications import EGO, MFK, MFKPLS, MFKPLSK
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
import itertools


from CPACS_Func import (
    add_new_aeromap,
    avl_update,
    change_reference_value,
    SU2_update,
)


def workflow(
    default_starting_dataset_path=None,
    default_first_dataset_path=None,
    default_second_dataset_path=None,
    default_third_dataset_path=None,
):
    """
    Allows the user to select the number of fidelity levels for the surrogate model
    and to decide whether to follow the workflow proposed by the code or provide
    custom dataset paths for training.

    Parameters:
    - default_starting_dataset_path: Default path for the starting dataset (used for fidelity level >= 1).
    - default_first_dataset_path: Default path for the first dataset (used if the user skips input or enters an invalid path).
    - default_second_dataset_path: Default path for the second dataset (for fidelity level >= 2).
    - default_third_dataset_path: Default path for the third dataset (for fidelity level == 3).

    Returns:
    - fidelity_level: The selected number of fidelity levels.
    - selected_paths: A dictionary containing the dataset paths for the selected fidelity levels.
    """

    # Input validation for fidelity level
    while True:
        fidelity_level = (
            input(
                "Choose how many fidelity levels to train the surrogate model (1, 2, 3) [default: 2]: "
            )
            or "2"
        )
        if fidelity_level in {"1", "2", "3"}:
            fidelity_level = int(fidelity_level)
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    # Determine whether to follow the default workflow or provide custom paths
    workflow = (
        (
            input(
                "Choose whether to follow the workflow proposed by the code (YES) or enter custom dataset paths (NO) [default: NO]: "
            )
            or "NO"
        )
        .strip()
        .upper()
    )

    # Prepare dataset paths
    selected_paths = {}
    if workflow == "NO":
        # Prompt user for the starting dataset path
        starting_dataset_path = input(
            f"Insert first file.csv path containing points to visualize the DOE [default: {default_starting_dataset_path}]: "
        ).strip()
        if not starting_dataset_path:
            print(f"No valid path provided. Using default: {default_starting_dataset_path}")
            starting_dataset_path = default_starting_dataset_path
        selected_paths["starting_dataset_path"] = starting_dataset_path

        # Prompt user for dataset paths based on fidelity level
        if fidelity_level >= 1:
            first_dataset_path = input(
                f"Insert second file.csv path to train the simple surrogate model [default: {default_first_dataset_path}]: "
            ).strip()
            if not first_dataset_path:
                print(f"No valid path provided. Using default: {default_first_dataset_path}")
                first_dataset_path = default_first_dataset_path
            selected_paths["first_dataset_path"] = first_dataset_path

        if fidelity_level >= 2:
            second_dataset_path = input(
                f"Insert third file.csv path to train the m-f surrogate model [default: {default_second_dataset_path}]: "
            ).strip()
            if not second_dataset_path:
                print(f"No valid path provided. Using default: {default_second_dataset_path}")
                second_dataset_path = default_second_dataset_path
            selected_paths["second_dataset_path"] = second_dataset_path

        if fidelity_level == 3:
            third_dataset_path = input(
                f"Insert fourth file.csv path to train the m-f surrogate model [default: {default_third_dataset_path}]: "
            ).strip()
            if not third_dataset_path:
                print(f"No valid path provided. Using default: {default_third_dataset_path}")
                third_dataset_path = default_third_dataset_path
            selected_paths["third_dataset_path"] = third_dataset_path

    else:
        # Workflow is "YES" -> Paths should remain empty
        selected_paths["starting_dataset_path"] = None
        if fidelity_level >= 1:
            selected_paths["first_dataset_path"] = None
        if fidelity_level >= 2:
            selected_paths["second_dataset_path"] = None
        if fidelity_level == 3:
            selected_paths["third_dataset_path"] = None

    # Output summary
    print(f"\nFidelity level: {fidelity_level}")
    print(f"Selected paths: {selected_paths}")

    return fidelity_level, selected_paths


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
    Ask the user to enter the number of samples and range of the domain.
    If no input is provided, use default parameters.

    Returns:
        tuple: Number of samples (int) and ranges (dict).
    """
    # Default values
    default_samples = 30
    default_ranges = {
        "altitude": [10000, 10000],
        "machNumber": [0.1, 0.7],
        "angleOfAttack": [-2, 10],
        "angleOfSideslip": [0, 0],
    }

    try:
        # Prompt user for the number of samples
        n_samples_input = input(f"Enter the number of samples (default: {default_samples}): ")
        n_samples = int(n_samples_input) if n_samples_input.strip() else default_samples

        print("Enter the range for each dimension (press Enter to use default values):")
        ranges = {}
        for key, default_range in default_ranges.items():
            try:
                min_input = input(f"Minimum {key} (default: {default_range[0]}): ")
                max_input = input(f"Maximum {key} (default: {default_range[1]}): ")

                # Use defaults if input is empty
                min_value = float(min_input) if min_input.strip() else default_range[0]
                max_value = float(max_input) if max_input.strip() else default_range[1]

                if min_value > max_value:
                    raise ValueError(f"Minimum value cannot be greater than maximum for '{key}'.")

                ranges[key] = [min_value, max_value]
            except ValueError as ve:
                print(f"Invalid input for {key}: {ve}. Using default range {default_range}.")
                ranges[key] = default_range

        return n_samples, ranges
    except ValueError as e:
        raise ValueError(f"Invalid input for the number of samples: {e}")


def lh_sampling(ranges, n_samples, physical_domain_limits, random_state=None):
    """
    Perform Latin Hypercube Sampling of the given domain.
    Apply physical domain limits to filter points.

    Parameters:
        ranges (dict): Variable ranges for sampling.
        n_samples (int): Number of samples to generate.
        physical_domain_limits (dict): Physical constraints for filtering points.
        random_state (int, optional): Random state for reproducibility.

    Returns:
        tuple: A dictionary of sampled points and a corresponding numpy array.
    """

    # Perform Latin Hypercube Sampling
    sampling = LHS(
        xlimits=np.array(list(ranges.values())), criterion="ese", random_state=random_state
    )
    samples = sampling(n_samples)

    # Extract physical domain limits
    p1, p2, p3, p4 = (
        physical_domain_limits["p1"],
        physical_domain_limits["p2"],
        physical_domain_limits["p3"],
        physical_domain_limits["p4"],
    )

    # Calculate the lines: y = m*x + c
    m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c1 = p1[1] - m1 * p1[0]

    m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    c2 = p3[1] - m2 * p3[0]

    # Map sampled values back to variable names
    sampled_dict = {key: samples[:, idx] for idx, key in enumerate(ranges.keys())}

    # Post-process sampled data to apply precision
    for key in sampled_dict:
        if key in ["altitude", "machNumber"]:
            sampled_dict[key] = np.round(sampled_dict[key], 1)
        elif key in ["angleOfAttack", "angleOfSideslip"]:
            sampled_dict[key] = np.round(sampled_dict[key]).astype(int)

    # Extract relevant variables for filtering
    altitude = sampled_dict["altitude"]
    mach = sampled_dict["machNumber"]
    aoa = sampled_dict["angleOfAttack"]
    aos = sampled_dict["angleOfSideslip"]

    # Filter points within the physical domain
    mask = (mach <= m1 * aoa + c1) & (mach <= m2 * aoa + c2)

    # Apply the mask to all variables
    sampled_dict = {key: np.array(values)[mask] for key, values in sampled_dict.items()}

    # Convert back to array for plotting
    sampled_array = np.column_stack([sampled_dict[key] for key in ranges.keys()])

    return sampled_dict, sampled_array


def plot_doe(
    sampled_array,
    ranges,
    n_samples,
    physical_domain_limits,
    plot_dim1="angleOfAttack",
    plot_dim2="machNumber",
    highlight_points=None,
):
    """
    Plot the Design of Experiment (DoE) with optional highlights and physical domain limits.

    Parameters:
        sampled_array (numpy.ndarray): Array of sampled data points where each row is a sample and each column corresponds to a variable.
        ranges (dict): Dictionary specifying the range for each variable. Keys are variable names, values are lists with [min, max].
        n_samples (int): Number of samples used in the DoE.
        physical_domain_limits (dict): Dictionary containing physical constraints of the domain.
        plot_dim1 (str, optional): Name of the variable to plot on the x-axis. Defaults to "angleOfAttack".
        plot_dim2 (str, optional): Name of the variable to plot on the y-axis. Defaults to "machNumber".
        highlight_points (numpy.ndarray, optional): Array of points to highlight on the plot.
    """

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

    ax.plot(sampled_array[:, PLOT1], sampled_array[:, PLOT2], ".")

    # Highlight specific points if provided
    if highlight_points is not None:
        ax.plot(
            highlight_points[:, PLOT1],
            highlight_points[:, PLOT2],
            "ro",
            label="High Variance Points",
        )

    # plot of pysical domain limits

    p1 = physical_domain_limits["p1"]
    p2 = physical_domain_limits["p2"]

    x_a = [p1[0], p2[0]]
    y_a = [p1[1], p2[1]]
    plt.plot(x_a, y_a, linestyle="-.", color="black", marker="o")

    p3 = physical_domain_limits["p3"]
    p4 = physical_domain_limits["p4"]

    x_b = [p3[0], p4[0]]
    y_b = [p3[1], p4[1]]
    plt.plot(x_b, y_b, linestyle="-.", color="black", marker="o")

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


def doe_workflow(selected_paths, directory_path, output_filename, physical_domain_limits):
    """
    Workflow to handle Design of Experiment (DoE).

    Parameters:
        selected_paths (dict): Dictionary with paths to datasets.
        directory_path (str): Directory to save new DoE files.
        output_filename (str): Filename to save the generated DoE.
        physical_domain_limits (dict): Physical limits of the domain for validation and sampling.

    Returns:
        tuple: A tuple containing:
            - samples (dict): Dictionary representation of the DoE.
            - ranges (dict): Ranges of each parameter in the DoE.
            - sampled_array (ndarray): Numpy array of the sampled DoE.
            - n_samples (int): Number of samples in the DoE.
            - full_path (str): Full path to the DoE file.
    """

    if selected_paths["starting_dataset_path"] is None:

        print(
            "You choose to follow the workflow proposed by the code, now you're going to define de DOE"
        )

        # Insert a domain of interest
        n_samples, ranges = get_user_inputs()

        # Latin Hypercube Sampling (DoE)
        print("Sampling...")
        samples, sampled_array = lh_sampling(ranges, n_samples, physical_domain_limits)

        # Save DoE to CSV
        full_path = os.path.join(directory_path, output_filename)
        save_to_csv(samples, full_path)
        print(f"New DoE saved in {full_path}")

    # AGGIUNGERE VERIFICHE

    else:
        try:
            # Load the DoE from the file
            doe_path = selected_paths["starting_dataset_path"]
            if not os.path.exists(doe_path):
                raise FileNotFoundError(f"The file '{doe_path}' does not exist.")

            # Read CSV into DataFrame
            df = pd.read_csv(doe_path)
            if df.empty:
                raise ValueError(f"The file '{doe_path}' is empty or invalid.")

            print(f"Loaded database from: {doe_path}")

            # Convert DataFrame to dictionary
            samples = df.to_dict(orient="list")

            # Compute ranges and processed samples
            ranges = {col: [df[col].min(), df[col].max()] for col in df.columns}
            sampled_array = df.to_numpy()
            n_samples = len(df)
            full_path = doe_path

        except Exception as e:
            raise RuntimeError(f"Failed to load database: {e}")

    return samples, ranges, sampled_array, n_samples, full_path


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
    selected_paths,
    default_kriging_dataset_path,
    full_path,
    samples,
    aeromap_uid,
    aeromap_name,
    avl_parameters,
):
    """
    Workflow to update CPACS file with new aeroMap and AVL parameters,
    and optionally run AVL simulations to generate a Kriging dataset.

    Parameters:
        input_cpacs_path (str): Path to the input CPACS file to be updated.
        directory_path (str): Directory for saving generated files.
        selected_paths (dict): Dictionary to store paths of important files during the workflow.
        default_first_kriging_dataset_path (str): Path to the default Kriging dataset.
        full_path (str): Path to the DoE file used for simulations.
        samples (dict): Dictionary containing sampled design points.
        aeromap_uid (str): Unique identifier for the new aeroMap in CPACS.
        aeromap_name (str): Name of the new aeroMap to be added.
        avl_parameters (dict): Parameters for AVL simulations.

    Returns:
        dict: Updated `selected_paths` dictionary with the path to the generated dataset.
    """

    # al momento lasciamo la possibilita di usare il dataset di default ma poi andra tolto (anche dagli argomenti)

    # Check if the first dataset path is already set
    if selected_paths["first_dataset_path"] is not None:
        # Return the existing selected_paths without modification
        return selected_paths

    print("Updating aeromap and reference value for AVL simulations")
    input("Press ENTER to continue....")

    # Aeromap updating on CPACS
    tixi = Tixi3()
    tixi.open(input_cpacs_path)

    try:
        # add the new aeroMap
        add_new_aeromap(tixi, samples, aeromap_uid, aeromap_name)
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
    selected_paths["first_dataset_path"] = kriging_dataset_path

    return selected_paths


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


def select_coefficient_to_predict(y, coefficent_to_predict=None):
    """Select the coefficient to predict."""
    if coefficent_to_predict is None:
        coefficent_to_predict = (
            (input("Insert which coefficient to predict (CL, CD, CM) [default: CL]: ") or "CL")
            .strip()
            .upper()
        )

    if coefficent_to_predict == "CM":
        coefficent_to_predict = "CMy"

    if coefficent_to_predict not in y:
        raise KeyError(
            f"The specified coefficient '{coefficent_to_predict}' is not available in the dataset."
        )

    return coefficent_to_predict


def prepare_data(kriging_dataset_path, coefficent_to_predict=None):
    """Load and prepare dataset, split data, and select coefficient."""
    # Load and split the dataset
    df = load_and_split_data(kriging_dataset_path)
    X = df["dataset"]["X"]
    y = df["dataset"]["y"]

    print(X)

    # Select coefficient to predict
    which_coefficent = select_coefficient_to_predict(y, coefficent_to_predict)

    coefficent = y[which_coefficent]

    # Split data into training and test sets
    train_test_values = test_training_data(X, coefficent)
    X_train = train_test_values["X_train"]
    X_test = train_test_values["X_test"]
    y_train = train_test_values["y_train"]
    y_test = train_test_values["y_test"]

    print("Data split into training and test sets.")

    print(f"X_train: {X_train}")
    print(f"y_train: {y_train}")
    print(f"X_test: {X_test}")
    print(f"y_test: {y_test}")

    return X_train, y_train, X_test, y_test, X, coefficent, which_coefficent


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
    model = MFKPLSK(
        theta0=theta, theta_bounds=[1e-06, 100.0], corr=corr, poly=poly, hyper_opt="TNC"
    )

    # Add high-fidelity data if provided
    if Xt_hf is None and yt_hf is None:
        # Set training values for the fidelity levels
        model.set_training_values(Xt_lf, yt_lf, name=0)  # Low-fidelity
        model.set_training_values(Xt_mf, yt_mf)  # Mid-fidelity
    else:
        # Set training values for the fidelity levels
        model.set_training_values(Xt_lf, yt_lf, name=0)  # Low-fidelity
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
    X,
    y,
    physical_domain_limits,
    mach,
    aoa,
    selected_mach,
    X_LF=None,
    y_LF=None,
    X_MF=None,
    y_MF=None,
):

    # Prediction and metrics
    rms = compute_rms_error(model, X_test, y_test)
    predictions = predict_model(model, X_test, y_test)
    y_pred = predictions["y_pred"]
    var = predictions["variance"]

    print(f"RMS Error: {rms}")

    # Plot validation and response surfaces
    plot_validation(y_test, y_pred, which_coefficent)
    plot_response_surface(
        altitude,
        aos,
        X,
        y,
        physical_domain_limits,
        model,
        which_coefficent,
        mach_range=None,
        aoa_range=None,
        X_LF=X_LF,
        y_LF=y_LF,
        X_MF=X_MF,
        y_MF=y_MF,
    )
    plot_coefficent_alpha_for_mach(
        altitude,
        aos,
        X,
        y,
        physical_domain_limits,
        model,
        selected_mach,
        which_coefficent,
        aoa_range=None,
        X_LF=X_LF,
        y_LF=y_LF,
        X_MF=X_MF,
        y_MF=y_MF,
    )

    return rms, predictions, y_pred, var


def high_variance_new_doe_pippi(
    X,
    var,
    n_samples,
    fraction_of_new_samples,
    X_test,
    sampled_array,
    ranges,
    output_filename,
    directory_path,
    physical_domain_limits,
    iteration_number,
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

    input("Press ENTER to continue: ")

    new_aeromap = {key: top_n_X_test[:, idx] for idx, key in enumerate(ranges.keys())}

    # Convert back to array for plotting
    new_aeromap_array = np.column_stack([new_aeromap[key] for key in ranges.keys()])

    print("New aeromap with high variance points:")

    for key, value in new_aeromap.items():
        print(f"{key}: {value}")

    full_path = os.path.join(directory_path, output_filename)
    save_to_csv(new_aeromap, full_path)

    return new_aeromap, full_path, new_aeromap_array


def high_variance_new_doe(
    X,
    var,
    n_samples,
    fraction_of_new_samples,
    X_test,
    sampled_array,
    ranges,
    output_filename,
    directory_path,
    physical_domain_limits,
    iteration_number,
):

    print("Selecting DOE points with highest variance...")
    var_flat = var.flatten()
    sorted_indices = np.argsort(var_flat)[::-1]
    n_new_samples = n_samples // fraction_of_new_samples
    top_n_indices = sorted_indices[:n_new_samples]
    top_n_X_test = np.vstack(X_test[top_n_indices])

    # Print results
    print(f"Top {n_new_samples} variances: {var_flat[top_n_indices]}")
    print(f"Top {n_new_samples} X_test samples: {top_n_X_test}")

    # Filter points with Mach > 0.7
    mach_above_threshold = X[X[:, 1] >= 0.7]

    # Generate all combinations of min and max values for the ranges
    range_bounds = [
        list(bound) for bound in ranges.values()
    ]  # Convert ranges to list of [min, max]
    extreme_values = np.array(list(itertools.product(*range_bounds)))

    if iteration_number == 1:
        # Combine high variance points, Mach >= 0.7, and extreme values
        additional_points = np.vstack([mach_above_threshold, extreme_values])
    elif iteration_number == 2:
        # Use only the extreme values
        additional_points = extreme_values

    # Combine all points
    combined_points = np.vstack([top_n_X_test, additional_points])

    # Remove duplicate points
    new_X = np.unique(combined_points, axis=0)

    print(f"new_X: {new_X}")

    input("Press ENTER to continue: ")

    new_aeromap = {key: new_X[:, idx] for idx, key in enumerate(ranges.keys())}

    # Convert back to array for plotting
    new_aeromap_array = np.column_stack([new_aeromap[key] for key in ranges.keys()])

    print("New aeromap with high variance and additional points:")

    for key, value in new_aeromap.items():
        print(f"{key}: {value}")

    full_path = os.path.join(directory_path, output_filename)
    save_to_csv(new_aeromap, full_path)

    return new_aeromap, full_path, new_aeromap_array


def sm_workflow(
    iteration_number,
    selected_paths_updated,
    directory_path,
    theta,
    corr,
    poly,
    selected_mach,
    altitude,
    aos,
    n_samples,
    fidelity_level,
    physical_domain_limits,
    fraction_of_new_samples=None,
    ranges=None,
    sampled_array=None,
    coefficent_to_predict=None,
    X_LF=None,
    y_LF=None,
    X_MF=None,
    y_MF=None,
    X_train_LF=None,
    y_train_LF=None,
    X_train_MF=None,
    y_train_MF=None,
    base_model_name=None,
    model_extension=None,
    output_filename=None,
):
    """
    Workflow for training surrogate models with support for multiple fidelity levels.

    Parameters:
    """

    # Training based on fidelity_level
    if fidelity_level >= 1:
        if iteration_number == 1:
            kriging_dataset_path = selected_paths_updated["first_dataset_path"]

            # Load and prepare the dataset
            X_train, y_train, X_test, y_test, X, coefficent, which_coefficent = prepare_data(
                kriging_dataset_path, coefficent_to_predict
            )

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
                X,
                coefficent,  # cambia anche gli altri!!
                physical_domain_limits,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            new_aeromap, full_path, new_aeromap_array = None, None, None

    if fidelity_level >= 2:
        # First iteration
        if iteration_number == 1:
            if selected_paths_updated["second_dataset_path"] is None:
                new_aeromap, full_path, new_aeromap_array = high_variance_new_doe(
                    X,
                    var,
                    n_samples,
                    fraction_of_new_samples,
                    X_test,
                    sampled_array,
                    ranges,
                    output_filename,
                    directory_path,
                    physical_domain_limits,
                    iteration_number,
                )
            else:
                new_aeromap, full_path = None, None
                aeromap_path = selected_paths_updated["second_dataset_path"]
                df = pd.read_csv(aeromap_path)
                new_aeromap_array = df.iloc[:, :4].to_numpy()

        if iteration_number == 2:
            # Second iteration
            kriging_dataset_path = selected_paths_updated["second_dataset_path"]

            # Load and prepare the dataset
            X_train, y_train, X_test, y_test, X, coefficent, which_coefficent = prepare_data(
                kriging_dataset_path, coefficent_to_predict
            )

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
                X,
                coefficent,  # cambia anche gli altri!!
                physical_domain_limits,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
                X_LF,
                y_LF,
            )

            new_aeromap, full_path, new_aeromap_array = None, None, None

    if fidelity_level >= 3:
        if iteration_number >= 2:
            if selected_paths_updated["third_dataset_path"] is None:
                new_aeromap, full_path, new_aeromap_array = high_variance_new_doe(
                    X,
                    var,
                    n_samples,
                    fraction_of_new_samples,
                    X_test,
                    sampled_array,
                    ranges,
                    output_filename,
                    directory_path,
                    physical_domain_limits,
                )
            else:
                new_aeromap, full_path = None, None
                aeromap_path = selected_paths_updated["third_dataset_path"]
                df = pd.read_csv(aeromap_path)
                new_aeromap_array = df.iloc[:, :4].to_numpy()

        if iteration_number == 3:

            # Third iteration
            kriging_dataset_path = selected_paths_updated["third_dataset_path"]
            # Load and prepare the dataset
            X_train, y_train, X_test, y_test, X, coefficent, which_coefficent = prepare_data(
                kriging_dataset_path, coefficent_to_predict
            )

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
                X,
                coefficent,  # cambia anche gli altri!!
                physical_domain_limits,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
                X_LF,
                y_LF,
                X_MF,
                y_MF,
            )

    input("Press ENTER to continue: ")

    if fidelity_level == iteration_number:
        # Saving of the model
        print("Saving model...")
        save_model(model, directory_path, base_model_name, model_extension)

    return (
        new_aeromap,
        new_aeromap_array,
        full_path,
        which_coefficent,
        model,
        rms,
        X,
        coefficent,
        X_train,
        y_train,
    )


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
    altitude,
    aos,
    X,
    y,
    physical_domain_limits,
    model,
    coeff,
    mach_range=None,
    aoa_range=None,
    X_LF=None,
    y_LF=None,
    X_MF=None,
    y_MF=None,
):
    """
    Plot the response surface for a surrogate model and compare with DoE points.

    Args:
        altitude (float): Altitude to generate the response surface at.
        aos (float): Angle of sideslip (AoS) for the surface.
        X (np.ndarray): High-fidelity DoE points (Mach, AoA, Altitude).
        y (np.ndarray): High-fidelity Coefficient values.
        model (object): Trained surrogate model.
        coeff (str): Coefficient to visualize (e.g., CL, CD, etc.).
        mach_range (list, optional): Range of Mach numbers [min, max].
        aoa_range (list, optional): Range of AoA [min, max].
        X_LF (np.ndarray, optional): Low-fidelity DoE points.
        y_LF (np.ndarray, optional): Low-fidelity Coefficient values.
        X_MF (np.ndarray, optional): Medium-fidelity DoE points.
        y_MF (np.ndarray, optional): Medium-fidelity Coefficient values.
    """

    p1 = physical_domain_limits["p1"]
    p2 = physical_domain_limits["p2"]
    p3 = physical_domain_limits["p3"]
    p4 = physical_domain_limits["p4"]

    # Calculate the lines: y = m*x + c
    m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])  # Slope of the lower line
    c1 = p1[1] - m1 * p1[0]  # Intercept of the lower line

    m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])  # Slope of the upper line
    c2 = p3[1] - m2 * p3[0]  # Intercept of the upper line

    # Generate grid points
    aoa_grid = np.linspace(X[:, 2].min(), X[:, 2].max(), 50)  # AoA values
    mach_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)  # Mach values
    AoA, Mach = np.meshgrid(aoa_grid, mach_grid)  # 2D grid

    # Filter points within the lines
    mask = (Mach <= m1 * AoA + c1) & (Mach <= m2 * AoA + c2)

    # Apply the filter
    AoA_filtered = AoA[mask]
    Mach_filtered = Mach[mask]

    # Reshape the filtered grid into a list of points for prediction
    X_pred = np.column_stack(
        [
            np.full(AoA_filtered.size, altitude),  # Altitude
            Mach_filtered,  # Mach values
            AoA_filtered,  # AoA values
            np.full(AoA_filtered.size, aos),  # AoS
        ]
    )

    # Predictions on filtered points
    predictions = model.predict_values(X_pred)

    # Number of valid points
    num_points = len(AoA_filtered)

    # Verify that the number of predictions matches the number of valid points
    if num_points != predictions.size:
        raise ValueError(
            f"The predictions ({predictions.size}) do not match the number of valid points ({num_points}). "
            "Check the grid or physical domain."
        )

    # Create the response surface using an interpolation method
    coeff_pred_surface = predictions  # Directly use the filtered data

    # Plot the response surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Predicted surface with trisurf (for irregular grids)
    ax.plot_trisurf(
        AoA_filtered,  # Change Mach to AoA
        Mach_filtered,  # Change AoA to Mach
        coeff_pred_surface.flatten(),
        cmap="viridis",
        alpha=0.5,
        edgecolor="none",
    )

    # Filter DoE points at the selected altitude and AoS
    doe_idx = (X[:, 0] == altitude) & (X[:, 3] == aos)
    doe_points = X[doe_idx]
    doe_coeff = y[doe_idx]

    ax.scatter(
        doe_points[:, 2],  # AoA
        doe_points[:, 1],  # Mach
        doe_coeff,  # Coefficients
        color="blue",
        marker="x",
        label="DoE Points",
        depthshade=False,
        zorder=10,
        alpha=1,
    )

    # Scatter Low-Fidelity DoE points if provided
    if X_LF is not None and y_LF is not None:
        # Filter Low-Fidelity DoE points
        lf_doe_idx = (X_LF[:, 0] == altitude) & (X_LF[:, 3] == aos)
        lf_doe_points = X_LF[lf_doe_idx]
        lf_doe_coeff = y_LF
        # Scatter DoE points
        scatter_lf = ax.scatter(
            lf_doe_points[:, 2],  # AoA of DoE points
            lf_doe_points[:, 1],  # Mach of DoE points
            lf_doe_coeff,  # values of DoE points
            marker="o",
            color="black",
            label="LF DoE Points",
            depthshade=False,
        )

    # Scatter Medium-Fidelity DoE points if provided
    if X_MF is not None and y_MF is not None:
        mf_doe_idx = (X_MF[:, 0] == altitude) & (X_MF[:, 3] == aos)
        mf_doe_points = X_MF[mf_doe_idx]
        mf_doe_coeff = y_MF["y_pred"]
        scatter_mf = ax.scatter(
            mf_doe_points[:, 2],  # AoA of MF points
            mf_doe_points[:, 1],  # Mach of MF points
            mf_doe_coeff,  # Coeff values
            marker="^",
            label="MF DoE Points",
            depthshade=False,
        )

    # Plot details
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel("Mach Number")
    ax.set_ylabel("Angle of Attack (AoA)")
    ax.set_zlabel(f"{coeff}")
    ax.set_title(f"Response Surface of {coeff} at Altitude = {altitude} m, AoS = {aos}°")
    ax.view_init(elev=25, azim=45)
    ax.legend()

    # Color bar
    colorbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.5, aspect=10)
    colorbar.set_label(f"Predicted {coeff}")

    plt.show()


def plot_coefficent_alpha_for_mach(
    altitude,
    aos,
    X,
    y,
    physical_domain_limits,
    model,
    selected_mach_numbers,
    which_coefficent,
    aoa_range=None,
    X_LF=None,
    y_LF=None,
    X_MF=None,
    y_MF=None,
):
    """
    Plot the Coefficient vs AoA graph for three selected Mach numbers.

    Args:
        X (np.ndarray): DoE points (Mach, AoA, Altitude).
        y (np.ndarray): Coefficient values.
        physical_domain_limits (dict, optional): Physical domain limits (p1, p2, p3, p4).
        model (object): Trained surrogate model.
        selected_mach_numbers (list): Three Mach numbers to visualize.
        which_coefficent (str): Coefficient name (e.g., CL, CD).
        X_LF (np.ndarray, optional): Low-fidelity DoE points.
        y_LF (np.ndarray, optional): Low-fidelity Coefficient values.
        X_MF (np.ndarray, optional): Medium-fidelity DoE points.
        y_MF (np.ndarray, optional): Medium-fidelity Coefficient values.
    """

    p1 = physical_domain_limits["p1"]
    p2 = physical_domain_limits["p2"]
    p3 = physical_domain_limits["p3"]
    p4 = physical_domain_limits["p4"]

    # Calculate the lines: y = m*x + c
    m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])  # Slope of the lower line
    c1 = p1[1] - m1 * p1[0]  # Intercept of the lower line

    m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])  # Slope of the upper line
    c2 = p3[1] - m2 * p3[0]  # Intercept of the upper line

    plt.figure(figsize=(10, 6))

    # Define colors for each iteration
    colors = ["black", "blue", "red"]

    # Plot DoE points and lines for reference
    for i, mach in enumerate(selected_mach_numbers):

        color = colors[i]
        # Generate grid points
        aoa_grid = np.linspace(X[:, 2].min(), X[:, 2].max(), 50)  # AoA values
        print(f"X[:, 2].min(): {X[:, 2].min()}")
        print(f"aoa_grid: {aoa_grid}")

        mach_grid = np.full_like(aoa_grid, mach)  # Corresponding Mach values for this grid

        # Calculate Mach values using the equation of the lines
        mask = (mach_grid <= m1 * aoa_grid + c1) & (mach_grid <= m2 * aoa_grid + c2)

        # Apply the mask to AoA values
        aoa_filtered = aoa_grid[mask]

        # DoE points (scatter)
        mach_idx = np.isclose(X[:, 1], mach)
        aoa_doe_values = X[mach_idx, 2]  # AoA for this Mach
        coef_doe_values = y[mach_idx]  # Coefficient values for this Mach
        plt.scatter(
            aoa_doe_values,
            coef_doe_values,
            label=f"DoE Points Mach = {mach:.2f}",
            color=color,
            alpha=0.7,
            marker="x",
        )

        # Scatter Low-Fidelity points if provided
        if X_LF is not None and y_LF is not None:
            mach_idx_lf = np.isclose(X_LF[:, 1], mach)
            aoa_values_lf = X_LF[mach_idx_lf, 2]
            coef_values_lf = y_LF[mach_idx_lf]
            plt.scatter(
                aoa_values_lf,
                coef_values_lf,
                label=f"LF DoE Points Mach = {mach:.2f}",
                color=color,
                alpha=0.7,
                marker="o",
            )

        # Scatter Medium-Fidelity points if provided
        if X_MF is not None and y_MF is not None:
            mach_idx_mf = np.isclose(X_MF[:, 1], mach)
            aoa_values_mf = X_MF[mach_idx_mf, 2]
            coef_values_mf = y_MF[mach_idx_mf]
            plt.scatter(
                aoa_values_mf,
                coef_values_mf,
                label=f"MF DoE Points Mach = {mach:.2f}",
                color=color,
                alpha=0.7,
                marker="^",
            )

        pred_points = np.column_stack(
            [
                np.full(aoa_filtered.shape, altitude),  # Altitude
                np.full(aoa_filtered.shape, mach),  # Mach number
                aoa_filtered,
                np.full(aoa_filtered.shape, aos),  # AoS
            ]
        )
        predictions = predict_model(model, pred_points, y_test=None)
        y_pred = predictions["y_pred"]

        plt.plot(
            aoa_filtered,
            y_pred,
            linestyle="-",
            alpha=0.8,
            label=f"Model Prediction Mach = {mach:.2f}",
            color=color,
        )

    # Plot details
    plt.xlabel("Angle of Attack (AoA)")
    plt.ylabel(f"{which_coefficent}")
    plt.title(f"{which_coefficent} vs AoA for Selected Mach Numbers")
    plt.legend()
    plt.grid(True)
    plt.show()
