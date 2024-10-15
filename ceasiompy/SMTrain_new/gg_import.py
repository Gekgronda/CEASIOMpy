import re
import os
import csv


def coefficent_extraction_from_directories(base_path):
    """
    Estrae i valori di altitude, Mach number, AoA, AoS, CL, e CD da file .dat nelle cartelle.

    Args:
        base_path (str): Il percorso base delle cartelle contenenti i file .dat.

    Returns:
        list: Una lista di dizionari contenenti i valori estratti da ogni cartella.
    """
    res = []

    # Scorri attraverso le cartelle nel percorso base
    for directory_name in os.listdir(base_path):
        if re.match(
            r"Case\d+_alt[\d\.]+_mach[\d\.]+_aoa[-]?[\d\.]+_aos[-]?[\d\.]+", directory_name
        ):
            directory_path = os.path.join(base_path, directory_name)
            file_path = os.path.join(directory_path, "forces_breakdown.dat")

            if os.path.isfile(file_path):
                print(f"Processing file: {file_path}")  # Debug message
                # Estrai i coefficienti dal file
                coefficents = data_extraction(file_path)
                # Aggiungi il nome della cartella come un nuovo campo
                coefficents["Directory Name"] = directory_name
                res.append(coefficents)
            else:
                print(f"File not found: {file_path}")  # Debug message

    return res


def data_extraction(file_path):
    """
    Extract coefficents values (CL, CD, ecc.), Mach number, AoA e AoS dal file .dat.

    Args:
        file_path (str): Path of file .dat.

    Returns:
        dict: Dictionary with extracted files.
    """

    # Estrai l'altitude, Mach number, AoA e AoS dal nome della cartella
    folder_name = os.path.basename(os.path.dirname(file_path))
    altitude = re.search(r"alt([\d\.]+)", folder_name).group(1)
    mach_number = re.search(r"mach([\d\.]+)", folder_name).group(1)
    aoa = re.search(r"aoa([-]?[\d\.]+)", folder_name).group(1)
    aos = re.search(r"aos([-]?[\d\.]+)", folder_name).group(1)

    # Define regex pattern for every coefficents and other values
    pattern_cl = r"Total CL:\s+([-+]?\d*\.\d+|\d+)"
    pattern_cd = r"Total CD:\s+([-+]?\d*\.\d+|\d+)"
    # pattern_csf = r"Total CSF:\s+([-+]?\d*\.\d+|\d+)"
    # pattern_cl_cd = r"Total CL/CD:\s+([-+]?\d*\.\d+|\d+)"
    # pattern_cmx = r"Total CMx:\s+([-+]?\d*\.\d+|\d+)"
    # pattern_cmy = r"Total CMy:\s+([-+]?\d*\.\d+|\d+)"
    # pattern_cmz = r"Total CMz:\s+([-+]?\d*\.\d+|\d+)"
    # pattern_cfx = r"Total CFx:\s+([-+]?\d*\.\d+|\d+)"
    # pattern_cfy = r"Total CFy:\s+([-+]?\d*\.\d+|\d+)"
    # pattern_cfz = r"Total CFz:\s+([-+]?\d*\.\d+|\d+)"

    # Altitude from the file path
    folder_name = os.path.basename(os.path.dirname(file_path))
    altitude = re.search(r"alt([\d\.]+)", folder_name)
    altitude = altitude.group(1) if altitude else None

    # Read file
    with open(
        file_path,
        "r",
    ) as file:
        content = file.read()

    # Extract values using defined regex
    data_extraction = {
        "Altitude": altitude,
        "Mach number": mach_number,
        "Angle of attack (AoA)": aoa,
        "Angle of sideslip (AoS)": aos,
        "Total CL": re.search(pattern_cl, content).group(1),
        "Total CD": re.search(pattern_cd, content).group(1),
        # "Total CSF": re.search(pattern_csf, content).group(1),
        # "Total CL/CD": re.search(pattern_cl_cd, content).group(1),
        # "Total CMx": re.search(pattern_cmx, content).group(1),
        # "Total CMy": re.search(pattern_cmy, content).group(1),
        # "Total CMz": re.search(pattern_cmz, content).group(1),
        # "Total CFx": re.search(pattern_cfx, content).group(1),
        # "Total CFy": re.search(pattern_cfy, content).group(1),
        # "Total CFz": re.search(pattern_cfz, content).group(1),
    }

    return data_extraction


def save_to_csv(data, filename):
    """
    Salva i dati estratti in un file CSV.

    Args:
        data (list): I dati da salvare nel file CSV.
        filename (str): Il nome del file CSV.
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)

    if data:
        # Apri il file CSV in modalità scrittura
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()  # Scrivi l'intestazione
            writer.writerows(data)  # Scrivi i dati
    else:
        print("No data to save to CSV")


# Example
# base_path = input("Insert directory path: ")
base_path = "/home/cfse/Stage_Gronda/CEASIOMpy/WKDIR/Workflow_027/Results/SU2"
extract_values = coefficent_extraction_from_directories(base_path)

# Salva i valori estratti in un file CSV
name = input("Insert the new .csv file name (write with .csv extrention): ")
csv_filename = f"/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/{name}"
save_to_csv(extract_values, csv_filename)

# Stampa i valori estratti
# print("Extracted Values:")
# print()
# for result in extract_values:
#     for key, val in result.items():
#         print(f"{key}: {val}")
#     print()  # Aggiungi una riga vuota per separare i risultati delle diverse cartelle
