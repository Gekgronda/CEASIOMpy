import re
import os
import csv


def extract_coefficients(base_path):
    """Estrae CL e CD dai file .dat nelle sottocartelle."""
    results = []

    for directory in os.listdir(base_path):
        directory_path = os.path.join(base_path, directory)
        file_path = os.path.join(base_path, directory, "forces_breakdown.dat")

        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                content = file.read()

            cl = re.search(r"Total CL:\s+([-+]?\d*\.\d+|\d+)", content).group(1)
            cd = re.search(r"Total CD:\s+([-+]?\d*\.\d+|\d+)", content).group(1)

            results.append({"Total CL": cl, "Total CD": cd})

    return results


def save_to_csv(data, filename):
    """Salva i dati estratti in un file CSV."""
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Total CL", "Total CD"])
        writer.writeheader()  # Scrive l'intestazione
        writer.writerows(data)


# Esecuzione
base_path = input("Insert directory path: ")
output_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/"
output_csv = input("Insert the output .csv file name (with .csv extention): ")

# Estrai i coefficienti
data = extract_coefficients(base_path)

# Salva i dati in un CSV
save_to_csv(data, f"{output_directory}{output_csv}")
