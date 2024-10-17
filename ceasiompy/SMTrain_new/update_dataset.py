import re
import os
import csv
import pandas as pd


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


def append_to_csv(data, filename):
    """Aggiunge i dati estratti come nuove colonne al file CSV esistente."""
    # Legge il CSV esistente
    df = pd.read_csv(filename)

    # Crea un DataFrame con i nuovi dati
    new_data = pd.DataFrame(data)

    # Aggiunge le nuove colonne al DataFrame esistente
    df["Total CL"] = new_data["Total CL"]
    df["Total CD"] = new_data["Total CD"]

    # Sovrascrive il file CSV con le nuove colonne
    df.to_csv(filename, index=False)


# Esecuzione
base_path = input("Insert directory path: ")
name = input("Insert the new .csv file name (write with .csv extension): ")
csv_filename = f"/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/{name}"
data = extract_coefficients(base_path)
append_to_csv(data, csv_filename)
