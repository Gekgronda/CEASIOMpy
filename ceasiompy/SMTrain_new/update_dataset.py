import re
import os
import csv

import re
import os
import csv


def extract_coefficients(base_path):
    """Estrae CL e CD dai file .dat nelle sottocartelle."""
    results = []

    for directory in os.listdir(base_path):
        file_path = os.path.join(base_path, directory, "forces_breakdown.dat")

        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                content = file.read()

            cl = re.search(r"Total CL:\s+([-+]?\d*\.\d+|\d+)", content).group(1)
            cd = re.search(r"Total CD:\s+([-+]?\d*\.\d+|\d+)", content).group(1)

            results.append({"Total CL": cl, "Total CD": cd, "Directory": directory})

    return results


def append_to_csv(data, filename):
    """Aggiunge i dati estratti al file CSV."""
    with open(filename, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Total CL", "Total CD", "Directory"])
        writer.writerows(data)


# Esecuzione
base_path = "/home/cfse/Stage_Gronda/CEASIOMpy/WKDIR/Workflow_029/Results/SU2"
csv_filename = "dataset_500_points.csv"
data = extract_coefficients(base_path)
append_to_csv(data, csv_filename)
