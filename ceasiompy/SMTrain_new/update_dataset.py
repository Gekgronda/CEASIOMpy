import re
import os
import pandas as pd


def sort_natural(s):
    """Funzione per ordinare in modo naturale stringhe come 'Case01', 'Case10', ecc."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_coefficients(base_path):
    """Estrae CL e CD dai file .dat nelle sottocartelle, in ordine naturale."""
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

            # Cerca i valori di CL e CD
            cl_match = re.search(r"Total CL:\s+([-+]?\d*\.\d+|\d+)", content)
            cd_match = re.search(r"Total CD:\s+([-+]?\d*\.\d+|\d+)", content)

            # Aggiunge i risultati solo se entrambi i valori sono trovati
            if cl_match and cd_match:
                cl = float(cl_match.group(1))
                cd = float(cd_match.group(1))
                results.append({"Index": i, "Total CL": cl, "Total CD": cd})
            else:
                print(f"Valori mancanti in {file_path}")

        print(f"Directory: {directory}, Total CL: {cl}, Total CD: {cd}")

    return results


def append_to_csv(data, filename):
    """Aggiunge i dati estratti come nuove colonne al file CSV esistente."""
    # Legge il CSV esistente e aggiunge un indice sequenziale
    df = pd.read_csv(filename)
    df["Index"] = range(len(df))

    # Crea un DataFrame con i nuovi dati e imposta l'indice
    new_data = pd.DataFrame(data)
    new_data.set_index("Index", inplace=True)

    print("Dati da appendere (primi 5):")
    print(new_data.head())

    # Unisce i DataFrame sui rispettivi indici
    df = df.set_index("Index").join(new_data, how="left", rsuffix="_new")

    # Sovrascrive il file CSV con le nuove colonne
    df.to_csv(filename, index=False, float_format="%.6f")
    print(f"Dati salvati nel file {filename} con successo.")


# Esecuzione
# qui devi dargli il path della cartella su2 dove sono contenuti i Workflow
base_path = input("Insert directory path: ")
# modifica il path del workflow
csv_filename = "/home/cfse/Stage_Gronda/datasets/takeoff4.csv"
data = extract_coefficients(base_path)
append_to_csv(data, csv_filename)
