import re
import os
import pandas as pd


def sort_natural(s):
    """Ordina stringhe in modo naturale come 'Case01', 'Case10', ecc."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_coefficients(base_path):
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


def append_to_new_csv(data, original_filename):
    """Crea un nuovo file CSV con i dati estratti e lo salva con suffisso '_TRAIN'."""
    if not os.path.isfile(original_filename):
        print(f"Errore: Il file {original_filename} non esiste.")
        return

    # Legge il CSV esistente e aggiunge un indice sequenziale
    df = pd.read_csv(original_filename)
    if "Index" not in df.columns:
        df["Index"] = range(len(df))

    # Crea un DataFrame con i nuovi dati e imposta l'indice
    new_data = pd.DataFrame(data)
    if not new_data.empty:
        new_data.set_index("Index", inplace=True)

        # Unisce i DataFrame sui rispettivi indici
        df = df.set_index("Index").join(new_data, how="left", rsuffix="_new")

        # Salva il nuovo file con suffisso '_TRAIN'
        new_filename = os.path.splitext(original_filename)[0] + "_TRAIN.csv"
        df.to_csv(new_filename, index=False, float_format="%.6f")
        print(f"Dati salvati nel file {new_filename} con successo.")
    else:
        print("Nessun dato da aggiungere al file CSV.")


# Esecuzione
if __name__ == "__main__":
    base_path = input("Inserisci il percorso della directory: ").strip()
    csv_filename = (
        input("Inserisci il percorso del file CSV: ").strip()
        or "/wrk/Gronda/labAR/RANS/RANS_train_dataset.csv"
    )

    if os.path.isdir(base_path):
        data = extract_coefficients(base_path)
        append_to_new_csv(data, csv_filename)
    else:
        print(f"Errore: La directory {base_path} non esiste.")
