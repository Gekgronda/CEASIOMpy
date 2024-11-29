import re
import os
import pandas as pd


def sort_natural(s):
    """Ordina stringhe in modo naturale come 'Case01', 'Case10', ecc."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_coefficients(base_path):
    """Estrae CLtot, CDtot, Cmtot dai file .txt nelle sottocartelle, in ordine naturale."""
    results = []

    # Ordina le directory in ordine naturale
    directories = sorted(
        [d for d in os.listdir(base_path) if d.startswith("Case")], key=sort_natural
    )

    for i, directory in enumerate(directories):
        directory_path = os.path.join(base_path, directory)
        file_path = os.path.join(directory_path, "ft.txt")

        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                content = file.read()

            # Usa regex per cercare i valori
            matches = {
                "CLtot": re.search(r"CLtot\s*=\s*([-+]?\d*\.\d+|\d+)", content),
                "CDtot": re.search(r"CDtot\s*=\s*([-+]?\d*\.\d+|\d+)", content),
                "Cmtot": re.search(r"Cmtot\s*=\s*([-+]?\d*\.\d+|\d+)", content),
            }

            # Verifica se tutti i valori sono stati trovati
            if all(matches.values()):
                results.append(
                    {
                        "Index": i,
                        "Total CL": float(matches["CLtot"].group(1)),
                        "Total CD": float(matches["CDtot"].group(1)),
                        "Total CMy": float(matches["Cmtot"].group(1)),
                    }
                )
            else:
                # Stampa un messaggio se manca qualche valore
                print("Valori mancanti nel file:")
                for key, match in matches.items():
                    if not match:
                        print(f" - {key}")

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
    base_path = (
        input("Inserisci il percorso della directory: ").strip()
        or "/wrk/Gronda/labAR/AVL/takeoff0/Workflow_001/Results/PyAVL"
    )
    csv_filename = (
        input("Inserisci il percorso del file CSV: ").strip()
        or "/wrk/Gronda/labAR/AVL/takeoff0/takeoff0.csv"
    )

    if os.path.isdir(base_path):
        data = extract_coefficients(base_path)
        print(data)
        append_to_new_csv(data, csv_filename)
    else:
        print(f"Errore: La directory {base_path} non esiste.")
