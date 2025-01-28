import re
import os
import pandas as pd

# Percorsi principali
principal_path = "/wrk/Gronda/d150/mesh_dependecy_euler"
su2_path = "Workflow_001/Results/SU2"
csv_output_path = "/wrk/Gronda/d150/mesh_dependecy_euler/mesh_dep_results.csv"


def sort_natural(s):
    """Ordina stringhe in modo naturale come 'Case01', 'Case10', ecc."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


results = []

# Filtra e ordina le directory Workflow
directories = sorted(
    [
        d
        for d in os.listdir(principal_path)
        if d.startswith("Workflow") and "002" <= d.split("_")[1] <= "012"
    ],
    key=sort_natural,
)

# Itera sulle directory Workflow
for directory in directories:
    cases_path = os.path.join(principal_path, directory, su2_path)
    if not os.path.isdir(cases_path):
        print(f"Percorso non trovato: {cases_path}")
        continue

    # Filtra e ordina le directory Case
    cases = sorted([c for c in os.listdir(cases_path) if c.startswith("Case")], key=sort_natural)

    for case in cases:
        case_path = os.path.join(cases_path, case)
        file_path = os.path.join(case_path, "forces_breakdown.dat")

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

            # Se tutti i valori sono presenti, aggiungili ai risultati
            if all(matches.values()):
                results.append(
                    {
                        "Case": case,
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
        else:
            print(f"File non trovato: {file_path}")

# Crea il DataFrame finale
df = pd.DataFrame(results)

# Salva il DataFrame come file CSV
if not df.empty:
    df.to_csv(csv_output_path, index=False, float_format="%.6f")
    print(f"Dati salvati nel file {csv_output_path}.")
else:
    print("Nessun dato trovato. File CSV non creato.")
