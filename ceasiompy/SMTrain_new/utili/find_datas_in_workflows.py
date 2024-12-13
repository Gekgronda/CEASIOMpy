import re
import os
import csv


def sort_natural(s):
    """Ordina stringhe in modo naturale, es. Workflow001, Workflow010, ecc."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_coefficients(base_path):
    """Estrae CL e CD dai file .dat all'interno della struttura specificata."""
    results = []

    # Trova tutte le directory Workflow ordinate naturalmente
    workflow_dirs = sorted(
        [d for d in os.listdir(base_path) if d.startswith("Workflow")], key=sort_natural
    )

    for workflow in workflow_dirs:
        workflow_path = os.path.join(base_path, workflow, "Results", "SU2")

        if os.path.isdir(workflow_path):
            # Trova la directory Case00_*
            case_dirs = [d for d in os.listdir(workflow_path) if d.startswith("Case00_")]
            if not case_dirs:
                print(f"Attenzione: Nessuna directory Case00_* trovata in {workflow_path}")
                continue

            # Usa la prima directory trovata (è unica per specifica)
            case_dir = os.path.join(workflow_path, case_dirs[0])
            file_path = os.path.join(case_dir, "forces_breakdown.dat")

            if os.path.isfile(file_path):
                with open(file_path, "r") as file:
                    content = file.read()

                # Estrai CL e CD
                cl_match = re.search(r"Total CL:\s+([-+]?\d*\.\d+|\d+)", content)
                cd_match = re.search(r"Total CD:\s+([-+]?\d*\.\d+|\d+)", content)

                if cl_match and cd_match:
                    results.append(
                        {
                            "Workflow": workflow,
                            "Total CL": float(cl_match.group(1)),
                            "Total CD": float(cd_match.group(1)),
                        }
                    )
                else:
                    print(f"Attenzione: Valori CL o CD mancanti in {file_path}")
            else:
                print(f"Attenzione: File forces_breakdown.dat non trovato in {case_dir}")

    return results


def save_to_csv(data, filename):
    """Salva i dati estratti in un file CSV."""
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Workflow", "Total CL", "Total CD"])
        writer.writeheader()  # Scrive l'intestazione
        writer.writerows(data)


# Esecuzione
base_path = "/wrk/Gronda/labAR/EULER/mesh_dependency/nuovo_cpacs_handling/"
output_csv = "/wrk/Gronda/labAR/EULER/mesh_dependency/nuovo_cpacs_handling/results3.csv"

# Estrai i coefficienti
data = extract_coefficients(base_path)

# Salva i dati in un CSV
save_to_csv(data, output_csv)
print(f"Dati salvati in: {output_csv}")
