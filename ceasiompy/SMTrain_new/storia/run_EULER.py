import os
import shutil
import pandas as pd
import subprocess
from ambiance import Atmosphere

# Definire i percorsi
principal_path = "/wrk/Gronda/EULER"
original_config_path = "/wrk/Gronda/EULER/Workflow_001/Results/SU2/Case00_alt1000.0_mach0.3_aoa0.0_aos0.0/ConfigCFD_new.cfg"
dataset_path = "/home/cfse/Stage_Gronda/datasets/takeoff0.csv"

# Leggere il dataset
data = pd.read_csv(dataset_path)

# Loop attraverso le righe del dataset
for index, row in data.iterrows():
    # 1. Creare la cartella Case{n}
    case_dir = os.path.join(principal_path, f"Case{index}")
    os.makedirs(case_dir, exist_ok=True)

    # 2. Copiare il file config nella cartella
    config_path = os.path.join(case_dir, "ConfigCFD_new.cfg")
    shutil.copy(original_config_path, config_path)

    # 3. Calcolare pressione e temperatura
    altitude = row["altitude"]  # Altitudine in metri
    atmosphere = Atmosphere(altitude)
    pressure = atmosphere.pressure[0]  # Pressione (Pa)
    temperature = atmosphere.temperature[0]  # Temperatura (K)

    # 4. Leggere e aggiornare il file config
    with open(config_path, "r") as file:
        lines = file.readlines()

    updated_lines = []
    added_sideslip_angle = False  # Per tenere traccia dell'aggiunta di SIDESLIP_ANGLE
    for line in lines:
        if line.startswith("MACH_NUMBER="):
            line = f"MACH_NUMBER={row['machNumber']}\n"
        elif line.startswith("AOA="):
            line = f"AOA={row['angleOfAttack']}\n"
        elif line.startswith("FREESTREAM_PRESSURE="):
            line = f"FREESTREAM_PRESSURE={pressure}\n"
        elif line.startswith("FREESTREAM_TEMPERATURE="):
            line = f"FREESTREAM_TEMPERATURE={temperature}\n"
        updated_lines.append(line)

        # Aggiungere la riga SIDESLIP_ANGLE subito dopo AOA, se non già presente
        if line.startswith("AOA=") and not added_sideslip_angle:
            updated_lines.append(f"SIDESLIP_ANGLE={row['angleOfSideslip']}\n")
            added_sideslip_angle = True

    # 5. Salvare il file aggiornato
    with open(config_path, "w") as file:
        file.writelines(updated_lines)

    # 6. Eseguire il comando per SU2 nella cartella specifica
    os.system(f"mpirun -np 8  /soft/SU2/bin/SU2_CFD {config_path}")

    # 7. Controllare il codice di ritorno per assicurarsi che l'esecuzione sia completata
    # if process.returncode != 0:
    #     print(f"Errore durante l'esecuzione di SU2 per Case{index}")
    #     break

print("Esecuzione completata.")


# data = pd.read_csv(dataset_path)

# # Estrai le altitudini
# altitudes = data["altitude"]

# # Calcola pressione e temperatura per ogni altitudine
# pressures = []
# temperatures = []

# for altitude in altitudes:
#     atmosphere = Atmosphere(altitude)
#     pressures.append(atmosphere.pressure[0])  # Pressione (Pa)
#     temperatures.append(atmosphere.temperature[0])  # Temperatura (K)

# # Stampa i risultati
# print("Altitudini:", altitudes.tolist())
# print("Pressioni (Pa):", pressures)
# print("Temperature (K):", temperatures)
