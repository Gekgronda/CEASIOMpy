import os
import subprocess

# Codice che serve ad aggiornare il numero di iterazioni dal file config e rilancia le simulazioni (cambia anche nome
# del file di riavvio e impostazioni Config
# Esiste una versione semplificata chiamata "run.py" nella cartella RANS


# Cartella principale
base_directory = "/wrk/Gronda/labAR/RANS/"

# Lista delle directory ordinate alfabeticamente


directory_list = sorted(
    [
        os.path.join(base_directory, dir_name)
        for dir_name in os.listdir(base_directory)
        if dir_name.startswith("Case") and os.path.isdir(os.path.join(base_directory, dir_name))
    ]
)

for directory_path in directory_list:
    print(f"Processing directory: {directory_path}")

    # Percorso del file di configurazione
    config_file_path = os.path.join(directory_path, "Config_labAR_jack.cfg")
    restart_file_path = os.path.join(directory_path, "restart_flow.dat")
    solution_file_path = os.path.join(directory_path, "solution_flow.dat")

    # Controlla se il file di configurazione esiste
    if not os.path.exists(config_file_path):
        print(f"File {config_file_path} non trovato. Procedo con la prossima directory.")
        continue

    # Controlla se il file "restart_flow.dat" esiste e rinominalo
    if os.path.exists(restart_file_path):
        try:
            os.rename(restart_file_path, solution_file_path)
            print(f"Rinominato {restart_file_path} in {solution_file_path}")
        except OSError as e:
            print(f"Errore durante il rinominare {restart_file_path}: {e}")
            continue
    else:
        print(f"File {restart_file_path} non trovato in {directory_path}. Procedo comunque.")

    # Legge il file di configurazione
    with open(config_file_path, "r") as config_file:
        config_content = config_file.read()

    # Modifica il parametro ITER e RESTART_SOL
    lines = config_content.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()

        # Modifica ITER se necessario
        if line.startswith("ITER ="):
            try:
                iter_value = int(line.split("=")[1].strip())
                if iter_value <= 1000:
                    lines[i] = "ITER = 2000"
                    print(f"Modificato ITER a 2000 in {config_file_path}")
            except (ValueError, IndexError):
                print(f"Errore nel parsing di ITER: {line}")
                continue

        # Modifica RESTART_SOL
        if line.startswith("RESTART_SOL="):
            if "NO" in line:
                lines[i] = "RESTART_SOL= YES"
                print(f"Modificato RESTART_SOL a YES in {config_file_path}")

    # Sovrascrive il file di configurazione con le modifiche
    with open(config_file_path, "w") as config_file:
        config_file.write("\n".join(lines))

    # Cambia la directory di lavoro corrente alla directory della configurazione
    os.chdir(directory_path)

    # Esegui il comando per il programma SU2
    command = f"mpirun -np 8 /soft/SU2/bin/SU2_CFD {config_file_path}"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"Esecuzione completata per {directory_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione del comando in {directory_path}:\n{e.stderr}")

    # Ora puoi aggiungere ulteriori controlli o elaborazioni specifiche per questa directory

    # Ripeti il ciclo per le altre directory
