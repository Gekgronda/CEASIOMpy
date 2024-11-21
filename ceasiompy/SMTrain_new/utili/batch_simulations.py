import os

# Lista delle directory da controllare
# Cartella principale
base_directory = "/home/benedetti/CEASIOM/CEASIOMpy/WKDIR/"

# Lista delle directory da controllare
directory_list = [
    base_directory + f"Workflow_{i:03}/Results/SU2/Case00_alt0.0_mach0.3_aoa0.0_aos0.0/"
    for i in range(32, 50)
]


for directory_path in directory_list:
    # Aggiungi il percorso del file Config.cfg specifico per questa directory
    config_file_path = os.path.join(directory_path, "ConfigCFD.cfg")

    # Controlla se il file "hosts" esiste nella directory
    hosts_file_path = os.path.join(directory_path, "hosts")
    if not os.path.exists(hosts_file_path):
        # Crea il file "hosts" e scrivi i contenuti
        with open(hosts_file_path, "w") as hosts_file:
            hosts_file.write("cfs9:8\n")
            hosts_file.write("cfs12:8\n")
            hosts_file.write("cfs10:8\n")
            hosts_file.write("\n")

    # Leggi il contenuto del file Config.cfg
    with open(config_file_path, "r") as config_file:
        config_content = config_file.read()

    # Trova la riga INNER_ITER e ottieni il suo valore
    lines = config_content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("INNER_ITER"):
            try:
                inner_iter = int(line.split("=")[1])
                # Se INNER_ITER è minore di 1000, impostalo a 1000
                if inner_iter < 2500:
                    lines[i] = "INNER_ITER = 2500"
            except (ValueError, IndexError):
                pass  # Gestione degli errori se INNER_ITER non è un numero o il formato è diverso

    # Sovrascrivi il file Config.cfg con le modifiche
    with open(config_file_path, "w") as config_file:
        config_file.write("\n".join(lines))

     # Cambia la directory di lavoro corrente alla directory in cui si trova il file ConfigCFD.cfg
    os.chdir(directory_path)

    # Esegui il tuo calcolo (in questo caso solo una simulazione di esempio) utilizzando il file "hosts"
    os.system(f"mpirun -np 24 -machinefile {hosts_file_path} /soft/SU2/bin/SU2_CFD {config_file_path}")

    # Ora puoi aggiungere ulteriori controlli o elaborazioni specifiche per questa directory

    # Ripeti il ciclo per le altre directory

    
