import os
import shutil

# Percorso del file ConfigCFD base
original_config_path = "/home/cfse/Stage_Gronda/CEASIOMpy/WKDIR/Workflow_067/Results/SU2/Case01_alt1000.0_mach0.4_aoa5.0_aos3.0/ConfigCFD.cfg"

# Ciclo che varia il numero del workflow
for i in range(89, 90):
    # Percorso esistente
    base_destination_path = f"/home/cfse/Stage_Gronda/CEASIOMpy/WKDIR/Workflow_{i:03d}/Results/"

    # Percorso che va creato e in cui salvare il file
    destination_directory = os.path.join(
        base_destination_path, "SU2/Case01_alt1000.0_mach0.4_aoa5.0_aos3.0/"
    )
    destination_file_path = os.path.join(destination_directory, "ConfigCFD.cfg")

    # Crea la directory di destinazione se non esiste
    os.makedirs(destination_directory, exist_ok=True)

    # Copia il file nella nuova directory
    shutil.copy(original_config_path, destination_file_path)

    # Percorso del file di configurazione nella nuova cartella
    config_path = os.path.join(destination_directory, "ConfigCFD.cfg")

    # Leggi il file di configurazione e modifica la linea MESH_FILENAME
    with open(config_path, "r") as config_file:
        config_content = config_file.read()

    lines = config_content.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("MESH_FILENAME"):
            lines[index] = (
                f"MESH_FILENAME = /users/disk19/cfse/Stage_Gronda/CEASIOMpy/WKDIR/Workflow_{i:03d}/Results/GMSH/mesh.su2"
            )

    # Sovrascrivi il file di configurazione con le modifiche
    with open(config_path, "w") as config_file:
        config_file.write("\n".join(lines))

    # Cambia la directory di lavoro corrente alla directory corrente del workflow
    directory_path = f"/home/cfse/Stage_Gronda/CEASIOMpy/WKDIR/Workflow_{i:03d}/Results/SU2/Case01_alt1000.0_mach0.4_aoa5.0_aos3.0/"
    os.chdir(directory_path)

    # Esegui il calcolo con mpirun e SU2_CFD
    os.system(f"mpirun -np 8 /soft/SU2/bin/SU2_CFD {config_path}")

    print(f"Completato per Workflow {i:03d}")
