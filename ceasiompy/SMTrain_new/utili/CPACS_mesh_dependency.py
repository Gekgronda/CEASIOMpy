import os
import csv
from tixi3.tixi3wrapper import Tixi3

# Parametri aggiornati
farfield_factors = [10.0, 15.0]
mesh_farfields = [5.0, 7.0]
fuselage_factors = [6]
wing_factors = [15.0]
n_power_factors = [2.0, 2.2]
n_power_fields = [0.9, 1.0]
le_te_layers = [10]
# feature_angle = [40]

# Specifica manuale delle combinazioni desiderate
# Ogni tupla rappresenta una combinazione (farfield_factor, mesh_farfield, fuselage_factor, wing_factor, n_power_factor, n_power_field)
manual_combinations = [
    (8.0, 20.0, 4, 2, 2.0, 1.0, 4),
    (10.0, 20.0, 4, 2, 2.0, 1.0, 4),
    (12.0, 20.0, 4, 2, 2.0, 1.0, 4),
    (8.0, 15.0, 4, 2, 2.0, 1.0, 4),
    (8.0, 25.0, 4, 2, 2.0, 1.0, 4),
    (8.0, 20.0, 5, 2, 2.0, 1.0, 4),
    (8.0, 20.0, 6, 2, 2.0, 1.0, 4),
    (8.0, 20.0, 4, 3, 2.0, 1.0, 4),
    (8.0, 20.0, 4, 4, 2.0, 1.0, 4),
    (8.0, 20.0, 4, 2, 2.0, 1.0, 6),
    (8.0, 20.0, 4, 2, 2.0, 1.0, 8),
]

# Generazione automatica delle combinazioni se manual_combinations è vuoto
if manual_combinations:
    parameter_combinations = manual_combinations
    print("Usando combinazioni specificate manualmente.")
else:
    parameter_combinations = [
        (farfield, mesh, fuselage, wing, n_power, n_field, refine_factor)
        for farfield in farfield_factors
        for mesh in mesh_farfields
        for fuselage in fuselage_factors
        for wing in wing_factors
        for n_power in n_power_factors
        for n_field in n_power_fields
        for refine_factor in le_te_layers
    ]
    print("Usando tutte le combinazioni possibili.")

# Verifica del numero totale di combinazioni
print(f"Numero totale di combinazioni: {len(parameter_combinations)}")

# Percorsi principali
input_file = "/wrk/Gronda/d150/mesh_dependecy_euler/D150.xml"

output_csv = "/wrk/Gronda/d150/mesh_dependecy_euler/mesh_dep-22-1.csv"

# Esporta tutte le combinazioni in un file CSV
with open(output_csv, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    # Scrive l'intestazione
    csvwriter.writerow(
        [
            "farfield_factor",
            "mesh_farfield",
            "fuselage_factor",
            "wing_factor",
            "n_power_factor",
            "n_power_field",
            "le_te_layers",
        ]
    )
    # Scrive tutte le combinazioni
    csvwriter.writerows(parameter_combinations)

print(f"File CSV con le combinazioni creato: {output_csv}")


# Funzione per aggiornare i valori nel file XML
def update_values(tixi, parameters):
    (
        farfield_factor,
        farfield,
        fuselage_factor,
        wing_factor,
        n_power,
        n_power_field,
        le_te_layers,
    ) = parameters
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/farfield_factor",
        str(farfield_factor),
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/mesh_size/farfield",
        str(farfield),
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/mesh_size/fuselage/factor",
        str(fuselage_factor),
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/mesh_size/wings/factor",
        str(wing_factor),
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/n_power_factor", str(n_power)
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/n_power_field",
        str(n_power_field),
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/refine_factor",
        str(le_te_layers),
    )


# Loop sulle combinazioni
for params in parameter_combinations:
    # Controlla che il file esista
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Il file {input_file} non esiste. Verifica il percorso.")

    # Carica il file XML
    tixi = Tixi3()
    tixi.open(input_file)

    # Aggiorna i valori dei parametri
    update_values(tixi, params)

    # Salva il file aggiornato
    tixi.save(input_file)
    tixi.close()

    # Esegui il comando
    command = f"ceasiompy_run -m {os.path.abspath(input_file)} CPACS2GMSH"  # SU2Run"
    os.system(command)

print("Pipeline completata!")
