import os
from itertools import product
from tixi3.tixi3wrapper import Tixi3

# Parametri aggiornati
farfield_factors = [10.0, 15.0]
mesh_farfields = [5.0, 7.0]
fuselage_factors = [6.0]
wing_factors = [15.0]
n_power_factors = [2.0, 2.2]
n_power_fields = [1.0]

# Combinazioni dei parametri
parameter_combinations = list(
    product(
        farfield_factors,
        mesh_farfields,
        fuselage_factors,
        wing_factors,
        n_power_factors,
        n_power_fields,
    )
)

# Verifica del numero totale di combinazioni
print(f"Numero totale di combinazioni: {len(parameter_combinations)}")

# Percorsi principali
input_file = "/wrk/Gronda/labAR/EULER/mesh_dependency/nuovo_cpacs_handling/00_ToolInput.xml"


# Funzione per aggiornare i valori nel file XML
def update_values(tixi, parameters):
    farfield_factor, farfield, fuselage_factor, wing_factor, n_power, n_power_field = parameters
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/farfield_factor", str(farfield_factor)
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/mesh_size/farfield", str(farfield)
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/mesh_size/fuselage/factor",
        str(fuselage_factor),
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/mesh_size/wings/factor", str(wing_factor)
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/n_power_factor", str(n_power)
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/n_power_field", str(n_power_field)
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
    command = f"ceasiompy_run -m {os.path.abspath(input_file)} CPACS2GMSH SU2Run"
    os.system(command)

print("Pipeline completata!")
