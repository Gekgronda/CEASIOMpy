import os
import pandas as pd
from tixi3.tixi3wrapper import Tixi3


def update_values(tixi, aero_parameters, mesh_parameters):
    """
    Aggiorna i valori nel file CPACS per parametri aerodinamici e della mesh.

    Parameters:
        tixi (Tixi3): Oggetto Tixi per manipolare il file CPACS.
        aero_parameters (tuple): Parametri aerodinamici (altitude, mach, aoa, aos).
        mesh_parameters (tuple): Parametri della mesh.
    """
    altitude, mach, aoa, aos = aero_parameters
    (
        farfield_factor,
        farfield,
        fuselage_factor,
        wing_factor,
        n_power,
        n_power_field,
        le_te_layers,
        feature_angle,
    ) = mesh_parameters

    # Aggiorna i parametri aerodinamici
    tixi.updateTextElement(
        "/cpacs/vehicles/aircraft/model/analyses/aeroPerformance/aeroMap[@uID='new']/aeroPerformanceMap/altitude",
        "; ".join(map(str, altitude)),
    )
    tixi.updateTextElement(
        "/cpacs/vehicles/aircraft/model/analyses/aeroPerformance/aeroMap[@uID='new']/aeroPerformanceMap/machNumber",
        "; ".join(map(str, mach)),
    )
    tixi.updateTextElement(
        "/cpacs/vehicles/aircraft/model/analyses/aeroPerformance/aeroMap[@uID='new']/aeroPerformanceMap/angleOfAttack",
        "; ".join(map(str, aoa)),
    )
    tixi.updateTextElement(
        "/cpacs/vehicles/aircraft/model/analyses/aeroPerformance/aeroMap[@uID='new']/aeroPerformanceMap/angleOfSideslip",
        "; ".join(map(str, aos)),
    )

    # Aggiorna i parametri della mesh
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
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/n_power_factor",
        str(n_power),
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/n_power_field",
        str(n_power_field),
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/number_layer",
        str(le_te_layers),
    )
    tixi.updateTextElement(
        "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/feature_angle",
        str(feature_angle),
    )


# Percorsi principali
input_file = "/wrk/Gronda/labAR/EULER/00_symmetry/takeoff100/ToolOutput.xml"
dataset_path = "/wrk/Gronda/labAR/EULER/00_symmetry/datasets/takeoff.csv"

# Leggi il dataset
data = pd.read_csv(dataset_path)

# Estrai i dati come array
altitude = data["altitude"].values
mach = data["machNumber"].values
aoa = data["angleOfAttack"].values
aos = data["angleOfSideslip"].values

# Parametri aggiornati
farfield_factors = [10.0]
mesh_farfields = [5.0]
fuselage_factors = [6]
wing_factors = [15.0]
n_power_factors = [2.0]
n_power_fields = [0.9]
le_te_layers = [10]
feature_angle = [40]

# Usa il primo set di parametri come esempio (adatta per iterazioni o altri scopi)
mesh_parameters = (
    farfield_factors[0],
    mesh_farfields[0],
    fuselage_factors[0],
    wing_factors[0],
    n_power_factors[0],
    n_power_fields[0],
    le_te_layers[0],
    feature_angle[0],
)

# Apri e aggiorna il file CPACS
tixi = Tixi3()
tixi.open(input_file)
update_values(tixi, (altitude, mach, aoa, aos), mesh_parameters)
tixi.save(input_file)
tixi.close()
