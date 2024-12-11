import os
import pandas as pd
from tixi3.tixi3wrapper import Tixi3


def create_or_update_element(tixi, xpath, value):
    """
    Crea un elemento nel file CPACS se non esiste, altrimenti lo aggiorna.

    Parameters:
        tixi (Tixi3): Oggetto Tixi per manipolare il file CPACS.
        xpath (str): Percorso XPath dell'elemento.
        value (str | int | float | list | tuple): Valore da assegnare all'elemento.
    """
    parent_xpath = "/".join(xpath.split("/")[:-1])
    element_name = xpath.split("/")[-1]

    # Assicurati che il valore sia una stringa valida
    if isinstance(value, (list, tuple)):
        value = "; ".join(map(str, value))  # Converti liste o tuple in stringhe separate da ";"
    else:
        value = str(value)  # Converti tutto il resto in stringa

    # Crea o aggiorna l'elemento
    if not tixi.checkElement(xpath):
        tixi.createElement(parent_xpath, element_name)
    tixi.updateTextElement(xpath, value)


def update_values(
    tixi, aero_parameters, euler_mesh_params, rans_mesh_params, extra_mesh_params, su2_params
):
    """
    Aggiorna o crea i valori nel file CPACS per parametri aerodinamici, della mesh e SU2.

    Parameters:
        tixi (Tixi3): Oggetto Tixi per manipolare il file CPACS.
        aero_parameters (tuple): Parametri aerodinamici (altitude, mach, aoa, aos).
        mesh_params (list[dict]): Lista di dizionari con path e value per i parametri principali della mesh.
        extra_mesh_params (list[dict]): Lista di dizionari con path e value per i parametri extra della mesh.
        su2_params (list[dict]): Lista di dizionari con path e value per i parametri SU2.
    """
    altitude, mach, aoa, aos = aero_parameters

    # Aggiorna i parametri aerodinamici
    create_or_update_element(
        tixi,
        "/cpacs/vehicles/aircraft/model/analyses/aeroPerformance/aeroMap[@uID='aeromap_empty']/aeroPerformanceMap/altitude",
        "; ".join(map(str, altitude)),
    )
    create_or_update_element(
        tixi,
        "/cpacs/vehicles/aircraft/model/analyses/aeroPerformance/aeroMap[@uID='aeromap_empty']/aeroPerformanceMap/machNumber",
        "; ".join(map(str, mach)),
    )
    create_or_update_element(
        tixi,
        "/cpacs/vehicles/aircraft/model/analyses/aeroPerformance/aeroMap[@uID='aeromap_empty']/aeroPerformanceMap/angleOfAttack",
        "; ".join(map(str, aoa)),
    )
    create_or_update_element(
        tixi,
        "/cpacs/vehicles/aircraft/model/analyses/aeroPerformance/aeroMap[@uID='aeromap_empty']/aeroPerformanceMap/angleOfSideslip",
        "; ".join(map(str, aos)),
    )

    # Aggiorna i parametri della mesh
    for param in euler_mesh_params:
        create_or_update_element(tixi, param["path"], param["value"])

    for param in rans_mesh_params:
        create_or_update_element(tixi, param["path"], param["value"])

    # Aggiorna i parametri extra della mesh
    for param in extra_mesh_params:
        create_or_update_element(tixi, param["path"], param["value"])

    # Aggiorna i parametri SU2
    for param in su2_params:
        create_or_update_element(tixi, param["path"], param["value"])


# Percorsi principali
input_file = "/wrk/Gronda/labAR/EULER/00_symmetry/takeoff100/NewTool.xml"
dataset_path = "//wrk/Gronda/labAR/EULER/00_symmetry/datasets/takeoff.csv"
directory_path = "/wrk/Gronda/labAR/EULER/00_symmetry/takeoff100"

# Leggi il dataset
data = pd.read_csv(dataset_path)

# Estrai i dati come array
altitude = data["altitude"].values
mach = data["machNumber"].values
aoa = data["angleOfAttack"].values
aos = data["angleOfSideslip"].values

#
type_mesh = ["Euler"]  # type_mesh
symmetry = ["True"]  # symmetry


# Parametri mesh Euleriana
farfield_factors = [10.0]
mesh_farfields = [3.0]
fuselage_factors = [6]
wing_factors = [26.0]
n_power_factors = [2.0]
n_power_fields = [1.0]
le_te_layers = [14]
refine_truncated = ["False"]  # refine_truncated
auto_refine = ["True"]  # auto_refine

# parametri mesh RANS
number_layer = [10]
height_first_layer = [3.0]
max_thickness_layer = [100.0]
growth_ratio = [1.2]
growth_factor = [1.4]
feature_angle = [40]
surface_mesh_size = [5.0]  # gmshOptionsmin_max_mesh_factor
surface_max_size = [0.0008]  # DA CREARE:gmshOptionsmax_mesh_factor
surface_min_size = [0.0002]  # DA CREARE: gmshOptionsmin_mesh_factor

# SU2
config_type = type_mesh
aeromap = ["aeromap_empty"]  # aeroMapUID
derivatives = ["False"]  # calculateDampingDerivatives
rotation = [1.0]  # rotationRate
control_surfaces = ["False"]  # calculateControlSurfacesDeflections
cpu = [9]  # nbCPU
iters = [800]  # maxIter
cfl_adption = ["True"]  # value
cflAdFactorDown = [0.5]  # factor_down
cflAdFactorUp = [1.5]  # factor_up
cflMinValue = [0.5]  # min
cflMaxValue = [100.0]  # max
cfl_value = [1]  # cfl value
multiGrid = [3]  # multigridLevel
wettedArea = ["True"]  # updateWettedArea
extraLoads = ["False"]  # extractLoads


# Parametri della mesh euleriana
euler_mesh_params = [
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/farfield_factor",
        "value": farfield_factors,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/mesh_size/farfield",
        "value": mesh_farfields,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/mesh_size/fuselage/factor",
        "value": fuselage_factors,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/mesh_size/wings/factor",
        "value": wing_factors,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/n_power_factor",
        "value": n_power_factors,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/n_power_field",
        "value": n_power_fields,
    },
    {"path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/number_layer", "value": le_te_layers},
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/refine_truncated",
        "value": refine_truncated,
    },
    {"path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/auto_refine", "value": auto_refine},
]

# Parametri della mesh rans
rans_mesh_params = [
    {"path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/number_layer", "value": number_layer},
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/height_first_layer",
        "value": height_first_layer,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/max_thickness_layer",
        "value": max_thickness_layer,
    },
    {"path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/growth_ratio", "value": growth_ratio},
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/growth_factor",
        "value": growth_factor,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptions/feature_angle",
        "value": feature_angle,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptionsmin_max_mesh_factor",
        "value": surface_mesh_size,
    },
]
# Parametri extra della mesh
extra_mesh_params = [
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptionsmin_mesh_factor",
        "value": surface_min_size,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/mesh/gmshOptionsmax_mesh_factor",
        "value": surface_max_size,
    },
]

# Parametri SU2
su2_params = [
    {"path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/aeroMapUID", "value": aeromap},
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/options/calculateDampingDerivatives",
        "value": derivatives,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/options/config_type",
        "value": config_type,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/options/rotationRate",
        "value": rotation,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/options/calculateControlSurfacesDeflections",
        "value": control_surfaces,
    },
    {"path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/settings/nbCPU", "value": cpu},
    {"path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/settings/maxIter", "value": iters},
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/settings/cflNumber/adaptation/value",
        "value": cfl_adption,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/settings/cflNumber/adaptation/factor_down",
        "value": cflAdFactorDown,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/settings/cflNumber/adaptation/factor_up",
        "value": cflAdFactorUp,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/settings/cflNumber/adaptation/min",
        "value": cflMinValue,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/settings/cflNumber/adaptation/max",
        "value": cflMaxValue,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/settings/cflNumber/value",
        "value": cfl_value,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/settings/multigridLevel",
        "value": multiGrid,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/results/updateWettedArea",
        "value": wettedArea,
    },
    {
        "path": "/cpacs/toolspecific/CEASIOMpy/aerodynamics/su2/results/extractLoads",
        "value": extraLoads,
    },
]

# Apri e aggiorna il file CPACS
tixi = Tixi3()
tixi.open(input_file)
update_values(
    tixi,
    (altitude, mach, aoa, aos),
    euler_mesh_params,
    rans_mesh_params,
    extra_mesh_params,
    su2_params,
)
tixi.save(input_file)
tixi.close()


print("CPACS updated, running CEASIOMpy...")
# Combine both commands in a single shell session
command = (
    f"cd {os.path.abspath(directory_path)} && "
    f"ceasiompy_run -m {os.path.abspath(input_file)} CPACS2GMSH SU2Run"
)
os.system(command)
