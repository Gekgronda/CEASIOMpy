import pandas as pd
import numpy as np
import os
import subprocess
from tixi3.tixi3wrapper import Tixi3
from MFSM_Func import (
    choose_fidelity_workflow,
    find_and_save_file,
    doe_workflow,
    get_user_inputs,
    load_and_split_data,
    plot_distributions,
    normalize_data,
    lh_sampling,
    save_to_csv,
    get_latest_workflow,
    extract_coefficients_from_AVL,
    avl_workflow,
    extract_coefficients_from_SU2,
    append_to_new_csv,
    validate_inputs,
    test_training_data,
    MF_CoKriging,
    MF_Kriging,
    Kriging,
    predict_model,
    predict_mf_model,
    predict_mfco_model,
    plot_validation,
    plot_results,
    plot_doe,
    plot_coefficent_alpha_for_mach,
    plot_response_surface,
    save_model,
    sm_workflow,
)


from prove import su2_workflow
from sklearn.preprocessing import MinMaxScaler
from smt.utils.misc import compute_rms_error

# TO DO
# - AGGIUNGERE IL FATTO CHE LHS NON FACCIA DOPPIONI
# - SE METTI UN SOLO VALORE PER LHS NON FUNZIONA
# - CONFRONTO DOE CON DOMINIO FISICO
# - CONTROLLARE REFERENCE VALUES
# - FRAZIONE PER LE RANS
# -

# --------------------------------------------------------

## CPACS FILE and PATHS
input_cpacs_name = "D150_simple.xml"
cpacs_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/test_files/CPACSfiles"
directory_path = "/wrk/Gronda/validazione"
default_doe_path = "/wrk/Gronda/validazione/LHS_dataset_default.csv"
default_first_kriging_dataset_path = "/wrk/Gronda/validazione/LHS_dataset_TRAIN_default.csv"
default_second_kriging_dataset_path = (
    "/wrk/Gronda/labAR/prove_codice/EULER_dataset_TRAIN_default.csv"
)
# input_cpacs_name = "labARscaled.xml"
# directory_path = "/wrk/Gronda/labAR/prove_codice"
# cpacs_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/test_files/CPACSfiles"

# --------------------------------------------------------

## LHS OPTIONS
output_filename_lhs = "LHS_dataset.csv"


# --------------------------------------------------------

## SAVING OPTIONS
base_model_name = "surrogate_model_prova1"
model_extension = ".pkl"

# --------------------------------------------------------

## AVL OPTIONS
# Aeromap name
aeromap_avl_name = "avl_aeromap"
aeromap_avl_uid = aeromap_avl_name

# Parameters from avl code, for the moment update from here
integrate_fusolage = "False"
distribution = "cosine"
nchordwise = 20
nspanwise = 50
savePlots = "True"

avl_parameters = {
    "IntegrateFuselage": integrate_fusolage,
    "VortexDistribution/Distribution": distribution,
    "VortexDistribution/Nchordwise": int(nchordwise),
    "VortexDistribution/Nspanwise": int(nspanwise),
    "SavePlots": savePlots,
}


# --------------------------------------------------------

## FIRST KRIGING OPTIONS

theta1 = [0.01]
corr1 = "matern32"
poly1 = "constant"
selected_mach = [0.2, 0.5, 0.7]
altitude_for_response_surface = 10000
aos_for_response_surface = 0
fraction_of_new_samples = 5

# Name of the output dataset
output_filename_euler = "EULER_dataset.csv"


# --------------------------------------------------------

## EULER OPTIONS
# Aeromap name
aeromap_euler_uid = "euler_aeromap"
aeromap_euler_name = "euler_aeromap"

# General parameters
open_gmsh_gui = ["False"]
export_propellers = ["False"]
type_mesh = ["Euler"]  # type_mesh
symmetry = ["True"]  # symmetry

# Parametri mesh Euleriana
farfield_factor = [10.0]
mesh_farfield = [3.0]
fuselage_factor = [6]
wing_factor = [26.0]
engines = [0.23]
propellers = [0.23]
n_power_factor = [2.0]
n_power_field = [1.0]
le_te_layers = [14]
refine_truncated = ["False"]  # refine_truncated
auto_refine = ["True"]  # auto_refine

# SU2
config_type = type_mesh
derivatives = ["False"]  # calculateDampingDerivatives
rotation = [1.0]  # rotationRate
control_surfaces = ["False"]  # calculateControlSurfacesDeflections
includeActuatorDisk = ["None;None"]
cpu = [9]  # nbCPU
iters = [1]  # maxIter
cfl_adption = ["True"]  # value
cflAdFactorDown = [0.5]  # factor_down
cflAdFactorUp = [1.5]  # factor_up
cflMinValue = [0.5]  # min
cflMaxValue = [100.0]  # max
cfl_value = [1]  # cfl value
multiGrid = [3]  # multigridLevel
wettedArea = ["True"]  # updateWettedArea
extraLoads = ["False"]  # extractLoads

# Conversion of lists of dictionaries to simple dictionaries
# Common mesh parameters
common_mesh_params = {
    "open_gui": open_gmsh_gui,
    "type_mesh": type_mesh,
    "symmetry": symmetry,
    "exportPropellers": export_propellers,
}

# Euler mesh parameters
euler_mesh_params = {
    "farfield_factor": farfield_factor,
    "mesh_size/farfield": mesh_farfield,
    "mesh_size/fuselage/factor": fuselage_factor,
    "mesh_size/wings/factor": wing_factor,
    "mesh_size/engines": engines,
    "mesh_size/propellers": propellers,
    "n_power_factor": n_power_factor,
    "n_power_field": n_power_field,
    "number_layer": le_te_layers,
    "refine_truncated": refine_truncated,
    "auto_refine": auto_refine,
}

# SU2 parameters
su2_params = {
    "aeroMapUID": aeromap_euler_name,
    "options/calculateDampingDerivatives": derivatives,
    "options/config_type": config_type,
    "options/rotationRate": rotation,
    "options/calculateControlSurfacesDeflections": control_surfaces,
    "options/includeActuatorDisk": includeActuatorDisk,
    "settings/nbCPU": cpu,
    "settings/maxIter": iters,
    "settings/cflNumber/value": cfl_value,
    "settings/cflNumber/adaptation/value": cfl_adption,
    "settings/cflNumber/adaptation/factor_down": cflAdFactorDown,
    "settings/cflNumber/adaptation/factor_up": cflAdFactorUp,
    "settings/cflNumber/adaptation/min": cflMinValue,
    "settings/cflNumber/adaptation/max": cflMaxValue,
    "settings/multigridLevel": multiGrid,
    "results/updateWettedArea": wettedArea,
    "results/extractLoads": extraLoads,
}


# --------------------------------------------------------

## M-F KRIGING OPTIONS

theta1 = [0.01]
corr1 = "matern32"
poly1 = "constant"
selected_mach = [0.2, 0.5, 0.7]
altitude_for_response_surface = 10000
aos_for_response_surface = 0
fraction_of_new_samples2 = 5

# Name of the output dataset
output_filename_rans = "RANS_dataset.csv"


# --------------------------------------------------------


# ===== 0. CPACS =====

find_and_save_file(input_cpacs_name, cpacs_directory, directory_path)
input_cpacs_path = os.path.join(directory_path, input_cpacs_name)

print(f"CPACS name: {input_cpacs_name}")


# ===== 1. DOE =====

# Select how many fidelity level
fidelity_level, workflow = choose_fidelity_workflow()

try:
    samples, ranges, processed_samples, n_samples, full_path1 = doe_workflow(
        default_doe_path, directory_path, output_filename_lhs
    )

    print("Design of Experiment (DoE) ready:")
    print(samples)  # Mostra le prime righe del DoE generato o caricato

    if ranges is not None and processed_samples is not None and n_samples is not None:
        if len(ranges) > 0 and processed_samples.any():
            print("\nRanges:")
            print(ranges)
            print("\nProcessed Samples:")
            print(processed_samples)
            print("\nNumber of samples:")
            print(n_samples)

except FileNotFoundError as e:
    print(f"Error: {e}")

input("Press ENTER to continue....")


# ===== 2. AVL =====

first_kriging_dataset_path = avl_workflow(
    input_cpacs_path,
    directory_path,
    default_first_kriging_dataset_path,
    full_path1,
    samples,
    aeromap_avl_uid,
    aeromap_avl_name,
    avl_parameters,
)

input("Press ENTER to continue....")


# ====== 3. FIRST KRIGING MODEL =====

new_aeromap1, full_path2, which_coefficent1, top_n_X_test1, model1, rms1, X_train1, y_train1 = (
    sm_workflow(
        first_kriging_dataset_path,
        directory_path,
        output_filename_euler,
        theta1,
        corr1,
        poly1,
        selected_mach,
        altitude_for_response_surface,
        aos_for_response_surface,
        n_samples,
        fidelity_level,
        fraction_of_new_samples,
        ranges,
        processed_samples,
        base_model_name=base_model_name,
        model_extension=model_extension,
    )
)

input("Press ENTER to continue....")

# ====== 4. EULER ======

second_kriging_dataset_path = su2_workflow(
    input_cpacs_path,
    directory_path,
    default_second_kriging_dataset_path,
    full_path2,
    new_aeromap1,
    aeromap_euler_uid,
    aeromap_euler_name,
    common_mesh_params,
    euler_mesh_params,
    su2_params,
)

input("Press ENTER to continue....")

# ====== 5. MF KRIGING ======

new_aeromap2, full_path3, which_coefficent2, top_n_X_test2, model2, rms2, X_train2, y_train2 = (
    sm_workflow(
        second_kriging_dataset_path,
        directory_path,
        output_filename_rans,
        theta2,
        corr2,
        poly2,
        selected_mach,
        altitude_for_response_surface,
        aos_for_response_surface,
        n_samples,
        fidelity_level,
        fraction_of_new_samples=fraction_of_new_samples2,
        ranges=ranges,
        processed_samples=processed_samples,
        X_train_LF=X_train1,
        y_train_LF=y_train1,
        base_model_name=base_model_name,
        model_extension=model_extension,
    )
)
