import pandas as pd
import numpy as np
import os
import subprocess
from tixi3.tixi3wrapper import Tixi3
from MFSM_Func import (
    workflow,
    find_and_save_file,
    doe_workflow,
    get_user_inputs,
    load_and_split_data,
    plot_distributions,
    plot_doe,
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
    su2_workflow,
)

from sklearn.preprocessing import MinMaxScaler
from smt.utils.misc import compute_rms_error

# TO DO
# - AGGIUNGERE IL FATTO CHE LHS NON FACCIA DOPPIONI
# - SE METTI UN SOLO VALORE PER LHS NON FUNZIONA
# - CONFRONTO DOE CON DOMINIO FISICO da migliorare
# - CONTROLLARE REFERENCE VALUES
# - FRAZIONE PER LE RANS
# - I DEFAULT VANNO TOLTI E ANCHE DALLE FUNZIONI!


# --------------------------------------------------------

## CPACS FILE and PATHS
input_cpacs_name = "D150_simple.xml"
cpacs_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/test_files/CPACSfiles"
directory_path = "/wrk/Gronda/validazione/mengmeng"
default_doe_path = "/wrk/Gronda/validazione/mengmeng/AVL.csv"
default_first_kriging_dataset_path = "/wrk/Gronda/validazione/mengmeng/AVL_TRAIN.csv"
default_second_kriging_dataset_path = "/wrk/Gronda/validazione/mengmeng/EULER_TRAIN_N.csv"
default_third_kriging_dataset_path = None
# input_cpacs_name = "labARscaled.xml"
# directory_path = "/wrk/Gronda/labAR/prove_codice"
# cpacs_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/test_files/CPACSfiles"

# --------------------------------------------------------

## LHS OPTIONS
output_filename_lhs = "LHS_dataset.csv"

# pysical domain limits
p1, p2, p3, p4 = [-5, 0.5], [-3, 0.9], [4, 0.9], [15, 0.5]
# p1, p2, p3, p4 = [-5, 0.1], [-3, 0.9], [4, 0.9], [15, 0.1]


physical_domain_limits = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}


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
corr1 = "squar_exp"
poly1 = "linear"
selected_mach_for_aoa_plot = [0.5, 0.8, 0.9]
altitude_for_response_surface = 10000
aos_for_response_surface = 0
fraction_of_new_samples = 10

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

# 10 3 6 26 0.23 0.23 2 1 14
# (8.0, 20.0, 4, 2, 2.0, 1.0, 4)

# Parametri mesh Euleriana
farfield_factor = [8.0]
mesh_farfield = [20.0]
fuselage_factor = [4]
wing_factor = [2]
engines = [0.23]
propellers = [0.23]
n_power_factor = [2.0]
n_power_field = [1.0]
le_te_layers = [4]
refine_truncated = ["False"]  # refine_truncated
auto_refine = ["True"]  # auto_refine

# SU2
config_type = type_mesh
derivatives = ["False"]  # calculateDampingDerivatives
rotation = [1.0]  # rotationRate
control_surfaces = ["False"]  # calculateControlSurfacesDeflections
includeActuatorDisk = ["None;None"]
cpu = [9]  # nbCPU
iters = [1500]  # maxIter
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
euler_common_mesh_params = {
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
    "refine_factor": le_te_layers,
    "refine_truncated": refine_truncated,
    "auto_refine": auto_refine,
}

# SU2 parameters
euler_su2_params = {
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

theta2 = [0.001]
corr2 = "squar_exp"
poly2 = "quadratic"

fraction_of_new_samples2 = 8

# Name of the output dataset
output_filename_rans = "RANS_dataset.csv"


# --------------------------------------------------------

## RANS OPTIONS

# Aeromap name
aeromap_rans_uid = "RANS_aeromap"
aeromap_rans_name = "RANS_aeromap"

# General parameters
open_gmsh_gui = ["False"]
export_propellers = ["False"]
type_mesh = ["RANS"]  # type_mesh
symmetry = ["True"]  # symmetry

# parametri mesh RANS
farfield_factor = [3.0]
number_layer = [10]
height_first_layer = [3.0]
max_thickness_layer = [100.0]
growth_ratio = [1.2]
growth_factor = [1.4]
feature_angle = [40]
surface_mesh_size = [5.0]  # gmshOptionsmin_max_mesh_factor
surface_max_size = [0.0008]  # DA CREARE:gmshOptionsmax_mesh_factor
surface_min_size = [0.0002]  # DA CREARE: gmshOptionsmin_mesh_factor
intake_percent = [20.0]
exhaust_percent = [20.0]

# SU2
config_type = type_mesh
derivatives = ["False"]  # calculateDampingDerivatives
rotation = [1.0]  # rotationRate
control_surfaces = ["False"]  # calculateControlSurfacesDeflections
includeActuatorDisk = ["None;None"]
cpu = [9]  # nbCPU
iters = [5000]  # maxIter
cfl_adption = ["True"]  # value
cflAdFactorDown = [0.5]  # factor_down
cflAdFactorUp = [1.5]  # factor_up
cflMinValue = [0.5]  # min
cflMaxValue = [100.0]  # max
cfl_value = [1]  # cfl value
multiGrid = [3]  # multigridLevel
wettedArea = ["True"]  # updateWettedArea
extraLoads = ["False"]  # extractLoads

# Common mesh parameters
rans_common_mesh_params = {
    "open_gui": open_gmsh_gui,
    "type_mesh": type_mesh,
    "symmetry": symmetry,
    "exportPropellers": export_propellers,
}

# RANS mesh params
rans_mesh_params = {
    "farfield_factor": farfield_factor,
    "number_layer": number_layer,
    "height_first_layer": height_first_layer,
    "max_thickness_layer": max_thickness_layer,
    "growth_ratio": growth_ratio,
    "growth_factor": growth_factor,
    "feature_angle": feature_angle,
    "intake_percent": intake_percent,
    "exhaust_percent": exhaust_percent,
}

# aggiusta xk ultime due nn esistono
rans_gmsh_options = {
    "gmshOptionsmin_max_mesh_factor": surface_mesh_size,
    "gmshOptionsmin_mesh_factor": surface_min_size,
    "gmshOptionsmax_mesh_factor": surface_max_size,
}

# SU2 parameters
rans_su2_params = {
    "aeroMapUID": aeromap_rans_name,
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

# -------------------------------------------------------

print("Hai attivato ceasiompy??")
input("press ENTER to continue")

# ===== 0. CPACS =====

find_and_save_file(input_cpacs_name, cpacs_directory, directory_path)
input_cpacs_path = os.path.join(directory_path, input_cpacs_name)

print(f"CPACS name: {input_cpacs_name}")


# ===== 1. DOE =====

# Select how many fidelity level
fidelity_level, selected_paths = workflow(
    default_doe_path, default_first_kriging_dataset_path, default_second_kriging_dataset_path
)

input("Press ENTER to continue...")
if fidelity_level >= 1:

    try:
        samples, ranges, sampled_array, n_samples, full_path1 = doe_workflow(
            selected_paths, directory_path, output_filename_lhs, physical_domain_limits
        )

        print("Design of Experiment (DoE) ready:")
        print(samples)  # Mostra le prime righe del DoE generato o caricato

        if ranges is not None and sampled_array is not None and n_samples is not None:
            if len(ranges) > 0 and sampled_array.any():
                print("\nRanges:")
                print(ranges)
                print("\nProcessed Samples:")
                print(sampled_array)
                print("\nNumber of samples:")
                print(n_samples)

    except FileNotFoundError as e:
        print(f"Error: {e}")

    input("Press ENTER to continue....")

    plot_doe(
        sampled_array,
        ranges,
        n_samples,
        physical_domain_limits,
        plot_dim1="angleOfAttack",
        plot_dim2="machNumber",
    )

    # ===== 2. AVL =====

    iteration_number = 1

    selected_paths_updated = avl_workflow(
        input_cpacs_path,
        directory_path,
        selected_paths,
        default_first_kriging_dataset_path,
        full_path1,
        samples,
        aeromap_avl_uid,
        aeromap_avl_name,
        avl_parameters,
    )

    input("Press ENTER to continue....")

    # ====== 3. FIRST KRIGING MODEL =====

    (
        new_aeromap1,
        new_aeromap_array1,
        full_path2,
        which_coefficent1,
        model1,
        X1,
        y1,
        X_train1,
        y_train1,
    ) = sm_workflow(
        iteration_number,
        selected_paths_updated,
        directory_path,
        theta1,
        corr1,
        poly1,
        selected_mach_for_aoa_plot,
        altitude_for_response_surface,
        aos_for_response_surface,
        n_samples,
        fidelity_level,
        physical_domain_limits,
        fraction_of_new_samples,
        ranges,
        sampled_array,
        base_model_name=base_model_name,
        model_extension=model_extension,
        output_filename=output_filename_euler,
    )

    # print(X1)
    # print(X1.shape())

    # print(y1)
    # print(y1.shape())


if fidelity_level >= 2:

    input("Press ENTER to continue....")

    # ====== 4. PLOT NEW DOE ======

    plot_doe(
        sampled_array,
        ranges,
        n_samples,
        physical_domain_limits,
        plot_dim1="angleOfAttack",
        plot_dim2="machNumber",
        highlight_points=new_aeromap_array1,
    )

    # ====== 5. EULER ======

    iteration_number = 2

    selected_paths_updated1 = su2_workflow(
        fidelity_level,
        input_cpacs_path,
        directory_path,
        selected_paths_updated,
        default_second_kriging_dataset_path,
        full_path2,
        new_aeromap1,
        aeromap_euler_uid,
        aeromap_euler_name,
        euler_common_mesh_params,
        euler_mesh_params,
        euler_su2_params,
    )

    input("Press ENTER to continue....")

    # ====== 6. MF KRIGING ======

    (
        new_aeromap2,
        new_aeromap_array2,
        full_path3,
        which_coefficent2,
        model2,
        X2,
        y2,
        X_train2,
        y_train2,
    ) = sm_workflow(
        iteration_number,
        selected_paths_updated1,
        directory_path,
        theta2,
        corr2,
        poly2,
        selected_mach_for_aoa_plot,
        altitude_for_response_surface,
        aos_for_response_surface,
        n_samples,
        fidelity_level,
        physical_domain_limits,
        fraction_of_new_samples2,
        ranges,
        sampled_array,
        which_coefficent1,
        X_LF=X1,
        y_LF=y1,
        X_train_LF=X_train1,
        y_train_LF=y_train1,
        base_model_name=base_model_name,
        model_extension=model_extension,
        output_filename=output_filename_rans,
    )


if fidelity_level >= 3:

    input("Press ENTER to continue....")

    # ====== 7. PLOT NEW DOE ======

    plot_doe(
        sampled_array,
        ranges,
        n_samples,
        physical_domain_limits,
        plot_dim1="angleOfAttack",
        plot_dim2="machNumber",
        highlight_points=new_aeromap_array2,
    )

    # ====== 8. RANS ======

    interation_number = 3

    selected_paths_updated2 = su2_workflow(
        fidelity_level,
        input_cpacs_path,
        directory_path,
        selected_paths_updated1,
        default_third_kriging_dataset_path,
        full_path3,
        new_aeromap2,
        aeromap_rans_uid,
        aeromap_rans_name,
        rans_common_mesh_params,
        rans_mesh_params,
        rans_su2_params,
        rans_gmsh_options,
    )

    input("Press ENTER to continue....")

    # ====== 9. MF KRIGING ======

    (
        new_aeromap3,
        new_aeromap_array3,
        full_path4,
        which_coefficent3,
        model3,
        X3,
        y3,
        X_train3,
        y_train3,
    ) = sm_workflow(
        iteration_number,
        selected_paths_updated2,
        directory_path,
        output_filename_rans,
        theta3,
        corr3,
        poly3,
        selected_mach_for_aoa_plot,
        altitude_for_response_surface,
        aos_for_response_surface,
        n_samples,
        fidelity_level,
        physical_domain_limits,
        sampled_array,
        coefficent_to_predict=which_coefficent2,
        X_LF=X2,
        y_LF=y2,
        X_train_LF=X_train2,
        y_train_LF=y_train2,
        base_model_name=base_model_name,
        model_extension=model_extension,
    )
