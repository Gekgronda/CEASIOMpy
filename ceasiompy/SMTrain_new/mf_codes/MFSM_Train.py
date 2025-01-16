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
)
from CPACS_Func import (
    add_new_aeromap,
    avl_update,
    change_reference_value,
    euler_update,
    rans_update,
)

from prove import sm_workflow
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

# CPACS FILE and PATHS
input_cpacs_name = "D150_simple.xml"
cpacs_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/test_files/CPACSfiles"
directory_path = "/wrk/Gronda/validazione"
default_doe_path = "/wrk/Gronda/validazione/LHS_dataset_default.csv"
default_first_kriging_dataset_path = "/wrk/Gronda/validazione/LHS_dataset_TRAIN_default.csv"

# input_cpacs_name = "labARscaled.xml"
# directory_path = "/wrk/Gronda/labAR/prove_codice"
# cpacs_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/test_files/CPACSfiles"

# --------------------------------------------------------

# SAVING OPTIONS
base_model_name = "surrogate_model_prova1"
model_extension = ".pkl"

# --------------------------------------------------------

# AVL OPTIONS
# Aeromap name
aeromap_name = "avl_aeromap"
aeromap_uid = aeromap_name

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

# FIRST KRIGING OPTIONS

theta1 = [0.01]
corr1 = "matern32"
poly1 = "constant"
selected_mach = [0.2, 0.5, 0.7]
altitude_for_response_surface = 10000
aos_for_response_surface = 0
fraction_of_new_samples = 5


# ===== 0. CPACS =====

find_and_save_file(input_cpacs_name, cpacs_directory, directory_path)
input_cpacs_path = os.path.join(directory_path, input_cpacs_name)

print(f"CPACS name: {input_cpacs_name}")


# ===== 1. DOE =====

# Select how many fidelity level
fidelity_level, workflow = choose_fidelity_workflow()

try:
    samples, ranges, processed_samples, n_samples, full_path1 = doe_workflow(
        default_doe_path, directory_path
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
    aeromap_uid,
    aeromap_name,
    avl_parameters,
)

input("Press ENTER to continue....")


# ====== 3. FIRST KRIGING MODEL =====

new_aeromap, full_path = sm_workflow(
    first_kriging_dataset_path,
    directory_path,
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

# ====== 4. EULER ======
