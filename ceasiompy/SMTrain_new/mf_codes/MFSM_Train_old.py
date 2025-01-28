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

# CPACS FILE
input_cpacs_name = "D150_simple.xml"
directory_path = "/wrk/Gronda/validazione"
cpacs_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/test_files/CPACSfiles"
# input_cpacs_name = "labARscaled.xml"
# directory_path = "/wrk/Gronda/labAR/prove_codice"
# cpacs_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/test_files/CPACSfiles"

# Important for saving
base_model_name = "surrogate_model_prova1"
model_extension = ".pkl"

# ===== 0. CPACS =====

find_and_save_file(input_cpacs_name, cpacs_directory, directory_path)
input_cpacs_path = os.path.join(directory_path, input_cpacs_name)

print(f"CPACS name: {input_cpacs_name}")

# TO DO
# AGGIUNGERE IL FATTO CHE LHS NON FACCIA DOPPIONI
# SE METTI UN SOLO VALORE PER LHS NON FUNZIONA

# ===== 1. DOE =====

# Select how many fidelity level
fidelity_level, workflow = choose_fidelity_workflow()

# CHECK DEFAULT DATASET!!
default_doe_path = "/wrk/Gronda/validazione/LHS_dataset_default.csv"
# default_doe_path = "/wrk/Gronda/labAR/prove_codice/LHS_dataset_default.csv"

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

# QUI ANDREBBE PRESO IL DOE E CONFRONTATO CON IL DOMINIO FISICO, QUINDI
# RISCALATO IL DOE

# # Reference values == PER IL MOMENTO DISATTIVATO, SE ATTIVI RICORDA DI CAMBIARE ANCHE SOTTO
# # ref_area = 1
# # ref_lenght = 0.1
# # x_momentum = 1

# # reference_values = {
# #     "area": int(ref_area),
# #     "length": float(ref_lenght),
# #     "point/x": int(x_momentum),
# # }

# ==== 2. AVL ====

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


# ====== CHECK DEFAULT DATASET
default_first_kriging_dataset_path = "/wrk/Gronda/validazione/LHS_dataset_TRAIN_default.csv"
# # default_first_kriging_dataset_path = "/wrk/Gronda/labAR/prove_codice/LHS_dataset_TRAIN_default.csv"

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

# ============= 3. FIRST SURROGATE MODEL =============
# print("Training first surrogate model")

# #load and split = select coeff
# df1 = load_and_split_data(first_kriging_dataset_path)
# X1 = df1["dataset"]["X"]
# y1 = df1["dataset"]["y"]
# which_coefficent = input("Insert which coefficent to predict (CL, CD, CM) [default: CL]: ") or "CL"
# coefficent1 = y1[f"{which_coefficent}"]

# # divide train and test
# train_test_values1 = test_training_data(X1, coefficent1)
# X_train1 = train_test_values1["X_train"]
# X_test1 = train_test_values1["X_test"]
# y_train1 = train_test_values1["y_train"]
# y_test1 = train_test_values1["y_test"]

# print("y_test:")
# print(y_test1)
# print("Data splitted")
# input("Press ENTER to continue....")
# print("Training first surrogate model")

# Train module and predict
theta1 = [0.01]
corr1 = "matern32"
poly1 = "constant"  # normalizzazione tolta x kriging normalizza gia

# model1 = Kriging(X_train1, y_train1, theta1, corr1, poly1)

# print("Surrogate model trained, now making predictions and evaulating variance")
# RMSE X CONFRONTO
# rms1 = compute_rms_error(model1, X_test1, y_test1)
# predictions1 = predict_model(model1, X_test1, y_test1)
# y_pred1 = predictions1["y_pred"]
# print("y_pred:")
# print(y_pred1)
# plot_validation(y_test1, y_pred1, which_coefficent)
# var1 = predictions1["variance"]

# PLOT SUP DI RISPOSTA E CL-ALPHA

# mach_range = ranges["machNumber"]
# aoa_range = ranges["angleOfAttack"]
# print("mach range:")
# print(mach_range)

selected_mach = [0.2, 0.5, 0.7]
# print("X_test:")
# print(X_test1)
altitude_for_response_surface = 10000
aos_for_response_surface = 0
# plot_response_surface(
#     altitude_for_response_surface, aos_for_response_surface, X_train1, y_train1, model1, which_coefficent, mach_range, aoa_range
# )
# plot_cl_alpha_for_mach(X_train1, y_train1, model1, selected_mach)


# print(f"x train: {X_train1}")
# print(f"x test: {X_test1}")
# print(f"y test: {y_test1}")
# print(f"y pred: {y_pred1}")
# print(f"variance: {var1}")

# EVALUATE HIGH VARIANCE DATA AND SAMPLING
# var_flat1 = var1.flatten()

# print("Model evaluated, now selecting doe points with highest variance")
# input("Press ENTER to continue....")

fraction_of_new_samples = 5
# sorted_indices1 = np.argsort(var_flat1)[::-1]
# n_samples1 = n_samples // fraction_of_new_samples
# top_n_indices1 = sorted_indices1[:n_samples1]
# top_n_X_test1 = X_test1[top_n_indices1]

# # Stampa i risultati
# print(f"Top {n_samples1} variances: {var_flat1[top_n_indices1]}")
# print(f"Top {n_samples1} X_test samples: {top_n_X_test1}")

# plot_doe(
#     processed_samples,
#     ranges,
#     n_samples,
#     plot_dim1="angleOfAttack",
#     plot_dim2="machNumber",
#     highlight_points=top_n_X_test1,
# )

coefficent_to_predict, top_n_X_test1, model1, rms1 = sm_workflow(
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

altitude1 = top_n_X_test1[:, 0]
machNumber1 = top_n_X_test1[:, 1]
angleOfAttack1 = top_n_X_test1[:, 2]
angleOfSideslip1 = top_n_X_test1[:, 3]

input("Press ENTER to continue: ")
# print(altitude1)
# print(machNumber1)
# print(angleOfAttack1)
# print(angleOfSideslip1)

## inserisci doe con nuovi punti

# dictionary aeromap
aeromap_columns1 = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]
aeromap1 = {name: top_n_X_test1[:, i] for i, name in enumerate(aeromap_columns1)}

for key, value in aeromap1.items():
    print(f"{key}: {value}")

# print(aeromap1)
# print(type(aeromap1))


output_filename2 = "EULER_dataset.csv"
full_path2 = os.path.join(directory_path, output_filename2)

save_to_csv(aeromap1, full_path2)

input("Press ENTER to continue....")

# print("New aeromap saved")
# input("Press ENTER to continue....")

# LAUNCH EULER

# Update aeromap for euler
print("Updating aeromap for Euler simulations")
input("Press ENTER to continue....")

aeromap_uid = "euler_aeromap"
aeromap_name = "euler_aeromap"

tixi = Tixi3()
tixi.open(input_cpacs_path)

try:
    # add the new aeroMa
    add_new_aeromap(tixi, aeromap1, aeromap_uid, aeromap_name)
    # save the updated CPACS file
    tixi.save(input_cpacs_path)
    print("New aeroMap added successfully!")
except Exception as e:
    print(f"Error adding aeroMap: {e}")
finally:
    tixi.close()


print("Updating parameters for Euler simulations")
input("Press ENTER to continue....")

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
    "aeroMapUID": aeromap_name,
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

tixi = Tixi3()
tixi.open(input_cpacs_path)

try:
    euler_update(tixi, aeromap_name, common_mesh_params, euler_mesh_params, su2_params)
    # save the updated CPACS file
    tixi.save(input_cpacs_path)
    print("Euleran parameters updated successfully!")
except Exception as e:
    print(f"Error updating parameters: {e}")
finally:
    tixi.close()

input("Press ENTER to continue....")

# LAUNCH EULER COMMAND
print("CPACS updated, running GMSH and SU2 Module in CEASIOMpy...")
command = (
    f"cd {os.path.abspath(directory_path)} && "
    f"ceasiompy_run -m {os.path.abspath(input_cpacs_path)} CPACS2GMSH SU2Run"
)

default_second_kriging_dataset_path = (
    "/wrk/Gronda/labAR/prove_codice/EULER_dataset_TRAIN_default.csv"
)
train_second_kriging_dataset_path = None  # Initialize with None

try:
    print("Euler simulation started. Press Ctrl+C to interrupt manually.")
    # Run the command with subprocess.run()
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

    # Check if the process completed successfully
    if result.returncode == 0:
        print("Simulations completed successfully!")
    else:
        print("An error occurred during the simulations!")
except subprocess.CalledProcessError as e:
    print(f"Error occurred during the simulation: {e.stderr}")
    # Use the default dataset if there was an error
    train_second_kriging_dataset_path = default_second_kriging_dataset_path
    result = None  # Ensure result is defined
except KeyboardInterrupt:
    print("\nSimulation manually interrupted.")
    # Manually set to default dataset if interrupted
    train_second_kriging_dataset_path = default_second_kriging_dataset_path
    result = None  # Ensure result is defined

# DEFINE the CSV filename for the EULER dataset
csv_filename2 = "/wrk/Gronda/labAR/prove_codice/EULER_dataset.csv"

# PROCESS SIMULATION RESULTS OR USE DEFAULT DATASET
if result and result.returncode == 0:
    # If the process completed, analyze the results
    latest_workflow_path = get_latest_workflow(directory_path)
    if latest_workflow_path:
        results_path = os.path.join(latest_workflow_path, "Results", "SU2")
        print("Latest Workflow:", latest_workflow_path)
        print("Results Path:", results_path)

        if os.path.isdir(results_path):
            data2 = extract_coefficients_from_SU2(results_path)
            print(data2)
            train_second_kriging_dataset_path = append_to_new_csv(data2, csv_filename2)
        else:
            print(f"Error: The directory {results_path} does not exist.")
    else:
        print("No workflow found.")
else:
    # Use the default dataset if the process was interrupted
    print(f"Using the default Euler dataset: {default_second_kriging_dataset_path}")
    train_second_kriging_dataset_path = default_second_kriging_dataset_path

input("Press ENTER to continue....")

# TRAINING MULTI-FIDELITY KRIGING MODEL
print("Training new multi-fidelity surrogate model")

df2 = load_and_split_data(
    train_second_kriging_dataset_path,
    default_second_kriging_dataset_path,
)
print(df2)
X2 = df2["dataset"]["X"]
y2 = df2["dataset"]["y"]
coefficent2 = y2["CL"]

# Normalize data
# normalized_data2 = normalize_data(df2)

# Split test and training

train_test_values2 = test_training_data(X2, coefficent2)

X_train2 = train_test_values2["X_train"]
X_test2 = train_test_values2["X_test"]
y_train2 = train_test_values2["y_train"]
y_test2 = train_test_values2["y_test"]

print("Data splitted, now training and evaluating surrogate model")
input("Press ENTER to continue....")

# Train module and predict

theta2 = [0.01]
corr2 = "matern32"
poly2 = "constant"  #

model2 = MF_Kriging(X_train1, y_train1, X_train2, y_train2, theta2, corr2, poly2)
predictions2 = predict_mf_model(model2, X_test2, y_test2)

y_pred2 = predictions2["y_pred"]
var2 = predictions2["variance"]

# RMSE X CONFRONTO
rms2 = compute_rms_error(model2, X_test2, y_test2)
# aggiungi grafico confronto y e ypred

print(f"x train: {X_train2}")
print(f"x test: {X_test2}")
print(f"y test: {y_test2}")
print(f"y pred: {y_pred2}")
print(f"variance: {var2}")

# EVALUATE HIGH VARIANCE DATA AND SAMPLING

var_flat2 = var2.flatten()

sorted_indices2 = np.argsort(var_flat2)[::-1]
n_samples2 = n_samples1 // 5
top_n_indices2 = sorted_indices2[:n_samples2]
top_n_X_test2 = X_test2[top_n_indices2]

# Stampa i risultati
print(f"Top {n_samples2} variances: {var_flat2[top_n_indices2]}")
print(f"Top {n_samples2} X_test samples: {top_n_X_test2}")

altitude2 = top_n_X_test2[:, 0]
machNumber2 = top_n_X_test2[:, 1]
angleOfAttack2 = top_n_X_test2[:, 2]
angleOfSideslip2 = top_n_X_test2[:, 3]
print(altitude2)
print(machNumber2)
print(angleOfAttack2)
print(angleOfSideslip2)

# dictionary aeromap
aeromap_columns2 = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]
aeromap2 = {name: top_n_X_test2[:, i] for i, name in enumerate(aeromap_columns2)}

for key, value in aeromap2.items():
    print(f"{key}: {value}")

print(aeromap2)
print(type(aeromap2))
# miglioramento: aggiungere una funzione di densita per evitare che i putni siano troppo vicini
# miglioramento: confrontare ad ogni step i modelli kriging x valutare un effettivo miglioramento

output_filename3 = "RANS_dataset.csv"
full_path3 = os.path.join(directory_path, output_filename3)

save_to_csv(aeromap2, full_path3)

print("New aeromap saved")
input("Press ENTER to continue....")

# LAUNCH RANS

print("Updating aeromap for RANS simulations")
input("Press ENTER to continue....")


aeromap_uid = "RANS_aeromap"
aeromap_name = "RANS_aeromap"

tixi = Tixi3()
tixi.open(input_cpacs_path)

try:
    # add the new aeroMap
    add_new_aeromap(tixi, aeromap2, aeromap_uid, aeromap_name)
    # save the updated CPACS file
    tixi.save(input_cpacs_path)
    print("New aeroMap added successfully!")
except Exception as e:
    print(f"Error adding aeroMap: {e}")
finally:
    tixi.close()


print("Updating parameters for RANS simulations")
input("Press ENTER to continue....")

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
common_mesh_params = {
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
gmsh_options = {
    "gmshOptionsmin_max_mesh_factor": surface_mesh_size,
    "gmshOptionsmin_mesh_factor": surface_min_size,
    "gmshOptionsmax_mesh_factor": surface_max_size,
}

# SU2 parameters
su2_params = {
    "aeroMapUID": aeromap_name,
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


tixi = Tixi3()
tixi.open(input_cpacs_path)

try:
    # add the new aeroMa
    add_new_aeromap(tixi, aeromap2, aeromap_uid, aeromap_name)
    rans_update(tixi, aeromap_name, common_mesh_params, rans_mesh_params, gmsh_options, su2_params)
    # save the updated CPACS file
    tixi.save(input_cpacs_path)
    print("RANS parameters updated successfully!")
except Exception as e:
    print(f"Error updating parameters: {e}")
finally:
    tixi.close()


input("Press ENTER to continue....")


# LAUNCH RANS COMMAND
print("CPACS updated, running GMSH and SU2 Module in CEASIOMpy...")
command = (
    f"cd {os.path.abspath(directory_path)} && "
    f"ceasiompy_run -m {os.path.abspath(input_cpacs_path)} CPACS2GMSH SU2Run"
)

default_third_kriging_dataset_path = (
    "/wrk/Gronda/labAR/prove_codice/RANS_dataset_TRAIN_default.csv"
)
train_third_kriging_dataset_path = None  # Initialize with None

try:
    print("RANS simulation started. Press Ctrl+C to interrupt manually.")
    # Run the command with subprocess.run()
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

    # Check if the process completed successfully
    if result.returncode == 0:
        print("Simulations completed successfully!")
    else:
        print("An error occurred during the simulations!")
except subprocess.CalledProcessError as e:
    print(f"Error occurred during the simulation: {e.stderr}")
    # Use the default dataset if there was an error
    train_third_kriging_dataset_path = default_third_kriging_dataset_path
    result = None  # Ensure result is defined
except KeyboardInterrupt:
    print("\nSimulation manually interrupted.")
    # Manually set to default dataset if interrupted
    train_third_kriging_dataset_path = default_third_kriging_dataset_path
    result = None  # Ensure result is defined

# DEFINE the CSV filename for the EULER dataset
csv_filename3 = "/wrk/Gronda/labAR/prove_codice/RANS_dataset.csv"

# PROCESS SIMULATION RESULTS OR USE DEFAULT DATASET
if result and result.returncode == 0:
    # If the process completed, analyze the results
    latest_workflow_path = get_latest_workflow(directory_path)
    if latest_workflow_path:
        results_path = os.path.join(latest_workflow_path, "Results", "SU2")
        print("Latest Workflow:", latest_workflow_path)
        print("Results Path:", results_path)

        if os.path.isdir(results_path):
            data3 = extract_coefficients_from_SU2(results_path)
            print(data2)
            train_third_kriging_dataset_path = append_to_new_csv(data3, csv_filename3)
        else:
            print(f"Error: The directory {results_path} does not exist.")
    else:
        print("No workflow found.")
else:
    # Use the default dataset if the process was interrupted
    print(f"Using the default RANS dataset: {default_third_kriging_dataset_path}")
    train_third_kriging_dataset_path = default_third_kriging_dataset_path

input("Press ENTER to continue....")

# TRAINING FINAL MULTI-FIDELITY KRIGING MODEL
print("Training new multi-fidelity surrogate model")

df3 = load_and_split_data(
    train_third_kriging_dataset_path,
    default_third_kriging_dataset_path,
)
print(df3)
X3 = df3["dataset"]["X"]
y3 = df3["dataset"]["y"]
coefficent3 = y3["CL"]

# # Normalize data
# normalized_data3 = normalize_data(df3)

# Split test and training

train_test_values3 = test_training_data(X3, coefficent3)

X_train3 = train_test_values3["X_train"]
X_test3 = train_test_values3["X_test"]
y_train3 = train_test_values3["y_train"]
y_test3 = train_test_values3["y_test"]

print("Data splitted, now training and evaluating surrogate model")
input("Press ENTER to continue....")

# Train module and predict

theta3 = [0.01]
corr3 = "matern32"
poly3 = "constant"

model3 = MF_Kriging(
    X_train1, y_train1, X_train2, y_train2, theta3, corr3, poly3, X_train3, y_train3
)
predictions3 = predict_mf_model(model3, X_test3, y_test3)

y_pred3 = predictions3["y_pred"]
var3 = predictions3["variance"]

# RMSE X CONFRONTO
rms3 = compute_rms_error(model3, X_test3, y_test3)
# aggiungi grafico confronto y e ypred

print(f"x train: {X_train3}")
print(f"x test: {X_test3}")
print(f"y test: {y_test3}")
print(f"y pred: {y_pred3}")
print(f"variance: {var3}")

print("rmse comparison")
print(f"rms first kriging: {rms1}")
print(f"rms second kriging: {rms2}")
print(f"rms third kriging: {rms3}")

base_model_name = "surrogate_model"
model_extension = ".pkl"

save_model(model3, directory_path, base_model_name, model_extension)
