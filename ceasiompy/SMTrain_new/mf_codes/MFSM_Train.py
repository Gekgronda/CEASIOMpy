import pandas as pd
import numpy as np
import os
import subprocess
from tixi3.tixi3wrapper import Tixi3
from MFSM_Func import (
    get_user_inputs,
    load_and_split_data,
    plot_distributions,
    normalize_data,
    lh_sampling,
    save_to_csv,
    get_latest_workflow,
    extract_coefficients_from_AVL,
    append_to_new_csv,
    validate_inputs,
    test_training_data,
    MF_CoKriging,
    MF_Kriging,
    Kriging,
    predict_model,
    predict_mf_model,
    predict_mfco_model,
    plot_results,
)
from CPACS_Func import add_new_aeromap, avl_update, change_reference_value, euler_update
from sklearn.preprocessing import MinMaxScaler

# # CPACS FILE

input_cpacs_name = "PROVA.xml"
directory_path = "/wrk/Gronda/labAR/prove_codice"
input_cpacs_path = os.path.join(directory_path, input_cpacs_name)

# INSERT A DOMAIN OF INTEREST

# n_samples, ranges = get_user_inputs()

# # print(ranges.keys())

# # LATIN HYPERCUBE (DOE)
# print("Sampling")
# samples = lh_sampling(ranges, n_samples, "altitude", "angleOfAttack")  # change arguments to plot
# # samples is a dictionary with 4 key(altitude, machNumber, angleOfattack, angleOfSideslip)
# print(samples)
# column_name = samples.values()
# print(column_name)

# output_filename1 = "LHS_dataset.csv"
# output_directory21 = "/wrk/Gronda/labAR/prove_codice"
# full_path1 = os.path.join(output_directory21, output_filename1)

# save_to_csv(samples, full_path1)

# # QUI ANDREBBE PRESO IL DOE E CONFRONTATO CON IL DOMINIO FISICO, QUINDI
# # RISCALATO IL DOE

# # # UPDATE CPACS with FIRST AEROMAP, REFERENCE VALUE and AVL PARAMETERS
# aeromap_uid = "new_aeromap"
# aeromap_name = "new_aeromap"

# # Reference values
# ref_area = 0
# ref_lenght = 0.1
# x_momentum = 1

# reference_values = {
#     "area": int(ref_area),
#     "length": float(ref_lenght),
#     "point/x": int(x_momentum),
# }

# # Parameters from avl code, for the moment update from here
# integrate_fusolage = "False"
# distribution = "cosine"
# nchordwise = 20
# nspanwise = 50
# savePlots = "False"

# avl_parameters = {
#     "IntegrateFuselage": integrate_fusolage,
#     "VortexDistribution/Distribution": distribution,
#     "VortexDistribution/Nchordwise": int(nchordwise),
#     "VortexDistribution/Nspanwise": int(nspanwise),
#     "SavePlots": savePlots,
# }

# tixi = Tixi3()
# tixi.open(input_cpacs_path)

# try:
#     # add the new aeroMap
#     add_new_aeromap(tixi, samples, aeromap_uid, aeromap_name)
#     change_reference_value(tixi, reference_values)
#     avl_update(tixi, aeromap_name, avl_parameters)

#     # save the updated CPACS file
#     tixi.save(input_cpacs_path)
#     print("New aeroMap added successfully!")
# except Exception as e:
#     print(f"Error adding aeroMap: {e}")
# finally:
#     tixi.close()

# # SAVE

# # LAUNCH AVL

# print("CPACS updated, running PyAVL Module in CEASIOMpy...")
# # Combine both commands in a single shell session
# command = (
#     f"cd {os.path.abspath(directory_path)} && "
#     f"ceasiompy_run -m {os.path.abspath(input_cpacs_path)} PyAVL"
# )
# try:
#     result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
#     print("Command executed succesfully!")
#     print(result.stdout)
# except subprocess.CalledProcessError as e:
#     print("Error occurred during command execution!")
#     print(e.stderr)

# # Update dataset
# csv_filename1 = "/wrk/Gronda/labAR/prove_codice/LHS_dataset.csv"
# base_path = "/wrk/Gronda/labAR/prove_codice"
# latest_workflow_path = get_latest_workflow(base_path)

# # Find latest workflow
# if latest_workflow_path:
#     # Append Results and PyAVL to the latest Workflow path
#     results_path = os.path.join(latest_workflow_path, "Results", "PyAVL")
#     print("Latest Workflow:", latest_workflow_path)
#     print("Results Path:", results_path)
# else:
#     print("No Workflow found in the directory:", base_path)

# if os.path.isdir(results_path):
#     data1 = extract_coefficients_from_AVL(results_path)
#     print(data1)
#     train_first_kriging_dataset_path = append_to_new_csv(data1, csv_filename1)
# else:
#     print(f"Errore: La directory {latest_workflow_path} non esiste.")

# SAVE)

# # TRAIN KRIGING
train_first_kriging_dataset_path = ""  # aggiungendo questo va quello di default
train_first_kriging_path = f"{train_first_kriging_dataset_path}"
default_train_first_kriging_dataset_path = (
    "/wrk/Gronda/labAR/prove_codice/LHS_dataset_TRAIN_default.csv"
)
df1 = load_and_split_data(train_first_kriging_path, default_train_first_kriging_dataset_path)
print(df1)

# Normalize data
normalized_data1 = normalize_data(df1)

df_norm1 = normalized_data1["dataset"]["df"]
X_norm1 = normalized_data1["dataset"]["X_normalized"]
y_norm1 = normalized_data1["dataset"]["y_normalized"]
cl_norm1 = y_norm1["CL"]

# Split test and training

train_test_values1 = test_training_data(X_norm1, cl_norm1)

X_train1 = train_test_values1["X_train"]
X_test1 = train_test_values1["X_test"]
y_train1 = train_test_values1["y_train"]
y_test1 = train_test_values1["y_test"]

# Train module and predict

theta1 = [0.01]
corr1 = "matern32"
poly1 = "constant"  # linear e quadratic  danno problemi coi dati normalizzati

model1 = Kriging(X_train1, y_train1, theta1, corr1, poly1)
predictions1 = predict_model(model1, X_test1, y_test1)

y_pred1 = predictions1["y_pred"]
var1 = predictions1["variance"]

print(f"x train: {X_train1}")
print(f"x test: {X_test1}")
print(f"y test: {y_test1}")
print(f"y pred: {y_pred1}")
print(f"variance: {var1}")

# EVALUATE HIGH VARIANCE DATA AND SAMPLING

var_flat1 = var1.flatten()

sorted_indices1 = np.argsort(var_flat1)[::-1]
top_50_indices1 = sorted_indices1[:50]
top_50_X_test1 = X_test1[top_50_indices1]

# Stampa i risultati
print(f"Top 50 variances: {var_flat1[top_50_indices1]}")
print(f"Top 50 X_test samples: {top_50_X_test1}")

scaler_X1 = normalized_data1["scalers"]["scaler_X"]

X_original1 = scaler_X1.inverse_transform(top_50_X_test1)
print(f"Original top 50: {X_original1}")

altitude1 = X_original1[:, 0]
machNumber1 = X_original1[:, 1]
angleOfAttack1 = X_original1[:, 2]
angleOfSideslip1 = X_original1[:, 3]
print(altitude1)
print(machNumber1)
print(angleOfAttack1)
print(angleOfSideslip1)

# dictionary aeromap
aeromap_columns1 = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]
aeromap1 = {name: X_original1[:, i] for i, name in enumerate(aeromap_columns1)}

for key, value in aeromap1.items():
    print(f"{key}: {value}")

print(aeromap1)
print(type(aeromap1))
# miglioramento: aggiungere una funzione di densita per evitare che i putni siano troppo vicini
# miglioramento: confrontare ad ogni step i modelli kriging x valutare un effettivo miglioramento

# output_filename2 = "EULER_dataset.csv"
# output_directory2 = "/wrk/Gronda/labAR/prove_codice"
# full_path2 = os.path.join(output_directory2, output_filename2)

# save_to_csv(aeromap1, full_path2)

# LAUNCH EULER

# print(samples)
# print(aeromap)

aeromap_uid = "new_aeromap"
aeromap_name = "new_aeromap"

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

# SU2
config_type = type_mesh
derivatives = ["False"]  # calculateDampingDerivatives
rotation = [1.0]  # rotationRate
control_surfaces = ["False"]  # calculateControlSurfacesDeflections
includeActuatorDisk = ["None;None"]
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


# Conversion of lists of dictionaries to simple dictionaries

# Common mesh parameters
common_mesh_params = {
    "type_mesh": type_mesh,
    "symmetry": symmetry,
}

# Euler mesh parameters
euler_mesh_params = {
    "farfield_factor": farfield_factors,
    "mesh_size/farfield": mesh_farfields,
    "mesh_size/fuselage/factor": fuselage_factors,
    "mesh_size/wings/factor": wing_factors,
    "n_power_factor": n_power_factors,
    "n_power_field": n_power_fields,
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

# tixi = Tixi3()
# tixi.open(input_cpacs_path)

# try:
#     # add the new aeroMa
#     add_new_aeromap(tixi, aeromap1, aeromap_uid, aeromap_name)
#     euler_update(tixi, aeromap_name, common_mesh_params, euler_mesh_params, su2_params)
#     # save the updated CPACS file
#     tixi.save(input_cpacs_path)
#     print("New aeroMap added successfully!")
# except Exception as e:
#     print(f"Error adding aeroMap: {e}")
# finally:
#     tixi.close()

# print("CPACS updated, running GMSH and SU2 Module in CEASIOMpy...")
# # Combine both commands in a single shell session
# command = (
#     f"cd {os.path.abspath(directory_path)} && "
#     f"ceasiompy_run -m {os.path.abspath(input_cpacs_path)} CPACS2GMSH SU2Run"
# )
# try:
#     result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
#     print("Command executed succesfully!")
#     print(result.stdout)
# except subprocess.CalledProcessError as e:
#     print("Error occurred during command execution!")
#     print(e.stderr)


# TRAIN MF-KRIGING

# Find latest workflow

# csv_filename2 = "/wrk/Gronda/labAR/prove_codice/EULER_dataset.csv"
# latest_workflow_path = get_latest_workflow(base_path)

# if latest_workflow_path:
#     # Append Results and SU2 to the latest Workflow path
#     results_path = os.path.join(latest_workflow_path, "Results", "SU2")
#     print("Latest Workflow:", latest_workflow_path)
#     print("Results Path:", results_path)
# else:
#     print("No Workflow found in the directory:", base_path)

# if os.path.isdir(results_path):
#     data2 = extract_coefficients_from_SU2(results_path)
#     print(data2)
#     train_second_kriging_dataset_path = append_to_new_csv(data2, csv_filename2)
# else:
#     print(f"Errore: La directory {latest_workflow_path} non esiste.")

# print(train_second_kriging_dataset_path)


train_second_kriging_dataset_path = ""  # aggiungendo questo va quello di default
train_second_krigin_path = f"{train_second_kriging_dataset_path}"

default_train_second_kriging_dataset_path = (
    "/wrk/Gronda/labAR/prove_codice/EULER_dataset_TRAIN_default.csv"
)
df2 = load_and_split_data(train_second_krigin_path, default_train_second_kriging_dataset_path)
print(df2)

# Normalize data
normalized_data2 = normalize_data(df2)

df_norm2 = normalized_data2["dataset"]["df"]
X_norm2 = normalized_data2["dataset"]["X_normalized"]
y_norm2 = normalized_data2["dataset"]["y_normalized"]
cl_norm2 = y_norm2["CL"]

# Split test and training

train_test_values2 = test_training_data(X_norm2, cl_norm2)

X_train2 = train_test_values2["X_train"]
X_test2 = train_test_values2["X_test"]
y_train2 = train_test_values2["y_train"]
y_test2 = train_test_values2["y_test"]

# Train module and predict

theta2 = [0.01]
corr2 = "matern32"
poly2 = "constant"  # linear e quadratic  danno problemi coi dati normalizzati

model2 = MF_Kriging(X_train1, y_train1, X_train2, y_train2, theta2, corr2, poly2)
predictions2 = predict_mf_model(model2, X_test2, y_test2)

y_pred2 = predictions2["y_pred"]
var2 = predictions2["variance"]

print(f"x train: {X_train2}")
print(f"x test: {X_test2}")
print(f"y test: {y_test2}")
print(f"y pred: {y_pred2}")
print(f"variance: {var2}")

# EVALUATE HIGH VARIANCE DATA AND SAMPLING

var_flat2 = var2.flatten()

sorted_indices2 = np.argsort(var_flat2)[::-1]
top_10_indices1 = sorted_indices2[:10]
top_10_X_test1 = X_test2[top_10_indices1]

# Stampa i risultati
print(f"Top 50 variances: {var_flat2[top_10_indices1]}")
print(f"Top 50 X_test samples: {top_10_X_test1}")

scaler_X2 = normalized_data2["scalers"]["scaler_X"]

X_original2 = scaler_X2.inverse_transform(top_10_X_test1)
print(f"Original top 50: {X_original2}")

altitude2 = X_original2[:, 0]
machNumber2 = X_original2[:, 1]
angleOfAttack2 = X_original2[:, 2]
angleOfSideslip2 = X_original2[:, 3]
print(altitude2)
print(machNumber2)
print(angleOfAttack2)
print(angleOfSideslip2)

# dictionary aeromap
aeromap_columns2 = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]
aeromap2 = {name: X_original2[:, i] for i, name in enumerate(aeromap_columns2)}

for key, value in aeromap2.items():
    print(f"{key}: {value}")

print(aeromap2)
print(type(aeromap2))
# miglioramento: aggiungere una funzione di densita per evitare che i putni siano troppo vicini
# miglioramento: confrontare ad ogni step i modelli kriging x valutare un effettivo miglioramento

output_filename3 = "RANS2_dataset.csv"
output_directory3 = "/wrk/Gronda/labAR/prove_codice"
full_path3 = os.path.join(output_directory3, output_filename3)

save_to_csv(aeromap2, full_path3)


# EVALUATE HIGH VARIANCE DATA AND SAMPLING

# LAUNCH RANS

# TRAIN FINAL MF-KRIGING
