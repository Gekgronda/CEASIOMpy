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
from CPACS_Func import add_new_aeromap, avl_update, change_reference_value

# CPACS FILE

input_cpacs_name = "PROVA.xml"
directory_path = "/wrk/Gronda/labAR/prove_codice"
input_cpacs_path = os.path.join(directory_path, input_cpacs_name)

# INSERT A DOMAIN OF INTEREST

n_samples, ranges = get_user_inputs()

# print(ranges.keys())

# LATIN HYPERCUBE (DOE)
print("Sampling")
samples = lh_sampling(ranges, n_samples, "altitude", "angleOfAttack")  # change arguments to plot
# samples is a dictionary with 4 key(altitude, machNumber, angleOfattack, angleOfSideslip)
print(samples)
column_name = samples.values()
print(column_name)

# output_filename = "LHS_dataset.csv"
# output_directory = "/wrk/Gronda/labAR/prove_codice"
# full_path = os.path.join(output_directory, output_filename)

# save_to_csv(samples, full_path)

# # QUI ANDREBBE PRESO IL DOE E CONFRONTATO CON IL DOMINIO FISICO, QUINDI
# # RISCALATO IL DOE

# UPDATE CPACS with FIRST AEROMAP, REFERENCE VALUE and AVL PARAMETERS
aeromap_uid = "new_aeromap"
aeromap_name = "new_aeromap"

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
# #! CAPIRE COME FARE IN MODO CHE CERCHI DA SOLO IL BASE PATH (CAMBIA IL WORKFLOW) e il CSV
# # base_path = "/wrk/Gronda/labAR/AVL/00_symmetry/takeoff500/Workflow_001/Results/PyAVL"
# # csv_filename = "/wrk/Gronda/labAR/AVL/00_symmetry/datasets/AVL_takeoff.csv"
# csv_filename = "/wrk/Gronda/labAR/prove_codice/LHS_dataset.csv"
# base_path = "/wrk/Gronda/labAR/prove_codice"
# latest_workflow_path = get_latest_workflow(base_path)

# if latest_workflow_path:
#     # Append Results and PyAVL to the latest Workflow path
#     results_path = os.path.join(latest_workflow_path, "Results", "PyAVL")
#     print("Latest Workflow:", latest_workflow_path)
#     print("Results Path:", results_path)
# else:
#     print("No Workflow found in the directory:", base_path)

# if os.path.isdir(results_path):
#     data = extract_coefficients_from_AVL(results_path)
#     print(data)
#     train_dataset_path = append_to_new_csv(data, csv_filename)
# else:
#     print(f"Errore: La directory {latest_workflow_path} non esiste.")

# # SAVE)

# # # TRAIN KRIGING
# train_dataset_path = ""
# default_train_dataset_path = "/wrk/Gronda/labAR/prove_codice/LHS_dataset_TRAIN_default.csv"
# df = load_and_split_data(train_dataset_path, default_train_dataset_path)
# print(df)

# normalized_data = normalize_data(df)

# df_norm = normalized_data["dataset"]["df"]
# X_norm = normalized_data["dataset"]["X_normalized"]
# y_norm = normalized_data["dataset"]["y_normalized"]
# cl_norm = y_norm["CL"]

# X_train, X_test, y_train, y_test = test_training_data(X_norm, cl_norm)

# theta = [0.01]
# corr = "matern32"
# poly = "constant"  # linear e quadratic  danno problemi

# model = Kriging(X_train, y_train, theta, corr, poly)
# y_pred, var = predict_model(model, X_test, y_test)

# print(f"x train: {X_train}")
# print(f"x test: {X_test}")
# print(f"y test: {y_test}")
# print(f"y pred: {y_pred}")
# print(f"variance: {var}")

# # EVALUATE HIGH VARIANCE DATA AND SAMPLING

# var_flat = var.flatten()

# sorted_indices = np.argsort(var_flat)[::-1]
# top_50_indices = sorted_indices[:50]
# top_50_X_test = X_test[top_50_indices]

# # Stampa i risultati
# print(f"Top 50 variances: {var_flat[top_50_indices]}")
# print(f"Top 50 X_test samples: {top_50_X_test}")

# # LAUNCH EULER
# tixi = Tixi3()
# tixi.open(input_cpacs_path)

# try:
#     # add the new aeroMap
#     add_new_aeromap(tixi, samples, aeromap_uid, aeromap_name)
# except Exception as e:
#     print(f"Error adding aeroMap: {e}")
# finally:
#     tixi.close()

# TRAIN MF-KRIGING

# EVALUATE HIGH VARIANCE DATA AND SAMPLING

# LAUNCH RANS

# TRAIN FINAL MF-KRIGING
