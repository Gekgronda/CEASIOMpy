import pandas as pd
import numpy as np
import os
import sys
import subprocess
from tixi3.tixi3wrapper import Tixi3
from MFSM_Func import (
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
from CPACS_Func import add_new_aeromap, avl_update, change_reference_value, SU2_update
from sklearn.preprocessing import MinMaxScaler
from smt.utils.misc import compute_rms_error
import pandas as pd
import numpy as np
import os
import shutil
import csv
import re
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler
from smt.surrogate_models import KRG
from smt.applications import EGO, MFK
from smt.sampling_methods import LHS
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from smt.utils.misc import compute_rms_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import griddata
import pickle
from tixi3.tixi3wrapper import Tixi3
import os
import numpy as np
import pandas as pd
from tixi3.tixi3wrapper import Tixi3
from ceasiompy.utils.moduleinterfaces import CPACSInOut
from ceasiompy.utils.commonxpath import (
    AEROPERFORMANCE_XPATH,
    AVL_XPATH,
    AVL_AEROMAP_UID_XPATH,
    CEASIOMPY_XPATH,
    REF_XPATH,
    GMSH_XPATH,
    GMSH_SYMMETRY_XPATH,
    GMSH_FARFIELD_FACTOR_XPATH,
    GMSH_MESH_SIZE_ENGINES_XPATH,
    GMSH_MESH_SIZE_FACTOR_FUSELAGE_XPATH,
    GMSH_MESH_SIZE_FACTOR_WINGS_XPATH,
    GMSH_MESH_SIZE_FARFIELD_XPATH,
    MESH_XPATH,
    SU2_XPATH,
    SU2_AEROMAP_UID_XPATH,
)


# X_test = np.array(
#     [
#         [0.55368353, 0.025, 0.66666667, 0.0],
#         [0.80962553, 0.35, 0.2, 0.0],
#         [0.66582426, 0.425, 0.66666667, 0.0],
#         [0.26905951, 0.85, 0.66666667, 0.0],
#         [0.90474758, 0.6, 0.73333333, 0.0],
#         [0.0986722, 0.9, 0.66666667, 0.0],
#         [0.52277514, 0.25, 0.53333333, 0.0],
#         [0.63716228, 0.675, 0.0, 0.0],
#         [0.89943238, 0.925, 0.4, 0.0],
#         [0.30634615, 0.45, 0.6, 0.0],
#         [0.40970175, 0.125, 0.2, 0.0],
#         [0.68675412, 0.4, 0.6, 0.0],
#         [0.58268648, 0.95, 0.13333333, 0.0],
#         [0.01420061, 0.975, 0.46666667, 0.0],
#         [0.99021201, 0.975, 0.13333333, 0.0],
#         [0.33105682, 0.8, 0.8, 0.0],
#         [0.80615561, 0.275, 0.8, 0.0],
#         [0.04215055, 0.15, 0.2, 0.0],
#         [0.38230339, 0.95, 0.26666667, 0.0],
#         [0.64119381, 1.0, 1.0, 0.0],
#         [0.69296187, 0.5, 0.93333333, 0.0],
#         [0.98207875, 0.4, 0.13333333, 0.0],
#         [0.37434061, 0.475, 0.86666667, 0.0],
#         [0.27857673, 0.925, 0.86666667, 0.0],
#         [0.05083539, 0.325, 0.46666667, 0.0],
#         [0.4348938, 0.5, 0.33333333, 0.0],
#         [0.27449506, 0.55, 0.26666667, 0.0],
#         [0.54145857, 0.175, 0.93333333, 0.0],
#         [0.89517019, 0.875, 0.26666667, 0.0],
#         [0.51682813, 0.9, 0.86666667, 0.0],
#         [0.0449285, 0.975, 0.0, 0.0],
#         [0.40235072, 0.65, 0.0, 0.0],
#         [0.25271276, 0.675, 0.13333333, 0.0],
#         [0.13316084, 1.0, 0.53333333, 0.0],
#         [0.53646429, 0.025, 0.06666667, 0.0],
#         [0.6397196, 0.975, 0.53333333, 0.0],
#         [0.42949837, 0.15, 0.06666667, 0.0],
#         [0.04782678, 0.625, 0.4, 0.0],
#         [0.92185651, 0.8, 0.66666667, 0.0],
#         [0.49658022, 0.05, 0.93333333, 0.0],
#         [0.00323926, 0.575, 0.53333333, 0.0],
#         [0.65516377, 0.85, 0.13333333, 0.0],
#         [0.1913974, 0.225, 0.6, 0.0],
#         [0.67039734, 0.3, 0.66666667, 0.0],
#         [0.732866, 0.125, 0.73333333, 0.0],
#         [0.93808292, 0.05, 0.93333333, 0.0],
#         [0.82841928, 0.075, 0.66666667, 0.0],
#         [0.85235774, 0.675, 0.2, 0.0],
#         [0.83300239, 0.35, 0.4, 0.0],
#         [0.69673266, 0.275, 0.8, 0.0],
#         [0.95022765, 0.2, 0.26666667, 0.0],
#         [0.53239264, 0.375, 0.8, 0.0],
#         [0.36960708, 0.475, 0.0, 0.0],
#         [0.93193534, 0.275, 0.46666667, 0.0],
#         [0.87145235, 0.575, 0.26666667, 0.0],
#         [0.13681128, 0.475, 0.2, 0.0],
#         [0.63225826, 0.4, 0.93333333, 0.0],
#         [0.86839361, 0.375, 0.86666667, 0.0],
#         [0.96037668, 0.575, 0.73333333, 0.0],
#         [0.61924103, 0.85, 0.46666667, 0.0],
#         [0.16891309, 0.175, 0.4, 0.0],
#         [0.38205267, 0.5, 0.2, 0.0],
#         [0.111539, 0.0, 0.8, 0.0],
#         [0.767134, 0.375, 0.8, 0.0],
#         [0.11950178, 0.15, 0.73333333, 0.0],
#         [0.4625529, 0.8, 0.0, 0.0],
#         [0.59229396, 0.1, 0.33333333, 0.0],
#         [0.90832782, 0.425, 0.06666667, 0.0],
#         [0.79470285, 0.225, 0.73333333, 0.0],
#         [0.85354113, 0.75, 0.4, 0.0],
#         [0.73972562, 0.625, 0.86666667, 0.0],
#         [0.93571615, 0.95, 0.86666667, 0.0],
#         [0.93117316, 0.8, 0.06666667, 0.0],
#         [0.76954089, 0.175, 0.93333333, 0.0],
#         [0.31156106, 0.3, 0.6, 0.0],
#         [0.41387368, 0.55, 0.13333333, 0.0],
#         [0.22274706, 0.7, 1.0, 0.0],
#         [0.71012095, 0.4, 0.86666667, 0.0],
#         [0.82131897, 0.9, 0.33333333, 0.0],
#         [0.41505706, 0.775, 0.86666667, 0.0],
#         [0.17294462, 0.725, 0.4, 0.0],
#         [0.58550454, 0.375, 0.26666667, 0.0],
#         [0.09459053, 0.675, 0.0, 0.0],
#         [0.93649839, 0.3, 0.6, 0.0],
#         [0.84465572, 0.25, 0.66666667, 0.0],
#         [0.32510981, 0.425, 0.93333333, 0.0],
#         [0.9483924, 0.05, 0.33333333, 0.0],
#         [0.72560523, 0.475, 0.0, 0.0],
#         [0.75289327, 0.325, 0.6, 0.0],
#         [0.97727501, 0.7, 0.06666667, 0.0],
#         [0.40722466, 0.775, 0.93333333, 0.0],
#         [0.12832702, 0.225, 0.86666667, 0.0],
#         [0.24796919, 0.55, 0.86666667, 0.0],
#         [0.52619492, 0.575, 0.86666667, 0.0],
#         [0.53842991, 0.45, 0.26666667, 0.0],
#         [0.56630965, 0.525, 0.33333333, 0.0],
#         [0.35312995, 0.55, 0.46666667, 0.0],
#         [0.44701847, 0.9, 0.33333333, 0.0],
#         [0.76310247, 0.925, 0.66666667, 0.0],
#         [0.95576348, 0.1, 0.66666667, 0.0],
#         [0.68920112, 0.275, 0.4, 0.0],
#         [0.22893475, 0.375, 0.06666667, 0.0],
#         [0.61498887, 0.6, 0.6, 0.0],
#         [0.50366047, 0.15, 0.66666667, 0.0],
#         [0.19565959, 0.475, 0.6, 0.0],
#         [0.99592836, 0.85, 0.33333333, 0.0],
#         [0.34683194, 0.775, 0.13333333, 0.0],
#         [0.41697254, 0.05, 0.66666667, 0.0],
#         [0.27672142, 0.1, 0.06666667, 0.0],
#         [0.58970656, 0.425, 0.8, 0.0],
#         [0.54651303, 0.975, 0.4, 0.0],
#         [0.38657561, 0.775, 0.46666667, 0.0],
#         [0.00281806, 0.25, 0.8, 0.0],
#         [0.18854925, 0.725, 0.73333333, 0.0],
#         [0.12551898, 0.6, 0.06666667, 0.0],
#         [0.74828008, 0.75, 0.93333333, 0.0],
#         [0.01516337, 0.75, 0.06666667, 0.0],
#         [0.39388652, 0.275, 0.06666667, 0.0],
#         [0.69765529, 0.175, 0.2, 0.0],
#         [0.75947209, 0.825, 0.46666667, 0.0],
#         [0.92966885, 0.925, 0.6, 0.0],
#         [0.15870389, 0.825, 0.53333333, 0.0],
#         [0.40008424, 0.1, 0.66666667, 0.0],
#         [0.81549231, 0.325, 0.46666667, 0.0],
#         [0.21025132, 0.725, 0.2, 0.0],
#         [0.35455402, 0.5, 0.93333333, 0.0],
#         [0.23582446, 0.975, 0.13333333, 0.0],
#         [0.29785186, 0.025, 0.06666667, 0.0],
#         [0.33677317, 0.275, 0.66666667, 0.0],
#         [0.03367631, 0.675, 0.33333333, 0.0],
#         [0.52966484, 0.2, 0.13333333, 0.0],
#         [0.39440801, 0.65, 0.26666667, 0.0],
#         [0.05518784, 0.2, 0.6, 0.0],
#         [0.06967928, 0.125, 0.66666667, 0.0],
#         [0.24656518, 0.7, 0.73333333, 0.0],
#         [0.97001424, 0.35, 0.33333333, 0.0],
#         [0.88184207, 0.05, 0.6, 0.0],
#         [0.57373087, 0.5, 0.4, 0.0],
#         [0.98667188, 0.475, 0.46666667, 0.0],
#         [0.06767355, 0.65, 0.53333333, 0.0],
#         [0.59944441, 0.1, 0.73333333, 0.0],
#         [0.11651323, 0.6, 0.2, 0.0],
#         [0.86203542, 0.975, 0.53333333, 0.0],
#         [0.4803839, 0.725, 0.13333333, 0.0],
#         [0.01842269, 0.475, 0.86666667, 0.0],
#         [0.17851054, 0.925, 0.53333333, 0.0],
#         [0.35756263, 0.625, 0.53333333, 0.0],
#         [0.56864633, 0.925, 0.86666667, 0.0],
#         [0.06043284, 0.15, 0.4, 0.0],
#         [0.46638386, 0.75, 0.53333333, 0.0],
#     ]
# )


# var = np.array(
#     [
#         [2.17796081e-07],
#         [0.00000000e00],
#         [1.35734384e-08],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [1.10346382e-07],
#         [6.55168171e-09],
#         [0.00000000e00],
#         [0.00000000e00],
#         [9.26294909e-09],
#         [8.97057510e-09],
#         [0.00000000e00],
#         [0.00000000e00],
#         [1.93657698e-08],
#         [9.75074291e-09],
#         [7.91302928e-09],
#         [3.77296759e-08],
#         [1.87365749e-06],
#         [5.31140501e-09],
#         [4.92763407e-09],
#         [7.58627492e-09],
#         [2.94600223e-08],
#         [1.13023865e-08],
#         [0.00000000e00],
#         [8.91223890e-09],
#         [0.00000000e00],
#         [7.57119510e-09],
#         [4.87602251e-09],
#         [1.15711844e-06],
#         [1.08619175e-07],
#         [1.09837964e-08],
#         [2.49040751e-07],
#         [3.07446548e-08],
#         [3.53751111e-08],
#         [1.40013010e-08],
#         [7.92877632e-09],
#         [9.48787489e-09],
#         [4.03137791e-08],
#         [0.00000000e00],
#         [2.99153559e-09],
#         [0.00000000e00],
#         [0.00000000e00],
#         [9.83694994e-09],
#         [4.03137791e-08],
#         [3.48771480e-08],
#         [3.92282314e-09],
#         [0.00000000e00],
#         [9.75074288e-09],
#         [0.00000000e00],
#         [7.76390485e-09],
#         [2.25225266e-08],
#         [4.88824539e-09],
#         [6.42599216e-09],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [1.21343192e-08],
#         [2.56998793e-09],
#         [0.00000000e00],
#         [0.00000000e00],
#         [7.76390484e-09],
#         [0.00000000e00],
#         [5.23390626e-08],
#         [0.00000000e00],
#         [1.32696047e-08],
#         [0.00000000e00],
#         [1.08687730e-08],
#         [0.00000000e00],
#         [3.48222390e-08],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [5.02369046e-08],
#         [1.61410881e-08],
#         [6.89940347e-09],
#         [1.57607749e-08],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [1.10346378e-07],
#         [0.00000000e00],
#         [0.00000000e00],
#         [4.62616898e-09],
#         [2.61474873e-08],
#         [2.25225439e-08],
#         [5.29081614e-09],
#         [0.00000000e00],
#         [0.00000000e00],
#         [3.43902499e-08],
#         [4.62493625e-09],
#         [4.26566642e-09],
#         [1.03232436e-08],
#         [5.17854473e-09],
#         [3.03939556e-09],
#         [1.57607752e-08],
#         [1.39623812e-08],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [2.82626007e-08],
#         [1.17947086e-08],
#         [0.00000000e00],
#         [2.49004448e-08],
#         [0.00000000e00],
#         [1.00522380e-07],
#         [9.84979500e-09],
#         [1.38311590e-08],
#         [9.16130468e-09],
#         [0.00000000e00],
#         [0.00000000e00],
#         [0.00000000e00],
#         [9.27947443e-09],
#         [0.00000000e00],
#         [8.70421941e-09],
#         [2.99751314e-08],
#         [2.90517022e-09],
#         [0.00000000e00],
#         [0.00000000e00],
#         [4.19261256e-09],
#         [0.00000000e00],
#         [1.13023865e-08],
#         [0.00000000e00],
#         [5.31140503e-09],
#         [0.00000000e00],
#         [3.07446556e-08],
#         [8.83572781e-09],
#         [0.00000000e00],
#         [4.08200527e-09],
#         [0.00000000e00],
#         [8.55429242e-09],
#         [1.37168633e-08],
#         [0.00000000e00],
#         [0.00000000e00],
#         [9.95093143e-09],
#         [0.00000000e00],
#         [5.21803898e-09],
#         [5.63873569e-09],
#         [6.16848751e-09],
#         [0.00000000e00],
#         [3.53751107e-08],
#         [1.30117645e-08],
#         [7.58627497e-09],
#         [1.27772167e-08],
#         [0.00000000e00],
#         [2.94600220e-08],
#         [0.00000000e00],
#         [0.00000000e00],
#     ]
# )


# var_flat = var.flatten()

# sorted_indices = np.argsort(var_flat)[::-1]
# top_50_indices = sorted_indices[:50]
# top_50_X_test = X_test[top_50_indices]

# # Stampa i risultati
# print(f"Top 50 variances: {var_flat[top_50_indices]}")
# print(f"Top 50 X_test samples: {top_50_X_test}")

# # Estrazione delle prime 3 colonne per ogni array
# x1, y, z1 = (
#     top_50_X_test[:, 0],
#     top_50_X_test[:, 1],
#     top_50_X_test[:, 2],
# )  # Dati per il primo array
# x2, y2, z2 = X_test[:, 0], X_test[:, 1], X_test[:, 2]  # Dati per il secondo array

# # Creazione del plot 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # Scatter plot per il primo array (rosso)
# ax.scatter(x1, y, z1, c="r", marker="x", label="Dati 1")

# # Scatter plot per il secondo array (blu)
# ax.scatter(x2, y2, z2, c="b", marker=".", label="Dati 2")

# # Etichette degli assi
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# # Titolo
# ax.set_title("3D Scatter Plot di due gruppi di dati")

# # Legenda
# ax.legend()

# # Mostra il grafico
# plt.show()


# input_cpacs_name = "provaDellaProva.xml"
# directory_path = "/wrk/Gronda/labAR/prove_codice"
# input_cpacs_path = os.path.join(directory_path, input_cpacs_name)

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


# type_mesh = ["Euler"]  # type_mesh
# symmetry = ["True"]  # symmetry


# # Parametri mesh Euleriana
# farfield_factors = [10.0]
# mesh_farfields = [3.0]
# fuselage_factors = [6]
# wing_factors = [26.0]
# n_power_factors = [2.0]
# n_power_fields = [1.0]
# le_te_layers = [14]
# refine_truncated = ["False"]  # refine_truncated
# auto_refine = ["True"]  # auto_refine

# # SU2
# config_type = type_mesh
# aeromap = ["aeromap_empty"]  # aeroMapUID
# derivatives = ["False"]  # calculateDampingDerivatives
# rotation = [1.0]  # rotationRate
# control_surfaces = ["False"]  # calculateControlSurfacesDeflections
# includeActuatorDisk = []
# cpu = [9]  # nbCPU
# iters = [800]  # maxIter
# cfl_adption = ["True"]  # value
# cflAdFactorDown = [0.5]  # factor_down
# cflAdFactorUp = [1.5]  # factor_up
# cflMinValue = [0.5]  # min
# cflMaxValue = [100.0]  # max
# cfl_value = [1]  # cfl value
# multiGrid = [3]  # multigridLevel
# wettedArea = ["True"]  # updateWettedArea
# extraLoads = ["False"]  # extractLoads


# # Conversion of lists of dictionaries to simple dictionaries

# # Common mesh parameters
# common_mesh_params = {
#     "type_mesh": type_mesh,
#     "symmetry": symmetry,
# }

# aeromap_name = "new_aeromap"

# # Euler mesh parameters
# euler_mesh_params = {
#     "farfield_factor": farfield_factors,
#     "mesh_size/farfield": mesh_farfields,
#     "mesh_size/fuselage/factor": fuselage_factors,
#     "mesh_size/wings/factor": wing_factors,
#     "n_power_factor": n_power_factors,
#     "n_power_field": n_power_fields,
#     "number_layer": le_te_layers,
#     "refine_truncated": refine_truncated,
#     "auto_refine": auto_refine,
# }

# # SU2 parameters
# su2_params = {
#     "aeroMapUID": aeromap,
#     "options/calculateDampingDerivatives": derivatives,
#     "options/config_type": config_type,
#     "options/rotationRate": rotation,
#     "options/calculateControlSurfacesDeflections": control_surfaces,
#     "settings/nbCPU": cpu,
#     "settings/maxIter": iters,
#     "settings/cflNumber/value": cfl_value,
#     "settings/cflNumber/adaptation/value": cfl_adption,
#     "settings/cflNumber/adaptation/factor_down": cflAdFactorDown,
#     "settings/cflNumber/adaptation/factor_up": cflAdFactorUp,
#     "settings/cflNumber/adaptation/min": cflMinValue,
#     "settings/cflNumber/adaptation/max": cflMaxValue,
#     "settings/multigridLevel": multiGrid,
#     "results/updateWettedArea": wettedArea,
#     "results/extractLoads": extraLoads,
# }

# tixi = Tixi3()
# tixi.open(input_cpacs_path)
# avl_update(tixi, aeromap_name, avl_parameters)
# tixi.save(input_cpacs_path)
# tixi.close()


# tixi = Tixi3()
# tixi.open(input_cpacs_path)
# change_reference_value(tixi, reference_values)
# euler_update(tixi, aeromap_name, common_mesh_params, euler_mesh_params, su2_params)
# tixi.save(input_cpacs_path)
# tixi.close()

# # # TRAIN KRIGING
# first_kriging_dataset_path = ""  # aggiungendo questo va quello di default
# train_first_krigin_path = f"{first_kriging_dataset_path}"
# default_first_kriging_dataset_path = (
#     "/wrk/Gronda/labAR/prove_codice/LHS_dataset_TRAIN_default2.csv"
# )
# df = load_and_split_data(train_first_krigin_path, default_first_kriging_dataset_path)
# print(df)

# # Normalize data
# normalized_data1 = normalize_data(df)

# df_norm1 = normalized_data1["dataset"]["df"]
# X_norm1 = normalized_data1["dataset"]["X_normalized"]
# y_norm1 = normalized_data1["dataset"]["y_normalized"]
# cl_norm1 = y_norm1["CL"]

# # Split test and training

# train_test_values1 = test_training_data(X_norm1, cl_norm1)

# X_train1 = train_test_values1["X_train"]
# X_test1 = train_test_values1["X_test"]
# y_train1 = train_test_values1["y_train"]
# y_test1 = train_test_values1["y_test"]


# base_path = "/wrk/Gronda/labAR/prove_codice"
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
#     data = extract_coefficients_from_SU2(results_path)
#     print(data)
#     kriging_dataset_path = append_to_new_csv(data, csv_filename2)
# else:
#     print(f"Errore: La directory {latest_workflow_path} non esiste.")

# print(kriging_dataset_path)


# # # TRAIN MFKRIGING
# # train_dataset_path = ""  # aggiungendo questo va quello di default
# train_second_krigin_path = f"{kriging_dataset_path}"
# default_train_kriging_dataset_path = (
#     "/wrk/Gronda/labAR/prove_codice/EULER_dataset_TRAIN_default.csv"
# )
# df2 = load_and_split_data(train_second_krigin_path, default_train_kriging_dataset_path)
# print(df2)

# # Normalize data
# normalized_data = normalize_data(df2)

# df_norm2 = normalized_data["dataset"]["df"]
# X_norm2 = normalized_data["dataset"]["X_normalized"]
# y_norm2 = normalized_data["dataset"]["y_normalized"]
# cl_norm2 = y_norm2["CL"]

# # Split test and training

# train_test_values2 = test_training_data(X_norm2, cl_norm2)

# X_train2 = train_test_values2["X_train"]
# X_test2 = train_test_values2["X_test"]
# y_train2 = train_test_values2["y_train"]
# y_test2 = train_test_values2["y_test"]

# # Train module and predict

# theta2 = [0.01]
# corr2 = "matern32"
# poly2 = "constant"  # linear e quadratic  danno problemi coi dati normalizzati

# model2 = MF_Kriging(X_train1, y_train1, X_train2, y_train2, theta2, corr2, poly2)
# predictions2 = predict_mf_model(model2, X_test2, y_test2)

# y_pred2 = predictions2["y_pred"]
# var2 = predictions2["variance"]

# print(f"x train: {X_train2}")
# print(f"x test: {X_test2}")
# print(f"y test: {y_test2}")
# print(f"y pred: {y_pred2}")
# print(f"variance: {var2}")

# # EVALUATE HIGH VARIANCE DATA AND SAMPLING

# var_flat2 = var2.flatten()

# sorted_indices2 = np.argsort(var_flat2)[::-1]
# top_10_indices1 = sorted_indices2[:10]
# top_10_X_test1 = X_test2[top_10_indices1]

# # Stampa i risultati
# print(f"Top 50 variances: {var_flat2[top_10_indices1]}")
# print(f"Top 50 X_test samples: {top_10_X_test1}")

# scaler_X2 = normalized_data["scalers"]["scaler_X"]

# X_original2 = scaler_X2.inverse_transform(top_10_X_test1)
# print(f"Original top 50: {X_original2}")

# altitude2 = X_original2[:, 0]
# machNumber2 = X_original2[:, 1]
# angleOfAttack2 = X_original2[:, 2]
# angleOfSideslip2 = X_original2[:, 3]
# print(altitude2)
# print(machNumber2)
# print(angleOfAttack2)
# print(angleOfSideslip2)

# # dictionary aeromap
# aeromap_columns2 = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]
# aeromap2 = {name: X_original2[:, i] for i, name in enumerate(aeromap_columns2)}

# for key, value in aeromap2.items():
#     print(f"{key}: {value}")

# print(aeromap2)
# print(type(aeromap2))
# # miglioramento: aggiungere una funzione di densita per evitare che i putni siano troppo vicini
# # miglioramento: confrontare ad ogni step i modelli kriging x valutare un effettivo miglioramento

# output_filename3 = "RANS2_dataset.csv"
# output_directory3 = "/wrk/Gronda/labAR/prove_codice"
# full_path3 = os.path.join(output_directory3, output_filename3)

# save_to_csv(aeromap2, full_path3)


def launch_avl_simulations(
    default_first_kriging_dataset_path, directory_path, input_cpacs_path, full_path1
):

    first_kriging_dataset_path = None  # Initialize with None

    # LAUNCH AVL COMMAND
    print("CPACS updated, running PyAVL Module in CEASIOMpy...")
    command = (
        f"cd {os.path.abspath(directory_path)} && "
        f"ceasiompy_run -m {os.path.abspath(input_cpacs_path)} PyAVL"
    )

    try:
        # Run the command with subprocess.run()
        print("PyAVL simulation started. Press Ctrl+C to interrupt manually.")
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=sys.stdout,  # Forward standard output to the terminal
            stderr=sys.stderr,  # Forward standard error to the terminal
        )

        # Check if the process completed successfully
        if result.returncode == 0:
            print("Simulations completed successfully!")
        else:
            print("An error occurred during the simulations!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during the simulation: {e}")
        # Use the default dataset if there was an error
        first_kriging_dataset_path = default_first_kriging_dataset_path
        result = None  # Ensure result is defined
    except KeyboardInterrupt:
        print("\nSimulation manually interrupted.")
        # Manually set to default dataset if interrupted
        first_kriging_dataset_path = default_first_kriging_dataset_path
        result = None  # Ensure result is defined

    # PROCESS SIMULATION RESULTS OR USE DEFAULT DATASET
    if result and result.returncode == 0:
        # If the process completed, analyze the results
        latest_workflow_path = get_latest_workflow(directory_path)
        if latest_workflow_path:
            results_path = os.path.join(latest_workflow_path, "Results", "PyAVL")
            print("Latest Workflow:", latest_workflow_path)
            print("Results Path:", results_path)

            if os.path.isdir(results_path):
                data1 = extract_coefficients_from_AVL(results_path)
                print("Coefficents got from AVL simulations:")
                print(data1)
                first_kriging_dataset_path = append_to_new_csv(data1, full_path1)
            else:
                print(f"Error: The directory {results_path} does not exist.")
        else:
            print("No workflow found.")
    else:
        # Use the default dataset if the process was interrupted
        print(f"Using the default dataset: {default_first_kriging_dataset_path}")
        first_kriging_dataset_path = default_first_kriging_dataset_path

    return first_kriging_dataset_path


def avl_workflow(
    input_cpacs_path,
    directory_path,
    default_first_kriging_dataset_path,
    full_path1,
    samples,
    aeromap_uid,
    aeromap_name,
    avl_parameters,
):

    print("Do you want to proceed with AVL simulations? [default: NO]")
    print("If YES: proceed with AVL configuration")
    print("If NO: insert file.csv to train first kriging")
    avl_yes_or_not = input(": ") or "NO"

    if avl_yes_or_not.upper() == "YES":
        print("Updating aeromap and reference value for AVL simulations")
        input("Press ENTER to continue....")

        # Aeromap updating on CPACS
        tixi = Tixi3()
        tixi.open(input_cpacs_path)

        try:
            # add the new aeroMap
            add_new_aeromap(tixi, samples, aeromap_uid, aeromap_name)
            # change_reference_value(tixi, reference_values)
            # save the updated CPACS file
            tixi.save(input_cpacs_path)
            print("New aeroMap added successfully!")
        except Exception as e:
            print(f"Error adding aeroMap: {e}")
        finally:
            tixi.close()

        # AVL updating on CPACS
        print("Updating parameters for AVL simulations")
        input("Press ENTER to continue....")

        tixi = Tixi3()
        tixi.open(input_cpacs_path)

        try:
            # avl update
            avl_update(tixi, aeromap_name, avl_parameters)
            # save the updated CPACS file
            tixi.save(input_cpacs_path)
            print("AVL parameters updated successfully!")
        except Exception as e:
            print(f"Error updating parameters: {e}")
        finally:
            tixi.close()

        input("Press ENTER to continue....")

        # Obtain path of train dataset
        first_kriging_dataset_path = launch_avl_simulations(
            default_first_kriging_dataset_path, directory_path, input_cpacs_path, full_path1
        )

    else:

        first_kriging_dataset_path = input("Insert file.csv path: ")

        if not first_kriging_dataset_path:  # Use default path if no path is provided
            print(f"No path given. Using default path: {default_first_kriging_dataset_path}")
            first_kriging_dataset_path = default_first_kriging_dataset_path

    return first_kriging_dataset_path


def sm_workflow(
    kriging_dataset_path,
    directory_path,
    theta,
    corr,
    poly,
    selected_mach,
    altitude,
    aos,
    n_samples,
    fidelity_level,
    fraction_of_new_samples=None,
    ranges=None,
    processed_samples=None,
    coefficent_to_predict=None,
    X_train_LF=None,
    y_train_LF=None,
    X_train_MF=None,
    y_train_MF=None,
    base_model_name=None,
    model_extension=None,
):
    """
    Workflow for training surrogate models with support for multiple fidelity levels.

    Parameters:
    - fidelity_level: int (1, 2, 3) - The number of fidelity levels the model should have.
    """

    # Validate input
    if not isinstance(fidelity_level, int) or fidelity_level not in [1, 2, 3]:
        raise ValueError("fidelity_level must be an integer (1, 2, or 3).")

    # Load and prepare the dataset (only for LF)
    df = load_and_split_data(kriging_dataset_path)
    X = df["dataset"]["X"]
    y = df["dataset"]["y"]

    # Select coefficient to predict
    if coefficent_to_predict is None:
        which_coefficent = (
            input("Insert which coefficient to predict (CL, CD, CM) [default: CL]: ") or "CL"
        )
    else:
        which_coefficent = coefficent_to_predict

    if which_coefficent not in y:
        raise KeyError(
            f"The specified coefficient '{which_coefficent}' is not available in the dataset."
        )

    coefficent = y[which_coefficent]

    # Split data into training and test sets (LF data)
    train_test_values = test_training_data(X, coefficent)
    X_train = train_test_values["X_train"]
    X_test = train_test_values["X_test"]
    y_train = train_test_values["y_train"]
    y_test = train_test_values["y_test"]

    print("Data splitted into training and test sets.")

    # Initialize variables for iteration
    model = None
    top_n_X_test = None

    # Training based on fidelity_level
    if fidelity_level == 1:
        print(f"Training surrogate model...")
        model = Kriging(X_train, y_train, theta, corr, poly)
        # Prediction metrics and graphs
        rms, prediction, y_pred, var = prediction_metrics_plots(
            model,
            X_test,
            y_test,
            which_coefficent,
            altitude,
            aos,
            X_train,
            y_train,
            ranges["machNumber"],
            ranges["angleOfAttack"],
            selected_mach,
        )
        input("Press ENTER to continue: ")

        # Saving of the model
        print("Saving model...")
        save_model(model, directory_path, base_model_name, model_extension)

    elif fidelity_level == 2:
        if X_train_LF is None and y_train_LF is None:
            # First iteration
            print(f"Training first surrogate model...")
            model = Kriging(X_train, y_train, theta, corr, poly)
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            output_filename = "EULER_dataset.csv"

            new_aeromap, full_path = high_variance_new_doe(
                var,
                n_samples,
                fraction_of_new_samples,
                X_test,
                processed_samples,
                ranges,
                output_filename,
                directory_path,
            )

        else:
            # Second iteration
            print("Training final multi fidelity surrogate model...")
            model = MF_Kriging(X_train_LF, y_train_LF, X_train, y_train, theta, corr, poly)
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            input("Press ENTER to continue: ")

            # Saving of the model
            print("Saving model...")
            save_model(model, directory_path, base_model_name, model_extension)

    else:
        if X_train_LF is None and y_train_LF is None and X_train_MF is None and y_train_MF is None:
            # First iteration
            print(f"Training first surrogate model...")
            model = Kriging(X_train, y_train, theta, corr, poly)
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            output_filename = "EULER_dataset.csv"

            new_aeromap, full_path = high_variance_new_doe(
                var,
                n_samples,
                fraction_of_new_samples,
                X_test,
                processed_samples,
                ranges,
                output_filename,
                directory_path,
            )

        elif X_train_MF is None and y_train_MF is None:
            # Second iteration
            print(f"Training first multy fidelity surrogate model...")
            model = MF_Kriging(X_train_LF, y_train_LF, X_train, y_train, theta, corr, poly)
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            output_filename = "RANS_dataset.csv"

            # Fraction for RANS should be different
            new_aeromap, full_path = high_variance_new_doe(
                var,
                n_samples,
                fraction_of_new_samples,
                X_test,
                processed_samples,
                ranges,
                output_filename,
                directory_path,
            )

        else:
            # Third iteration
            print("Training final multi fidelity surrogate model...")
            model = MF_Kriging(
                X_train_LF, y_train_LF, X_train_MF, y_train_MF, theta, corr, poly, X_train, y_train
            )
            # Prediction metrics and graphs
            rms, prediction, y_pred, var = prediction_metrics_plots(
                model,
                X_test,
                y_test,
                which_coefficent,
                altitude,
                aos,
                X_train,
                y_train,
                ranges["machNumber"],
                ranges["angleOfAttack"],
                selected_mach,
            )

            input("Press ENTER to continue: ")
            # Saving of the model
            print("Saving model...")
            save_model(model, directory_path, base_model_name, model_extension)

    return new_aeromap, full_path, which_coefficent, top_n_X_test, model, rms, X_train, y_train


def prediction_metrics_plots(
    model,
    X_test,
    y_test,
    which_coefficent,
    altitude,
    aos,
    X_train,
    y_train,
    mach,
    aoa,
    selected_mach,
):

    # Prediction and metrics
    rms = compute_rms_error(model, X_test, y_test)
    predictions = predict_model(model, X_test, y_test)
    y_pred = predictions["y_pred"]
    var = predictions["variance"]

    print(f"RMS Error: {rms}")

    # Plot validation and response surfaces
    plot_validation(y_test, y_pred, which_coefficent)
    plot_response_surface(altitude, aos, X_train, y_train, model, which_coefficent, mach, aoa)
    plot_coefficent_alpha_for_mach(X_train, y_train, model, selected_mach, which_coefficent)

    return rms, predictions, y_pred, var


def high_variance_new_doe(
    var,
    n_samples,
    fraction_of_new_samples,
    X_test,
    processed_samples,
    ranges,
    output_filename,
    directory_path,
):

    print("Selecting DOE points with highest variance...")
    var_flat = var.flatten()
    sorted_indices = np.argsort(var_flat)[::-1]
    n_new_samples = n_samples // fraction_of_new_samples
    top_n_indices = sorted_indices[:n_new_samples]
    top_n_X_test = X_test[top_n_indices]

    # Print results
    print(f"Top {n_new_samples} variances: {var_flat[top_n_indices]}")
    print(f"Top {n_new_samples} X_test samples: {top_n_X_test}")

    # Plot DOE highlighting new points
    plot_doe(
        processed_samples,
        ranges,
        n_samples=n_new_samples,
        plot_dim1="angleOfAttack",
        plot_dim2="machNumber",
        highlight_points=top_n_X_test,
    )

    input("Press ENTER to continue: ")

    new_aeromap = {key: top_n_X_test[:, idx] for idx, key in enumerate(ranges.keys())}

    for key, value in new_aeromap.items():
        print(f"{key}: {value}")

    full_path = os.path.join(directory_path, output_filename)
    save_to_csv(new_aeromap, full_path)

    return new_aeromap, full_path


def launch_SU2_simulations(
    default_kriging_dataset_path, directory_path, input_cpacs_path, full_path
):

    kriging_dataset_path = None  # Initialize with None

    # Launch SU2
    print("CPACS updated, running GMSH and SU2 Module in CEASIOMpy...")
    command = (
        f"cd {os.path.abspath(directory_path)} && "
        f"ceasiompy_run -m {os.path.abspath(input_cpacs_path)} CPACS2GMSH SU2Run"
    )

    try:
        print("Simulations started. Press Ctrl+C to interrupt manually.")
        # Run the command with subprocess.run()
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=sys.stdout,  # Forward standard output to the terminal
            stderr=sys.stderr,  # Forward standard error to the terminal
        )

        # Check if the process completed successfully
        if result.returncode == 0:
            print("Simulations completed successfully!")
        else:
            print("An error occurred during the simulations!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during the simulation: {e}")
        # Use the default dataset if there was an error
        kriging_dataset_path = default_kriging_dataset_path
        result = None  # Ensure result is defined
    except KeyboardInterrupt:
        print("\nSimulation manually interrupted.")
        # Manually set to default dataset if interrupted
        kriging_dataset_path = default_kriging_dataset_path
        result = None  # Ensure result is defined

    # PROCESS SIMULATION RESULTS OR USE DEFAULT DATASET
    if result and result.returncode == 0:
        # If the process completed, analyze the results
        latest_workflow_path = get_latest_workflow(directory_path)
        if latest_workflow_path:
            results_path = os.path.join(latest_workflow_path, "Results", "SU2")
            print("Latest Workflow:", latest_workflow_path)
            print("Results Path:", results_path)

            if os.path.isdir(results_path):
                data = extract_coefficients_from_SU2(results_path)
                print(data)
                kriging_dataset_path = append_to_new_csv(data, full_path)
            else:
                print(f"Error: The directory {results_path} does not exist.")
        else:
            print("No workflow found.")
    else:
        # Use the default dataset if the process was interrupted
        print(f"Using the default Euler dataset: {default_kriging_dataset_path}")
        kriging_dataset_path = default_kriging_dataset_path

    return kriging_dataset_path


def su2_workflow(
    input_cpacs_path,
    directory_path,
    default_kriging_dataset_path,
    full_path,
    aeromap,
    aeromap_uid,
    aeromap_name,
    common_mesh_params,
    su2_mesh_params,
    su2_params,
    gmsh_options=None,
):

    print("Do you want to proceed with SU2 simulations? [default: NO]")
    print("If YES: proceed with SU2 configuration")
    print("If NO: insert file.csv to train Multi-Fidelity Kriging")
    su2_yes_or_not = input(": ") or "NO"

    if su2_yes_or_not.upper() == "YES":
        print("Updating aeromap and reference value for SU2 simulations")
        input("Press ENTER to continue....")

        # Aeromap updating on CPACS
        tixi = Tixi3()
        tixi.open(input_cpacs_path)

        try:
            # add the new aeroMa
            add_new_aeromap(tixi, aeromap, aeromap_uid, aeromap_name)
            # save the updated CPACS file
            tixi.save(input_cpacs_path)
            print("New aeroMap added successfully!")
        except Exception as e:
            print(f"Error adding aeroMap: {e}")
        finally:
            tixi.close()

        # SU2 updating on CPACS
        print("Updating parameters for SU2 simulations")
        input("Press ENTER to continue....")

        tixi = Tixi3()
        tixi.open(input_cpacs_path)

        try:
            # Euler update
            SU2_update(tixi, aeromap_name, common_mesh_params, su2_mesh_params, su2_params)
            # save the updated CPACS file
            tixi.save(input_cpacs_path)
            print("SU2 parameters updated successfully!")
        except Exception as e:
            print(f"Error updating parameters: {e}")
        finally:
            tixi.close()

        input("Press ENTER to continue....")

        # Obtain path of train dataset
        kriging_dataset_path = launch_SU2_simulations(
            default_kriging_dataset_path, directory_path, input_cpacs_path, full_path
        )

    else:

        kriging_dataset_path = input("Insert file.csv path: ")

        if not kriging_dataset_path:  # Use default path if no path is provided
            print(f"No path given. Using default path: {default_kriging_dataset_path}")
            kriging_dataset_path = default_kriging_dataset_path

    return kriging_dataset_path
