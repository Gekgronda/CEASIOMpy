import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from ceasiompy.SMTrain_new.codes.gg_sm2 import (
    latin_hypercube_sampling,
    get_model_params,
    match_outputs,
    add_noise,
    match_noisy_outputs,
    # noise da implementare
    predict_model,
    split_data,
    evaluate_model,
    plot_predictions,
    combine_models,
    # optimize_hyperparameters_ego,
    save_model,
    compare_models,
    train_surrogate_model,
)

# from ceasiompy.SMTrain_new.hyperopt_prova import optimize_hyperparameters_hyperopt


# def get_user_inputs():
#     """Ottiene i parametri dall'utente in base alle opzioni selezionate."""

#     # Applicare il campionamento?
#     use_sampling_input = (
#         input("Do you want to apply Latin Hypercube Sampling (yes/no)? [default=no]: ") or "no"
#     )
#     if use_sampling_input.lower() not in ["yes", "no"]:
#         print("Invalid input for sampling. Defaulting to 'no'.")
#         use_sampling = False
#     else:
#         use_sampling = use_sampling_input.lower() == "yes"

#     # Aggiungere rumore al dataset?
#     add_noise_input = (
#         input(
#             "Do you want to add noise to the dataset to increase data points (yes/no)? [default=no]: "
#         )
#         or "no"
#     )
#     if add_noise_input.lower() not in ["yes", "no"]:
#         print("Invalid input for noise addition. Defaulting to 'no'.")
#         add_noise = False
#     else:
#         add_noise = add_noise_input.lower() == "yes"

#     # Mostrare i grafici dei risultati?
#     show_plots_input = (
#         input("Do you want to display result plots (yes/no)? [default=yes]: ") or "yes"
#     )
#     if show_plots_input.lower() not in ["yes", "no"]:
#         print("Invalid input for displaying plots. Defaulting to 'yes'.")
#         show_plots = True
#     else:
#         show_plots = show_plots_input.lower() == "yes"

#     # Selezione del modello

#     model_type_cl = (
#         input(
#             "Choose model type for cl prediction (Kriging, KPLS, KPLSK, Linear, Quadratic, IDW, RBF, RMTB, RMTC) [default=Kriging]: "
#         )
#         or "Kriging"
#     )
#     if model_type_cl not in [
#         "Kriging",
#         "KPLS",
#         "KPLSK",
#         "Linear",
#         "Quadratic",
#         "IDW",
#         "RBF",
#         "RMTB",
#         "RMTC",
#     ]:
#         print("Invalid model type for CL. Setting to default 'Kriging'.")
#         model_type_cl = "Kriging"

#     model_type_cd = (
#         input(
#             "Choose model type for cd prediction (Kriging, KPLS, KPLSK, Linear, Quadratic, IDW, RBF, RMTB, RMTC) [default=Kriging]: "
#         )
#         or "Kriging"
#     )
#     if model_type_cd not in [
#         "Kriging",
#         "KPLS",
#         "KPLSK",
#         "Linear",
#         "Quadratic",
#         "IDW",
#         "RBF",
#         "RMTB",
#         "RMTC",
#     ]:
#         print("Invalid model type for CD. Setting to default 'Kriging'.")
#         model_type_cd = "Kriging"

#     # Raccogli i parametri per ciascun modello
#     print("Choice the paramenter of the cl model")
#     params_cl = get_model_params(model_type_cl)
#     print("Choice the paramenter of the cl model")
#     params_cd = get_model_params(model_type_cd)

#     return {
#         "use_sampling": use_sampling,
#         "add_noise": add_noise,
#         "show_plots": show_plots,
#         "model_type_cl": model_type_cl,
#         "model_type_cd": model_type_cd,
#         "params_cl": params_cl,
#         "params_cd": params_cd,
#     }


# MULRIFIDELITY KRIGNIG

# # HF database
# hf_df = pd.read_csv("/wrk/Gronda/labAR/RANS/RANS_train_dataset.csv")
# # LF database
# lf_df = pd.read_csv("/wrk/Gronda/labAR/EULER/takeoff0/takeoff_train.csv")

# # input and output hf
# X_hf = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
# y_cl_hf = df["Total CL"].values
# y_cd_hf = df["Total CD"].values

# # input and output lf
# X_hf = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
# y_cl_hf = df["Total CL"].values
# y_cd_hf = df["Total CD"].values

# # sampling


# # split data
# (
#     X_train_hf,
#     X_val_hf,
#     X_test_hf,
#     y_cl_train_hf,
#     y_cl_val_hf,
#     y_cl_test_hf,
#     y_cd_train_hf,
#     y_cd_val_hf,
#     y_cd_test_hf,
# ) = split_data(X_hf, y_cl_hf, y_cd_hf)
# X_train_hf, X_val_hf, X_test, y_cl_train, y_cl_val, y_cl_test, y_cd_train, y_cd_val, y_cd_test = (
#     split_data(X_hf, y_cl_hf, y_cd_hf)
# )


# Carica il database
name = input("Insert database path (with .csv extention): ").strip()
if not name:
    name = "updated_takeoff0_TRAIN.csv"
database_path = input("Insert database filepath: ").strip()
if not database_path:
    database_path = "/wrk/Gronda/labAR/EULER/takeoff0/"
elif not database_path.endswith("/"):
    database_path += "/"

file_path = os.path.join(database_path, name)
print("Selected file_path is: ", file_path)
df = pd.read_csv(file_path)

# Definisci gli input e output
X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values
y_cs = df["Total CSF"].values
y_cmx = df["Total CMx"].values
y_cmy = df["Total CMy"].values
y_cmz = df["Total CMz"].values


# Step 3: Chiede se fare sampling o aggiunta rumore,
# Se necessario, aggiorna il dataframe con campionamento o rumore (da migliorare)
apply_sampling = (
    input("Do you want to apply Latin Hypercube Sampling (yes/no)? [default=no]: ") or "no"
)
if apply_sampling.lower() not in ["yes", "no"]:
    print("Invalid input for sampling. Defaulting to 'no'.")
    apply_sampling = "no"
apply_sampling = apply_sampling.lower() == "yes"

add_noise = input("Do you want to add noise to the dataset (yes/no)? [default=no]: ") or "no"
if add_noise.lower() not in ["yes", "no"]:
    print("Invalid input for noise addition. Defaulting to 'no'.")
    add_noise = "no"
add_noise = add_noise.lower() == "yes"

if apply_sampling:
    num_samples = int(
        input("Enter the number of samples: ")
    )  # Aggiungi input per numero di campioni
    X_lhs = latin_hypercube_sampling(X, num_samples)
    y_cl, y_cd, y_cs, y_cmx, y_cmy, y_cmz = match_outputs(
        X_lhs, X, y_cl, y_cd, y_cs, y_cmx, y_cmy, y_cmz
    )
    X = X_lhs  # Aggiorna X_train con i campioni LHS
if add_noise:
    noise_level = float(
        input("Enter the noise level as a percentage of std deviation (e.g., 0.05 for 5%): ")
    )
    num_samples = int(input("Enter the number of noisy samples: "))
    X_noisy, _, _, _, _, _, _ = add_noise(
        X, y_cl, y_cd, y_cs, y_cmx, y_cmy, y_cmz, noise_level=noise_level, num_samples=num_samples
    )

    # Opzionale: Abbina i campioni rumorosi ai valori di output originali
    y_cl_noisy, y_cd_noisy, y_cs_noisy, y_cmx_noisy, y_cmy_noisy, y_cmz_noisy = (
        match_noisy_outputs(X_noisy, X, y_cl, y_cd, y_cs, y_cmx, y_cmy, y_cmz)
    )

    # Aggiorna i dati di training
    X_train = X_noisy
    y_cl = y_cl_noisy
    y_cd = y_cd_noisy
    y_cs = y_cs_noisy
    y_cmx = y_cmx_noisy
    y_cmy = y_cmy_noisy
    y_cmz = y_cmz_noisy


# Step 4: suddividi i dati per addestramento, validazione e test

(
    X_train,
    X_val,
    X_test,
    y_cl_train,
    y_cl_val,
    y_cl_test,
    y_cd_train,
    y_cd_val,
    y_cd_test,
    y_cs_train,
    y_cs_val,
    y_cs_test,
    y_cmx_train,
    y_cmx_val,
    y_cmx_test,
    y_cmy_train,
    y_cmy_val,
    y_cmy_test,
    y_cmz_train,
    y_cmz_val,
    y_cmz_test,
) = split_data(X, y_cl, y_cd, y_cs, y_cmx, y_cmy, y_cmz)

print("X_test shape:", X_test.shape)
print("y_cl_test shape:", y_cl_test.shape)
print("X_train shape:", X_train.shape)
print("y_cl_train shape:", y_cl_train.shape)

y_cl_train = y_cl_train.reshape(-1, 1)  # Aggiunge una dimensione (n_samples, 1)
y_cl_test = y_cl_test.reshape(-1, 1)  # Aggiunge una dimensione (n_samples, 1)

# Step 5: Valuta e suggerisce il modello migliore per tutti i target
use_moe = (
    input(
        "Do you want to use MOE algorithm to find best surrogate models for cl, cd, cs, cmx, cmy, and cmz (yes/no)? [default=no]: "
    )
    or "no"
)
if use_moe.lower() not in ["yes", "no"]:
    print("Invalid input for MOE use. Defaulting to 'no'.")
    use_moe = "no"
use_moe = use_moe.lower() == "yes"

if use_moe:
    targets = ["cl", "cd", "cs", "cmx", "cmy", "cmz"]
    y_train_test = {
        "cl": (y_cl_train, y_cl_test),
        "cd": (y_cd_train, y_cd_test),
        "cs": (y_cs_train, y_cs_test),
        "cmx": (y_cmx_train, y_cmx_test),
        "cmy": (y_cmy_train, y_cmy_test),
        "cmz": (y_cmz_train, y_cmz_test),
    }

    # Reshape e valutazione modelli (ciclo cosi accorciamo sto codice)
    for target in targets:
        y_train, y_test = y_train_test[target]
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        print(f"Finding best expert for {target} model...")
        compare_models(X_test, y_test, X_train, y_train)


print("=============================================================================")
print("WARNING: IDW, RBF, RMTB and RMTC ARE WRITTEN IN C++ SO THEY CAN'T BE SAVED")
print("=============================================================================")

# Step 6: Chiede all'utente quale modello usare per ciascun target
valid_models = {"Kriging", "KPLS", "KPLSK", "Linear", "Quadratic", "IDW", "RBF", "RMTB", "RMTC"}

model_type_cl = (
    input(
        "Choose model type for cl prediction (Kriging, KPLS, KPLSK, Linear, Quadratic, IDW, RBF, RMTB, RMTC) [default=Kriging]: "
    )
    or "Kriging"
)
if model_type_cl not in valid_models:
    print("Invalid model type for CL. Setting to default 'Kriging'.")
    model_type_cl = "Kriging"

model_type_cd = (
    input(
        "Choose model type for cd prediction (Kriging, KPLS, KPLSK, Linear, Quadratic, IDW, RBF, RMTB, RMTC) [default=Kriging]: "
    )
    or "Kriging"
)
if model_type_cd not in valid_models:
    print("Invalid model type for CD. Setting to default 'Kriging'.")
    model_type_cd = "Kriging"

model_type_cs = (
    input(
        "Choose model type for cs prediction (Kriging, KPLS, KPLSK, Linear, Quadratic, IDW, RBF, RMTB, RMTC) [default=Kriging]: "
    )
    or "Kriging"
)
if model_type_cs not in valid_models:
    print("Invalid model type for CS. Setting to default 'Kriging'.")
    model_type_cs = "Kriging"

model_type_cmx = (
    input(
        "Choose model type for cmx prediction (Kriging, KPLS, KPLSK, Linear, Quadratic, IDW, RBF, RMTB, RMTC) [default=Kriging]: "
    )
    or "Kriging"
)
if model_type_cmx not in valid_models:
    print("Invalid model type for CMx. Setting to default 'Kriging'.")
    model_type_cmx = "Kriging"

model_type_cmy = (
    input(
        "Choose model type for cmy prediction (Kriging, KPLS, KPLSK, Linear, Quadratic, IDW, RBF, RMTB, RMTC) [default=Kriging]: "
    )
    or "Kriging"
)
if model_type_cmy not in valid_models:
    print("Invalid model type for CMy. Setting to default 'Kriging'.")
    model_type_cmy = "Kriging"

model_type_cmz = (
    input(
        "Choose model type for cmz prediction (Kriging, KPLS, KPLSK, Linear, Quadratic, IDW, RBF, RMTB, RMTC) [default=Kriging]: "
    )
    or "Kriging"
)
if model_type_cmz not in valid_models:
    print("Invalid model type for CMz. Setting to default 'Kriging'.")
    model_type_cmz = "Kriging"

# Step 7: Chiede all'utente se usare l'ottimizzatore o impostare manualmente gli iperparametri
use_optimizer = (
    # input(
    #     "Do you want to optimize hyperparameters (yes/no)?"
    #     "NB.  [default=no]: "
    # )
    # or
    "no"
)
if use_optimizer.lower() not in ["yes", "no"]:
    print("Invalid input for model choice. Defaulting to 'no'.")
    use_optimizer = "no"
use_optimizer = use_optimizer.lower() == "yes"

# Step 8: Ottimizza o richiede gli iperparametri manualmente
if use_optimizer:
    # Ottimizza gli iperparametri per tutti i coefficienti
    params_cl = optimize_hyperparameters_hyperopt(X_train, y_cl_train, X_val, y_cl_val)
    params_cd = optimize_hyperparameters_hyperopt(X_train, y_cd_train, X_val, y_cd_val)
    params_cs = optimize_hyperparameters_hyperopt(X_train, y_cs_train, X_val, y_cs_val)
    params_cmx = optimize_hyperparameters_hyperopt(X_train, y_cmx_train, X_val, y_cmx_val)
    params_cmy = optimize_hyperparameters_hyperopt(X_train, y_cmy_train, X_val, y_cmy_val)
    params_cmz = optimize_hyperparameters_hyperopt(X_train, y_cmz_train, X_val, y_cmz_val)
else:
    params_cl = (
        get_model_params(model_type_cl)
        if model_type_cl not in ["Linear", "Quadratic", "IDW"]
        else {}
    )
    params_cd = (
        get_model_params(model_type_cd)
        if model_type_cd not in ["Linear", "Quadratic", "IDW"]
        else {}
    )
    params_cs = (
        get_model_params(model_type_cs)
        if model_type_cs not in ["Linear", "Quadratic", "IDW"]
        else {}
    )
    params_cmx = (
        get_model_params(model_type_cmx)
        if model_type_cmx not in ["Linear", "Quadratic", "IDW"]
        else {}
    )
    params_cmy = (
        get_model_params(model_type_cmy)
        if model_type_cmy not in ["Linear", "Quadratic", "IDW"]
        else {}
    )
    params_cmz = (
        get_model_params(model_type_cmz)
        if model_type_cmz not in ["Linear", "Quadratic", "IDW"]
        else {}
    )

# Addestra i modelli con i parametri ottimizzati o scelti manualmente
print("Training cl model...")
model_cl = train_surrogate_model(X_train, y_cl_train, model_type_cl, params_cl)
print("Training cd model...")
model_cd = train_surrogate_model(X_train, y_cd_train, model_type_cd, params_cd)
model_cs = train_surrogate_model(X_train, y_cs_train, model_type_cs, params_cs)
model_cmx = train_surrogate_model(X_train, y_cmx_train, model_type_cmx, params_cmx)
model_cmy = train_surrogate_model(X_train, y_cmy_train, model_type_cmy, params_cmy)
model_cmz = train_surrogate_model(X_train, y_cmz_train, model_type_cmz, params_cmz)

# Step 9: Fai previsioni e valuta i modelli
cl_pred = predict_model(model_cl, X_test)
cd_pred = predict_model(model_cd, X_test)
cs_pred = predict_model(model_cs, X_test)
cmx_pred = predict_model(model_cmx, X_test)
cmy_pred = predict_model(model_cmy, X_test)
cmz_pred = predict_model(model_cmz, X_test)

# Valutazione dei modelli
errors_cl = evaluate_model(model_cl, X_test, y_cl_test, cl_pred, "CL")
errors_cd = evaluate_model(model_cd, X_test, y_cd_test, cd_pred, "CD")
errors_cs = evaluate_model(model_cs, X_test, y_cs_test, cs_pred, "CS")
errors_cmx = evaluate_model(model_cmx, X_test, y_cmx_test, cmx_pred, "CMx")
errors_cmy = evaluate_model(model_cmy, X_test, y_cmy_test, cmy_pred, "CMy")
errors_cmz = evaluate_model(model_cmz, X_test, y_cmz_test, cmz_pred, "CMz")

# Step 10: Mostra i risultati con i grafici
show_plots = input("Do you want to display result plots (yes/no)? [default=yes]: ") or "yes"
if show_plots.lower() not in ["yes", "no"]:
    print("Invalid input for displaying plots. Defaulting to 'yes'.")
    show_plots = True
else:
    show_plots = show_plots.lower() == "yes"

if show_plots:
    plot_predictions(y_cl_test, cl_pred, "CL")
    plot_predictions(y_cd_test, cd_pred, "CD")
    plot_predictions(y_cs_test, cs_pred, "CS")
    plot_predictions(y_cmx_test, cmx_pred, "CMx")
    plot_predictions(y_cmy_test, cmy_pred, "CMy")
    plot_predictions(y_cmz_test, cmz_pred, "CMz")


# Step 11: Combina i modelli di CL e CD
model = combine_models(model_cl, model_cd, model_cs, model_cmx, model_cmy, model_cmz)

model_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/z_surrogate_models/"
base_model_name = "surrogate_model_6coeff"
model_extension = ".pkl"

save_model(model, model_directory, base_model_name, model_extension)


# # Ottieni i parametri dall'utente
# user_inputs = get_user_inputs()
# apply_sampling = user_inputs["use_sampling"]
# add_noise = user_inputs["add_noise"]
# show_plots = user_inputs["show_plots"]

# # Adatta il modello per `cl`

# model_type_cl = user_inputs["model_type_cl"]
# params_cl = user_inputs["params_cl"]
# model_cl = train_surrogate_model(X_train, y_cl_train, model_type_cl, params_cl)


# # Adatta il modello per `cd`
# model_type_cd = user_inputs["model_type_cd"]
# params_cd = user_inputs["params_cd"]
# model_cd = train_surrogate_model(X_train, y_cd_train, model_type_cd, params_cd)

# # Fai previsioni e valuta il modello
# cl_pred = predict_model(model_cl, X_test)
# cd_pred = predict_model(model_cd, X_test)
# errors = evaluate_model(model_cl, model_cd, X_test, y_cl_test, y_cd_test, cl_pred, cd_pred)

# # Mostra i risultati con i grafici
# if show_plots:
#     plot_predictions(y_cl_test, y_cd_test, cl_pred, cd_pred)

# # Combina i modelli di CL e CD
# model = combine_models(model_cl, model_cd)

# # Salva il modello surrogato
# model_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/"
# model_name = "surrogate_model.pkl"
# save_model(model, os.path.join(model_directory, model_name))

# print(f"Model saved to {os.path.join(model_directory, model_name)}")


# Applica il campionamento Latin Hypercube se richiesto (DA CORREGGERE)
# if apply_sampling:
#     num_samples = int(input("Insert number of samples [default=100]: ") or "100")
#     X_lhs = latin_hypercube_sampling(X, num_samples)
#     y_cl_lhs, y_cd_lhs = match_outputs(X_lhs, X, y_cl, y_cd)
# else:
#     X_lhs, y_cl_lhs, y_cd_lhs = X, y_cl, y_cd


# VECCHIA INPUT
# def get_user_inputs():
#     """Ottiene i parametri dall'utente con valori di default e controlla la validità."""
#     try:
#         theta_input = (
#             input("Insert theta value (e.g., 0.1, 0.01, 0.001, 0.0001) [default=0.01]: ") or "0.01"
#         )
#         theta_value = float(theta_input)
#     except ValueError:
#         print("Theta value non valido, impostato a 0.01.")
#         theta_value = 0.01

#     corr_value = (
#         input(
#             "Insert correlation type (squar_exp, abs_exp, matern32, matern52) [default=matern32]: "
#         )
#         or "matern32"
#     )
#     if corr_value not in ["squar_exp", "abs_exp", "matern32", "matern52"]:
#         print("Correlation type non valido, impostato a 'matern32'.")
#         corr_value = "matern32"

#     poly_value = (
#         input("Insert polynomial type (constant, linear, quadratic) [default=quadratic]: ")
#         or "quadratic"
#     )
#     if poly_value not in ["constant", "linear", "quadratic"]:
#         print("Polynomial type non valido, impostato a 'quadratic'.")
#         poly_value = "quadratic"

#     use_sampling = (
#         input("Do you want to apply Latin Hypercube Sampling (yes/no)? [default=no]: ") or "no"
#     )

#     model_type = (
#         input("Choose model type (Kriging, Kriging_PLS) [default=Kriging_PLS]: ") or "Kriging_PLS"
#     )
#     if model_type not in ["Kriging", "Kriging_PLS"]:
#         print("Model type non valido, impostato a 'Kriging_PLS'.")
#         model_type = "Kriging_PLS"

#     return [theta_value], corr_value, poly_value, use_sampling.lower() == "yes", model_type
