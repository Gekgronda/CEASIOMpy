import pandas as pd
import numpy as np
from ceasiompy.SMTrain_new.gg_sm2 import (
    latin_hypercube_sampling,
    split_data,
    fit_model,
    predict_model,
    evaluate_model,
    plot_predictions,
    combine_models,
    save_model,
)


def get_user_inputs():
    """Ottiene i parametri dall'utente con valori di default"""
    theta_input = (
        input("Insert theta value (e.g., 0.1, 0.01, 0.001, 0.0001) [default=0.01]: ") or "0.01"
    )
    corr_value = (
        input(
            "Insert correlation type (squar_exp, abs_exp, matern32, matern52) [default=matern32]: "
        )
        or "matern32"
    )
    poly_value = (
        input("Insert polynomial type (constant, linear, quadratic) [default=quadratic]: ")
        or "quadratic"
    )
    use_sampling = (
        input("Do you want to apply Latin Hypercube Sampling (yes/no)? [default=no]: ") or "no"
    )

    return [float(theta_input)], corr_value, poly_value, use_sampling.lower() == "yes"


# Carica il database
name = input("Insert database name (with .csv extention): ") or "dataset_500_points.csv"
file_path = f"/home/cfse/Stage_Gronda/datasets/{name}"
df = pd.read_csv(file_path)

# Definisci gli input e output
X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values

# Ottieni i parametri dall'utente
theta_values, corr_value, poly_value, apply_sampling = get_user_inputs()

# Applica il campionamento Latin Hypercube se richiesto (DA CORREGGERE)
# if apply_sampling:
#     num_samples = int(input("Insert number of samples [default=100]: ") or "100")
#     X_lhs = latin_hypercube_sampling(X, num_samples)
#     y_cl_lhs, y_cd_lhs = match_outputs(X_lhs, X, y_cl, y_cd)
# else:
#     X_lhs, y_cl_lhs, y_cd_lhs = X, y_cl, y_cd

(
    X_train,
    y_cl_train,
    X_val,
    y_cl_val,
    X_test,
    y_cl_test,
    X_temp,
    y_cd_train,
    X_val,
    y_cd_val,
    X_test,
    y_cd_test,
) = split_data(X, y_cl, y_cd)

# Adatta il modello
model_cl, model_cd = fit_model(
    X_train, y_cl_train, y_cd_train, theta_values, corr_value, poly_value
)

# Fai previsioni e valuta il modello
cl_pred, cd_pred = predict_model(model_cl, model_cd, X_test)
errors = evaluate_model(model_cl, model_cd, X_test, y_cl_test, y_cd_test, cl_pred, cd_pred)

# Mostra i risultati con i grafici
plot_predictions(y_cl_test, y_cd_test, cl_pred, cd_pred)

# Combina i modelli di CL e CD
model = combine_models(model_cl, model_cd)

# Salva il modello surrogato
model_directory = "/home/cfse/Stage_Gronda/CEASIOMpy/ceasiompy/SMTrain_new/"
model_name = "surrogate_model.pkl"
save_model(model, f"{model_directory}{model_name}")

print(f"Model saved to {model_directory}{model_name}")
