import numpy as np
from smt.applications import EGO
from smt.applications.mixed_integer import MixedIntegerContext
from smt.surrogate_models import KRG, MixIntKernelType
from smt.utils.design_space import CategoricalVariable, DesignSpace


def Kriging(X_train, y_train, theta, corr, poly):
    """Train Kriging model."""
    model = KRG(theta0=[theta], corr=corr, poly=poly, print_global=False)
    model.set_training_values(X_train, y_train)
    model.train()
    return model


def map_categorical_values(X):
    """Map continuous values to their corresponding categorical labels."""
    # Map for theta
    theta_values = [0.0001, 0.001, 0.01, 0.1, 1]
    theta = theta_values[int(X[0, 0]) % len(theta_values)]

    # Map for corr
    corr_values = ["pow_exp", "abs_exp", "squar_exp"]
    corr = corr_values[int(X[0, 1]) % len(corr_values)]

    # Map for poly
    poly_values = ["constant", "linear", "quadratic"]
    poly = poly_values[int(X[0, 2]) % len(poly_values)]

    return theta, corr, poly


def evaluate_model(X, X_train, y_train, X_val, y_val):
    """Funzione obiettivo per ottimizzare il modello KRG."""
    # Converti i valori categorici
    theta, corr, poly = map_categorical_values(X)

    # Creare e addestrare il modello con i parametri selezionati
    model = Kriging(X_train, y_train, theta, corr, poly)

    # Prevedere su X_val e calcolare RMSE
    y_pred = model.predict_values(X_val)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
    return rmse.reshape((-1, 1))


def optimize_hyperparameters_ego2(model_type, X_train, y_train, X_val, y_val):
    # Definire lo spazio di progettazione per i parametri
    design_space = DesignSpace(
        [
            CategoricalVariable([0, 1, 2, 3, 4]),  # Indici per theta
            CategoricalVariable([0, 1, 2]),  # Indici per corr_value
            CategoricalVariable([0, 1, 2]),  # Indici per poly_value
        ]
    )

    # Configurare il contesto misto
    mixint = MixedIntegerContext(design_space)

    # Metodo di campionamento
    sampling = mixint.build_sampling_method(random_state=42)
    xdoe = sampling(3)  # Campioni DoE iniziali
    ydoe = np.array(
        [evaluate_model(x.reshape(1, -1), X_train, y_train, X_val, y_val) for x in xdoe]
    )

    # Configurare il modello surrogato
    surrogate = KRG(
        design_space=design_space,
        categorical_kernel=MixIntKernelType.GOWER,
        hyper_opt="Cobyla",
        print_global=False,
    )

    # Configurare l'ottimizzatore EGO
    ego = EGO(
        n_iter=15,  # Numero di iterazioni
        criterion="EI",  # Funzione obiettivo
        xdoe=xdoe,
        ydoe=ydoe,
        surrogate=surrogate,
        n_parallel=2,
        random_state=42,
    )

    # Ottimizzazione
    x_opt, y_opt, _, _, _ = ego.optimize(
        fun=lambda X: evaluate_model(X, X_train, y_train, X_val, y_val)
    )

    # Mappare i valori ottimizzati sui valori categorici effettivi
    theta_opt, corr_opt, poly_opt = map_categorical_values(x_opt)

    print(
        f"Migliori parametri trovati: theta={theta_opt}, corr={corr_opt}, poly={poly_opt} con RMSE={y_opt.item()}"
    )

    return theta_opt, corr_opt, poly_opt
