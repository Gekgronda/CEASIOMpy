import pandas as pd
import numpy as np
import os
from MFSM_Func import load_and_split_data, plot_distributions, normalize_data, lhs_sampling


# Default dataset
default_hf_path = "/wrk/Gronda/labAR/RANS/RANS_dataset_toff_TRAIN.csv"
default_mf_path = "/wrk/Gronda/labAR/EULER/takeoff0/updated_takeoff0_TRAIN.csv"
default_lf_path = "/wrk/Gronda/labAR/AVL/takeoff0/takeoff0_TRAIN.csv"

# Load database and with the func load_adn_split_data split data in inputs and outputs
# and change key name
hf_path = input("Insert high fidelity database path: ")
hf_df = load_and_split_data(hf_path, default_hf_path)
hf_df["high_fidelity"] = hf_df.pop("dataset")
mf_path = input("Insert medium fidelity database path: ")
mf_df = load_and_split_data(mf_path, default_mf_path)
mf_df["medium_fidelity"] = mf_df.pop("dataset")
lf_path = input("Insert low fidelity database path: ")
lf_df = load_and_split_data(lf_path, default_lf_path)
lf_df["low_fidelity"] = lf_df.pop("dataset")

# Merge in one dictionary
data = {**hf_df, **mf_df, **lf_df}

# Collect input and output of every dataset
X_hf = data["high_fidelity"]["X"]
y_hf = data["high_fidelity"]["y"]
X_mf = data["medium_fidelity"]["X"]
y_mf = data["medium_fidelity"]["y"]
X_lf = data["low_fidelity"]["X"]
y_lf = data["low_fidelity"]["y"]

# print(X_hf, y_hf)

# Plot hystograms
# High fidelity hystograms
print("Plotting high-fidelity data distributions...")
plot_distributions(data["high_fidelity"]["df"], "Variables distibuition - High Fidelity")
# Medium fidelity hystograms
print("Plotting medium-fidelity data distributions...")
plot_distributions(data["medium_fidelity"]["df"], "Variables distibuition - Medium Fidelity")
# Low fidelity hystograms
print("Plotting low-fidelity data distributions...")
plot_distributions(data["low_fidelity"]["df"], "Variables distibuition - Low Fidelity")

# Normalize dataset points
# HF normalization
hf_df_normalized = normalize_data(data["high_fidelity"]["df"])
print("Normalized HF data:")
print(hf_df_normalized.head())
# MF normalization
mf_df_normalized = normalize_data(data["medium_fidelity"]["df"])
print("Normalized MF data:")
print(mf_df_normalized.head())
# LF normalization
lf_df_normalized = normalize_data(data["low_fidelity"]["df"])
print("Normalized LF data:")
print(lf_df_normalized.head())

# PRIMA DI PROCEDERE CON IL TRAINING DEL MODELLO SURROGATO BISOGNA INTRODURRE
# UN CONFRONTO CON IL DOMINIO FISICO ED ESCLUDERE I PUNTI CHE STANNO FUORI
