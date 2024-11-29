import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from smt.surrogate_models import KRG
from smt.applications import EGO, MFK
from smt.sampling_methods import LHS
from sklearn.metrics import mean_squared_error
import numpy as np


def load_and_split_data(path, default_path):
    """
    Load database, split inputs and outputs, handle issues.

    Parameters:
        path (str): database path from user
        default_path (str): database default path.

    Returns:
        dict: Dictionary containing inputs and outputs.
    """

    def load_database(prompt, default_path):
        path = input(prompt).strip()
        if not path:
            print(f"No path given. Load default path: {default_path}")
            path = default_path
        try:
            df = pd.read_csv(path)
            print(f"Load database from: {path}")
            return df
        except Exception as e:
            print(f"Issues occurred during loading of {path}: {e}")
            print(f"Load default path: {default_path}")
            try:
                return pd.read_csv(default_path)
            except Exception as e_default:
                raise FileNotFoundError(
                    f"Issues occurred during loading of default path: {e_default}"
                )

    df = load_database(
        path,
        default_path,
    )

    def split_data(df):
        input_columns = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]
        output_columns = {
            "CL": "Total CL",
            "CD": "Total CD",
            "CSF": "Total CSF",
            "CMx": "Total CMx",
            "CMy": "Total CMy",
            "CMz": "Total CMz",
        }

        # Check input columns
        missing_inputs = [col for col in input_columns if col not in df.columns]
        if missing_inputs:
            raise KeyError(f"Missing input columns: {missing_inputs}")

        X = df[input_columns].values

        # Check and collect output columns
        y = {}
        for key, col in output_columns.items():
            if col in df.columns:
                y[key] = df[col].values
            else:
                print(f"Warning: Column '{col}' is missing and will be excluded from outputs.")

        return X, y

    X, y = split_data(df)

    return {"dataset": {"X": X, "y": y, "df": df}}


def plot_distributions(df, title):
    """
    Make hystogram for each DataFrame column.

    Parameters:
        df (pd.DataFrame): DataFrame containing data.
        title (str): Plot title.
    """
    df.hist(bins=30, figsize=(15, 10), color="skyblue", edgecolor="black")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def normalize_data(df, columns_to_normalize=None):
    """
    Normalize each dataframe column using MinMaxScaler (scaling each variable by their min and max values)

    Parameters:
        df (pd.DataFrame): DataFrame containing data to be normalized.
        columns_to_normalize (list or None): List of columns to be normalized. If None, all columns are normalized.

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    scaler = MinMaxScaler()
    if columns_to_normalize is None:
        columns_to_normalize = df.columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def mf_kriging():
    
    

