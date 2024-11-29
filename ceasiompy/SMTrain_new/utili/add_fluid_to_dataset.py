import os
import pandas as pd
import numpy as np
import re


def standard_atmosphere(altitude):
    """
    Calculates standard atmospheric values: temperature, pressure, and density.
    Altitude is in meters.
    """
    T0 = 288.15  # Sea level temperature (K)
    P0 = 101325  # Sea level pressure (Pa)
    rho0 = 1.225  # Sea level density (kg/m^3)
    g = 9.80665  # Gravitational acceleration (m/s^2)
    L = 0.0065  # Temperature lapse rate (K/m)
    R = 287.058  # Specific gas constant for air (J/(kg*K))

    if altitude < 11000:  # Troposphere
        T = T0 - L * altitude
        P = P0 * (T / T0) ** (g / (R * L))
    else:  # Stratosphere (simplified)
        T = 216.65
        P = P0 * 0.22336 * np.exp(-g * (altitude - 11000) / (R * T))

    rho = P / (R * T)  # Density
    return T, P, rho


def calculate_reynolds(mach, rho, T, reference_length=1.0):
    """
    Calculates the Reynolds number.
    """
    gamma = 1.4  # Specific heat ratio for air
    R = 287.058  # Specific gas constant for air (J/(kg*K))
    mu_ref = 1.716e-5  # Reference dynamic viscosity (kg/(m*s))
    T_ref = 273.15  # Reference temperature (K)

    # Dynamic viscosity using Sutherland's law
    mu = mu_ref * (T / T_ref) ** 1.5 * (T_ref + 110) / (T + 110)

    # Velocity calculation
    a = np.sqrt(gamma * R * T)  # Speed of sound
    velocity = mach * a

    # Reynolds number
    reynolds = (rho * velocity * reference_length) / mu
    return reynolds


def process_csv(file_path, output_path=None):
    """
    Reads a CSV file, calculates atmospheric values and Reynolds number,
    and saves the updated file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Read the CSV
    df = pd.read_csv(file_path)

    # Check required columns
    required_columns = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is missing from the CSV file.")

    # Calculate atmospheric values and Reynolds number
    T_list, P_list, rho_list, Re_list = [], [], [], []
    for _, row in df.iterrows():
        altitude = row["altitude"]
        mach = row["machNumber"]

        T, P, rho = standard_atmosphere(altitude)
        Re = calculate_reynolds(mach, rho, T)

        T_list.append(T)
        P_list.append(P)
        rho_list.append(rho)
        Re_list.append(Re)

    # Add new columns
    df["temperature"] = T_list
    df["pressure"] = P_list
    df["density"] = rho_list
    df["reynolds_number"] = Re_list

    # Save updated file
    base, ext = os.path.splitext(file_path)
    output_path = output_path or f"{base}_with_fluid{ext}"
    df.to_csv(output_path, index=False)
    print(f"Updated file saved as: {output_path}")


if __name__ == "__main__":
    # Default paths
    default_input_file = "/path/to/default/input.csv"
    default_output_file = None  # Will default to input file name with '_with_fluid'

    input_file = input("Enter the path to the input CSV file: ").strip() or default_input_file

    try:
        process_csv(input_file, default_output_file)
    except Exception as e:
        print(f"Error: {e}")
