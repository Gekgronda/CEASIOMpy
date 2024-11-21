import numpy as np
import pandas as pd
import os


# Funzione per generare i dati
def data_gen(n):
    """Genera un DataFrame con n predizioni per altitudine, numero di Mach, AoA e AoS."""
    altitude = np.random.randint(0, 1000, n)  # Altitudine random tra 0 e 15000 metri
    mach = np.round(np.random.uniform(0.1, 0.3, n), 1)  # Mach number tra 0.1 e 0.6 con 1 decimale
    aoa = np.random.randint(0, 15, n)  # Angolo di attacco tra -4 e 15 gradi
    aos = np.random.randint(-2, 2, n)  # Angolo di derapata tra -15 e 15 gradi

    df = pd.DataFrame(
        {"altitude": altitude, "machNumber": mach, "angleOfAttack": aoa, "angleOfSideslip": aos}
    )
    return df


# Funzione per salvare il DataFrame in un file CSV
def save_to_csv(df, filename):
    """Salva il dataset in un file CSV."""
    df.to_csv(filename, index=False)
    print(f"File saved to {filename}")


# Chiedi all'utente il numero di punti da generare
n = int(input("Insert the number of points: "))

# Genera i dati
df = data_gen(n)

# Chiedi all'utente il nome del file CSV
output_filename = input("Enter the name of the CSV file (without .csv extension): ") + ".csv"

# Percorso della cartella in cui salvare il file
output_directory = "/home/cfse/Stage_Gronda/datasets"

# Crea il percorso completo del file
full_path = os.path.join(output_directory, output_filename)

# Salva il DataFrame in un file CSV
save_to_csv(df, full_path)
