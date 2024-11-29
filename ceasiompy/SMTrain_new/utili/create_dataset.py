import numpy as np
import os
from smt.sampling_methods import LHS
import csv
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# Funzione per fare sampling del DoE specificato
def lh_sampling(ranges, n_samples, random_state=None):
    sampling = LHS(xlimits=ranges, criterion="ese", random_state=random_state)
    samples = sampling(n_samples)
    x = sampling(n_samples)

    ### For visualization only
    xlimits = ranges
    num = n_samples
    intervals = []
    subspace_bool = []
    for i in range(len(xlimits)):
        intervals.append(np.linspace(xlimits[i][0], xlimits[i][1], num + 1))
        subspace_bool.append(
            [
                [intervals[i][j] < x[kk][i] < intervals[i][j + 1] for kk in range(len(x))]
                for j in range(len(intervals[i]) - 1)
            ]
        )

    PLOT1 = 0  # First dimension to be depicted
    PLOT2 = 2  # Second dimension to be depicted

    fig, ax = plt.subplots(1)
    ax.plot(x[:, PLOT1], x[:, PLOT2], ".")
    for i in range(len(intervals[0])):
        ax.plot(
            [intervals[PLOT1][i], intervals[PLOT1][i]],
            [intervals[PLOT2][0], intervals[PLOT2][-1]],
            linewidth=0.5,
            color="k",
        )
        ax.plot(
            [intervals[PLOT1][0], intervals[PLOT1][-1]],
            [intervals[PLOT2][i], intervals[PLOT2][i]],
            linewidth=0.5,
            color="k",
        )
    ax.set_xlabel("Dimension " + str(PLOT1 + 1))
    ax.set_ylabel("Dimension " + str(PLOT2 + 1))
    ax.legend(["Initial LHS"], bbox_to_anchor=(1.05, 0.6))

    plt.show()

    return samples


def save_to_csv(samples, column_names, filename):
    """
    Salva i dati estratti in un file CSV.

    Args:
        samples: Dati da salvare (array NumPy).
        column_names: Nomi delle colonne (lista).
        filename: Nome completo del file di output.
    """
    # Arrotonda i valori
    rounded_samples = []
    for row in samples:
        rounded_row = [
            round(row[0], 1),  # Altitudine con 1 cifra decimale
            round(row[1], 2),  # Numero di Mach con 2 cifre decimali
            int(round(row[2])),  # Angolo di attacco con 0 cifra decimali
            int(round(row[3])),  # Angolo di sideslip con 0 cifra decimali
        ]
        rounded_samples.append(rounded_row)

    # Scrive i dati in un file CSV
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()  # Scrive l'intestazione
        writer.writerows([dict(zip(column_names, row)) for row in rounded_samples])
    print(f"File saved to {filename}")


# Chiedi all'utente il numero di samples
n = int(input("Insert the number of samples: "))

# Definizione dei range degli input
ranges = np.array(
    [
        [0, 1000],  # Altezza: da 0 a 1000 m
        [0.1, 0.5],  # Mach: da 0.1 a 0.5
        [0, 15],  # AoA: da 0° a 15°
        [-2, 2],  # AoS: da -2° a 2°
    ]
)

# Nomi delle colonne per il CSV
column_names = ["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]

# Genera i dati
samples = lh_sampling(ranges, n)

# Visualizzazione dei campioni generati
print("Campioni generati:")
print(samples)

# Chiedi all'utente il nome del file CSV
output_filename = input("Enter the name of the CSV file (with .csv extension): ")

# Percorso della cartella in cui salvare il file
output_directory = "/wrk/Gronda/dataframe_new/"

# Crea il percorso completo del file
full_path = os.path.join(output_directory, output_filename)

# Salva i dati in un file CSV
save_to_csv(samples, column_names, full_path)
