import pandas as pd

farfield_factors = [5.0, 10.0]
mesh_farfields = [1.0, 3.0, 5.0, 7.0]
fuselage_factors = [3.0, 5.0]
wing_factors = [5.0, 10.0, 15.0]
n_power_factors = [1.5, 2.0, 2.5]
n_power_fields = [0.9, 1.0]

parameter_combinations = [
    (farfield, mesh, fuselage, wing, n_power, n_field)
    for farfield in farfield_factors
    for mesh in mesh_farfields
    for fuselage in fuselage_factors
    for wing in wing_factors
    for n_power in n_power_factors
    for n_field in n_power_fields
]

columns = [
    "Farfield Factor",
    "Mesh Farfield",
    "Fuselage Factor",
    "Wing Factor",
    "N Power Factor",
    "N Power Field",
]
df_combinations = pd.DataFrame(parameter_combinations, columns=columns)

# Stampa del dataset
print(df_combinations)

# Salvataggio su un file CSV
csv_path = (
    "/wrk/Gronda/labAR/EULER/mesh_dependency/prova_cpacs_handling/parameter_combinations.csv"
)
df_combinations.to_csv(csv_path, index=False)
print(f"Dataset salvato come: {csv_path}")
