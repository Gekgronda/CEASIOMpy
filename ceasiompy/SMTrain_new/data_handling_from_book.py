import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/home/cfse/Stage_Gronda/datasets/takeoff_500points.csv")
# data = pd.read_csv("dataset_prova_libro.csv")

# print(data.shape)
# print(data.columns)
# print(data.head(5))


# SOME INFO ABOUT DATASET
# print(data.describe().apply(lambda s: s.apply("{0:.5f}".format)))


# CHECK FOR MISSING VALUES
# missing_values = data.isnull()
# missung_counts = missing_values.sum()
# has_missing_values = missing_values.any().any()

# print(missung_counts)
# print(has_missing_values)


# HISTOGRAMS, LOOK THE SHAPE
fig, axes = plt.subplots(1, 4, figsize=(14, 3))
data.hist(ax=axes)
plt.tight_layout()


# LOGARITMIC TRANSF
# col_name = "Total CD"
# # Applicazione della trasformazione logaritmica
# data[f"{col_name}_log"] = np.log(data[col_name])

# # Plot per confrontare la distribuzione prima e dopo la trasformazione
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Istogramma della colonna originale
# axes[0].hist(data[col_name], bins=30, color="blue", alpha=0.7)
# axes[0].set_title(f"Original {col_name} Distribution")

# # Istogramma della colonna trasformata
# axes[1].hist(data[f"{col_name}_log"], bins=30, color="green", alpha=0.7)
# axes[1].set_title(f"Log-transformed {col_name} Distribution")

# plt.tight_layout()


# CREATE A BOXPLOT, useful to remove some data
# data_subset = data.iloc[:, 0]
# fig, ax = plt.subplots()
# sns.boxplot(data=data_subset, ax=ax)
# sns.stripplot(data=data_subset, color="black", size=4, jitter=True, ax=ax)
# plt.ylabel("")


# CORRELATION MATRIX
# correlation_matrix = data.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, linewidths=0.5)
# plt.title("")


# SCATTER PLOTS
# fig, ax = plt.subplots(1, 6, figsize=(24, 6))

# for i, col in enumerate(data.columns[0:]):
#     sns.regplot(x=data[col], y=data["Total CD"], ax=ax[i])

# fig.suptitle("")
# fig.tight_layout()
# fig.subplots_adjust(top=0.95)


plt.show()
plt.clf()
plt.close()
