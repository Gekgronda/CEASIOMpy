import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from ceasiompy.SMTrain_new.gg_sm2 import split_data

# Load the dataset
name = input("Insert database name (with .csv extension): ") or "takeoff0.csv"
file_path = f"/home/cfse/Stage_Gronda/datasets/{name}"
df = pd.read_csv(file_path)

# Define inputs and outputs
X = df[["altitude", "machNumber", "angleOfAttack", "angleOfSideslip"]].values
y_cl = df["Total CL"].values
y_cd = df["Total CD"].values

# Split data using the custom split_data function
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

# Set seed for reproducibility
tf.random.set_seed(1234)

# Define a model for predicting Total CL
model_cl = Sequential(
    [
        Dense(120, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1), name="L1"),
        Dense(40, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1), name="L2"),
        Dense(1, activation="linear", name="L3"),
    ],
    name="model_cl",
)

model_cl.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=0.001))

# Train the model for Total CL
history_cl = model_cl.fit(X_train, y_cl_train, epochs=1000, validation_data=(X_val, y_cl_val))

model_cl.summary()

# Predict values for the first 5 samples of X_test
y_cl_pred = model_cl.predict(X_test)

# Print actual vs predicted for the first 5 samples of CL
print("First 5 values of Total CL - Actual vs Predicted:")
for i in range(5):
    print(f"Sample {i+1} - Actual: {y_cl_test[i]:.4f}, Predicted: {y_cl_pred[i][0]:.4f}")

# Plot actual vs predicted for the first 5 samples of Total CL
plt.figure(figsize=(10, 5))
plt.plot(range(len(X_test)), y_cl_test, marker="o", label="Actual CL")
plt.plot(range(len(X_test)), y_cl_pred, marker="x", label="Predicted CL")
plt.title("First 5 Predictions of Total CL")
plt.xlabel("Sample Index")
plt.ylabel("CL Value")
plt.legend()
plt.show()


print(X_test[0:4], y_cl_test[0:4])

# Define a model for predicting Total CD
# model_cd = Sequential(
#     [
#         Dense(120, activation="relu", name="L1"),
#         Dense(60, activation="relu", name="L2"),
#         Dense(1, activation="linear", name="L3"),
#     ],
#     name="model_cd",
# )

# model_cd.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.001))

# # Train the model for Total CD
# history_cd = model_cd.fit(X_train, y_cd_train, epochs=100, validation_data=(X_val, y_cd_val))

# # Predict values for the first 5 samples of X_train
# y_cd_pred = model_cd.predict(X_train[:5])

# # Print actual vs predicted for the first 5 samples of CD
# print("\nFirst 5 values of Total CD - Actual vs Predicted:")
# for i in range(5):
#     print(f"Sample {i+1} - Actual: {y_cd_train[i]:.4f}, Predicted: {y_cd_pred[i][0]:.4f}")

# # Plot actual vs predicted for the first 5 samples of Total CD
# plt.figure(figsize=(10, 5))
# plt.plot(range(5), y_cd_train[:5], marker="o", label="Actual CD")
# plt.plot(range(5), y_cd_pred, marker="x", label="Predicted CD")
# plt.title("First 5 Predictions of Total CD")
# plt.xlabel("Sample Index")
# plt.ylabel("CD Value")
# plt.legend()
# plt.show()
