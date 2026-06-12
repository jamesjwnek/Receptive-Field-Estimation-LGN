import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

df = pd.read_csv("simulation_data_sum_rf.csv")

train_df = df.sample(frac=0.8, random_state=42)
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
test_df = df.drop(train_df.index)
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

print(X_train.head())

model = models.Sequential([
	layers.Input(shape=(1024,)),
	layers.Dense(1, kernel_regularizer=regularizers.L1(0.001))
	])

optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss="mse", metrics=[keras.metrics.R2Score()])

model.fit(X_train, y_train, epochs=200, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

layer_weights = model.layers[0].get_weights()

weights = layer_weights[0] # Kernel weights matrix
biases = layer_weights[1]  # Bias vector

weights.shape = (32, 32)
plt.imshow(weights)
plt.show()