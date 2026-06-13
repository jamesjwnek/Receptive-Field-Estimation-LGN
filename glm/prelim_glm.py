import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping

no_lags = 8

df = pd.read_csv("C:/neurophysiology_movies/full_dataset.csv")


train_df = df.sample(frac=0.8, random_state=42)
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
test_df = df.drop(train_df.index)
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

#early stopping criteria
early_stopping = EarlyStopping(
    monitor='val_loss',       # Metric to monitor (e.g., validation loss)
    patience=5,               # Number of epochs to wait for improvement before stopping
    restore_best_weights=True # Rewind model weights to the best epoch's state
)

model = models.Sequential([
	layers.Input(shape=(400*no_lags,)), #input layer has 400 * no_lags entries
	layers.Dense(1, kernel_regularizer=regularizers.L1(0.001)) #output layer for regression is dense with one node, l1 regularization
	])

#adam optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.001)

#need to create own VAF formula in old tensorflow
def r2_score(y_true, y_pred):
    ss_res = keras.ops.sum(keras.ops.square(y_true - y_pred))
    ss_tot = keras.ops.sum(keras.ops.square(y_true - keras.ops.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

#compile using r^2 (VAF) as the metric
model.compile(optimizer=optimizer, loss="mse", metrics=[r2_score])

model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

layer_weights = model.layers[0].get_weights()

weights = layer_weights[0] # Kernel weights matrix
biases = layer_weights[1]  # Bias vector

for i in range(8):
	frame = weights[400*i:400*(i+1)]
	frame.shape = (20, 20)
	plt.imshow(frame)
	plt.figure()

plt.show()