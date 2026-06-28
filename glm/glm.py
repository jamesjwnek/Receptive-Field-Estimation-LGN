import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping

import json
from datetime import datetime
from pathlib import Path
import os

def r2_score(y_true, y_pred):
	ss_res = K.sum(K.square(y_true - y_pred))
	ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return 1 - ss_res / (ss_tot + K.epsilon())

class GLMModel:
	def __init__(self, n_lags, frame_width):
		self.regularizer_value = 0.01
		self.n_lags = n_lags
		self.frame_width = frame_width
		self.n_features = self.n_lags * self.frame_width * self.frame_width
		self.learning_rate = 0.0001

		self.model_noreg = models.Sequential([
			layers.Input(shape=(self.n_features,)),
			layers.Dense(1, kernel_initializer="random_normal", bias_initializer="zeros")
			])

		self.model_l1 = models.Sequential([
			layers.Input(shape=(self.n_features,)),
			layers.Dense(1, kernel_initializer="random_normal", bias_initializer="zeros", kernel_regularizer=regularizers.L1(self.regularizer_value))
			])

		self.model_l2 = models.Sequential([
			layers.Input(shape=(self.n_features,)),
			layers.Dense(1, kernel_initializer="random_normal", bias_initializer="zeros", kernel_regularizer=regularizers.L2(self.regularizer_value))
			])

		self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
		self.loss_fn = keras.losses.MeanSquaredError()

	def update_reg_val(self, reg_type, reg_value):
		updating = True
		if reg_type == "l1":
			reg = keras.regularizers.l1(reg_value)
			model = self.model_l1
		if reg_type == "l2":
			reg = keras.regularizers.l2(reg_value)
			model = self.model_l2
		else:
			model = self.model_noreg
			updating = False

		if updating:
			for layer in model.layers:
				if hasattr(layer, "kernel"):
					model.add_loss(lambda layer=layer: reg(layer.kernel))

		return model

	def update_lr(self, value):
		self.learning_rate = value
		self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

	def compile_models(self):
		self.model_noreg.compile(optimizer=self.optimizer, loss="mse", metrics=[r2_score])
		self.model_l1.compile(optimizer=self.optimizer, loss="mse", metrics=[r2_score])
		self.model_l2.compile(optimizer=self.optimizer, loss="mse", metrics=[r2_score])

	def train_manual(self, train_dataset, val_dataset, test_dataset, no_epochs, patience, reg_type, reg_value, learning_rate):
		self.update_lr(learning_rate["default"])
		model = self.update_reg_val(reg_type, reg_value)
		self.compile_models()

		val_losses = []
		val_r2s = []

		plt.ion()
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
		line1, = ax1.plot(val_losses)
		line2, = ax2.plot(val_r2s)

		stop_count = 0
		lowest_loss = np.inf

		for epoch in range(no_epochs):
			print(f"\nStart of epoch {epoch+1}")

			for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
				with tf.GradientTape() as tape:
					logits = model(x_batch_train, training=True)
					loss_value = self.loss_fn(y_batch_train, logits)

				grads = tape.gradient(loss_value, model.trainable_weights)
				self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

				if step % 50 == 0:
					print(f"Training loss at step {step}: {float(loss_value):.4f}. ------------ R^2 value: {r2_score(logits, y_batch_train)}", flush=True)

			val_total = 0
			r2_total = 0
			count = 0

			for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
				val_logits = model(x_batch_val, training=False)
				val_loss = self.loss_fn(y_batch_val, val_logits)
				val_total += val_loss
				r2_total += r2_score(y_batch_val, val_logits)
				count += 1

			val_loss_avg = val_total / count
			r2_avg = r2_total / count

			print("\n")
			print(f"Validation loss for epoch {epoch+1}: {float(val_loss_avg):.4f}. --------------- R^2 value: {float(r2_avg):.4f}", flush=True)
			val_losses.append(float(val_loss_avg.numpy()))
			val_r2s.append(float(r2_avg.numpy()))

			updated_learning_rate = False
			for key in learning_rate:
				if key != "default":
					left_endpoint = learning_rate[key][0]
					right_endpoint = learning_rate[key][1]

					if left_endpoint <= val_loss_avg and val_loss_avg < right_endpoint:
						self.optimizer.learning_rate.assign(key)
						print(f"learning_rate: {key}")
						updated_learning_rate = True
						break

			if (val_total / count) < lowest_loss:
				lowest_loss = val_total / count
				stop_count = 0
			else:
				stop_count += 1

			if patience != None:
				if stop_count > patience:
					break

			line1.set_ydata(val_losses)
			line1.set_xdata(range(len(val_losses)))

			line2.set_ydata(val_r2s)
			line2.set_xdata(range(len(val_r2s)))

			ax1.relim()
			ax1.autoscale_view()
			ax1.set_yscale("log")

			ax2.relim()
			ax2.autoscale_view()

			plt.draw()
			plt.pause(0.05)

		training_stats = {
			"val_losses": val_losses,
			"val_r2s": val_r2s
		}

		test_loss, test_accuracy = model.evaluate(test_dataset)
		print(f"Test Loss: {test_loss:.4f}")
		print(f"Test Accuracy: {test_accuracy:.4f}")

		layer_weights = model.layers[0].get_weights()
		weights = layer_weights[0]
		biases = layer_weights[1]

		plt.ioff()
		plt.show()
		

		run_data = {
			"training_method": "manual",
			"patience": patience,
			"regularizer_type": reg_type,
			"regularizer_value": reg_value,
			"learning_rate": learning_rate,
			"no_epochs": no_epochs,
			"training_stats": training_stats,
			"test_loss": test_loss,
			"test_accuracy": test_accuracy,
			"weights": weights.tolist(),
			"biases": biases.tolist(),
			"no_lags": self.n_lags,
			"frame_width": self.frame_width

		}

		self.export_run(run_data)

	def train_auto(self, train_dataset, val_dataset, test_dataset, no_epochs, patience, reg_type, reg_value, learning_rate):
		model = self.update_reg_val(reg_type, reg_value)
		self.update_lr(learning_rate)
		self.compile_models()

		if patience == None:
			history = model.fit(train_dataset, epochs=no_epochs, validation_data=val_dataset)
		else:
			early_stopping = EarlyStopping(
				monitor='val_loss',       # Metric to monitor (e.g., validation loss)
				patience=patience,               # Number of epochs to wait for improvement before stopping
				restore_best_weights=True # Rewind model weights to the best epoch's state
			)

			history = model.fit(train_dataset, epochs=no_epochs, validation_data=val_dataset, callbacks=[early_stopping])

		test_loss, test_accuracy = model.evaluate(test_dataset)
		print(f"Test Loss: {test_loss:.4f}")
		print(f"Test Accuracy: {test_accuracy:.4f}")

		layer_weights = model.layers[0].get_weights()
		weights = layer_weights[0]
		biases = layer_weights[1]

		run_data = {
			"training_method": "auto",
			"patience": patience,
			"regularizer_type": reg_type,
			"regularizer_value": reg_value,
			"learning_rate": learning_rate,
			"no_epochs": no_epochs,
			"training_stats": history.history,
			"test_loss": test_loss,
			"test_accuracy": test_accuracy,
			"weights": weights.tolist(),
			"biases": biases.tolist(),
			"no_lags": self.n_lags,
			"frame_width": self.frame_width

		}

		self.export_run(run_data)

	def export_run(self, run_dict):
		now = str(datetime.now())
		now = now.replace(" ", "_")
		now = now.replace(":", "-")

		file_name = "run_" + now + ".json"
		with open("C:/neurophysiology_projects/glm/runs/" + file_name, "w", encoding="utf-8") as file:
			json.dump(run_dict, file, indent=4)

