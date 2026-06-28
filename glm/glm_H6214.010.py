import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping

from glm import GLMModel


train_df = np.load("C:/neurophysiology_data/datasets/H6214.010/H6214.010_train_dataset.npy", mmap_mode="r")
val_df = np.load("C:/neurophysiology_data/datasets/H6214.010/H6214.010_val_dataset.npy", mmap_mode="r")
test_df = np.load("C:/neurophysiology_data/datasets/H6214.010/H6214.010_test_dataset.npy", mmap_mode="r")

def generator_train():
    for row in train_df:
        yield row[:-1].astype(np.float32), row[-1].astype(np.float32)

def generator_val():
    for row in val_df:
        yield row[:-1].astype(np.float32), row[-1].astype(np.float32)

def generator_test():
    for row in test_df:
        yield row[:-1].astype(np.float32), row[-1].astype(np.float32)

batch_size = 64
train_ds = tf.data.Dataset.from_generator(
    generator_train, 
    output_signature=(
        tf.TensorSpec(shape=(train_df.shape[1]-1,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    generator_train, 
    output_signature=(
        tf.TensorSpec(shape=(val_df.shape[1]-1,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_generator(
    generator_train, 
    output_signature=(
        tf.TensorSpec(shape=(test_df.shape[1]-1,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

patience = 50
regularizer_type = "l1"
regularizer_value = 0.01
learning_rate = {
    "default": 1e-4,
    5e-5: (100, 200),
    1e-5: (0, 100)
}
no_epochs = 500
no_lags = 8
frame_width = 30

model = GLMModel(no_lags, frame_width)

model.train_manual(train_ds, val_ds, test_ds, no_epochs, patience, regularizer_type, regularizer_value, learning_rate)

