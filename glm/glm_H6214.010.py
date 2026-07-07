import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping

from glm import GLMModel
from movieprepper import create_dataset
import gc

batch_size = 64
patience = 20
regularizer_type = "l1"
regularizer_value = 0.01
learning_rate = {
    "default": 1e-4,
    5e-5: (100, 200),
    1e-5: (0, 100)
}
no_epochs = 200
no_lags = 8

unprocessed_width = 480
stride = 60
crop_width = 120
downsample_factor = 4

def create_stride_starts_list(unprocessed_width, stride, crop_width):
    i = 0
    while unprocessed_width - crop_width >= i:
        yield i
        i += stride

stride_list = list(create_stride_starts_list(unprocessed_width, stride, crop_width))

for i in stride_list:
    for j in stride_list:
        print("Starting new model run. Coordinates: ", (i, j))
        crop_window = (i, j, i+crop_width, j+crop_width)

        train_ds = create_dataset(crop_window, downsample_factor, "est", "est_resp", "train", "", batch_size, no_lags)
        val_ds = create_dataset(crop_window, downsample_factor, "reg", "reg_resp", "validation", "reg_", batch_size, no_lags)
        test_ds = create_dataset(crop_window, downsample_factor, "pred", "pred_resp", "test", "pred_", batch_size, no_lags)

        run_model = GLMModel(no_lags, unprocessed_width, i, j, crop_width, downsample_factor)
        run_model.train_manual(train_ds, val_ds, no_epochs, patience, regularizer_type, regularizer_value, learning_rate)
        run_model.test_model(test_ds)
        run_model.export_run("trial_one")

        del train_ds
        del val_ds
        del test_ds

        gc.collect()
        K.clear_session()

