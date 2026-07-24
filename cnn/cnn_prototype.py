import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping
from skimage.measure import block_reduce
from moviepreppercnn import create_cnn_dataset, get_response
from sklearn.metrics import r2_score
import gc
batch_size = 64
unprocessed_width = 480
crop_width = 480
crop_coords = (0, 0, 480, 480)
downsample_factor = 16
no_lags = 8
movies_no = 20
no_epochs = 200
cropped_width = 30
cropped_width = crop_width // downsample_factor

def bang_out_movie(downsampled_movie, cropped_width):
    template = np.array([])
    for i in range(no_lags-1):
        if i == 0:
            template = np.concatenate((np.zeros((no_lags - i - 1, cropped_width,cropped_width)), downsampled_movie[:-no_lags+1+i,:]))
        elif i == 1:
            next_time_frame = np.concatenate((np.zeros((no_lags - i - 1, cropped_width,cropped_width)), downsampled_movie[:-no_lags+1+i,:]))
            template = np.stack((template, next_time_frame), axis=-1)
        else:
            next_time_frame = np.concatenate((np.zeros((no_lags - i - 1, cropped_width,cropped_width)), downsampled_movie[:-no_lags+1+i,:]))
            template = np.concatenate((template, next_time_frame[:,:,:,np.newaxis]), axis=-1)

    return np.concatenate((template, downsampled_movie[:,:,:,np.newaxis]), axis=-1)


def bang_out_dataset(movies_no, no_lags, crop_coords, downsample_factor, file_type_name, resp_data_key, folder_name_end, movie_file_name_middle, batch_size):
    crop_width = crop_coords[2] - crop_coords[0]
    cropped_width = crop_width // downsample_factor

    trials, reps, resp = get_response(file_type_name, resp_data_key)

    movie_template = np.array([])
    for movie in range(movies_no):
        with h5py.File(f"C:/neurophysiology_data/movies/H6214.010/nguyen_clips_{folder_name_end}/McGill_clips_hc_{movie_file_name_middle}0000_{(movie+1):02d}.mat", 'r') as f:
            movie_data = f["mvMovie"]
            downsampled_movie = block_reduce(movie_data[:,crop_coords[0]:crop_coords[2],crop_coords[1]:crop_coords[3]], block_size=(1, downsample_factor, downsample_factor), func=lambda block, axis: np.mean(block, axis=axis))
            processed_movie = bang_out_movie(downsampled_movie, cropped_width)
        if movie == 0:
            movie_template = processed_movie
        else:
            movie_template = np.concatenate((movie_template, processed_movie), axis=0)

    final_movie = np.concatenate([movie_template]*reps, axis=0)
    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((final_movie, resp))

    del final_movie
    gc.collect()

    return dataset.batch(batch_size = batch_size).prefetch(tf.data.AUTOTUNE)


train_dataset = bang_out_dataset(20, no_lags, crop_coords, downsample_factor, "est", "est_resp", "train", "", batch_size)
val_dataset = bang_out_dataset(5, no_lags, crop_coords, downsample_factor, "reg", "reg_resp", "validation", "reg_", batch_size)
test_dataset = bang_out_dataset(5, no_lags, crop_coords, downsample_factor, "pred", "pred_resp", "test", "pred_", batch_size)

"""def r2_score(y_true, y_pred):
    ss_res = tf.math.reduce_sum(tf.math.square(y_true - y_pred))
    ss_tot = tf.math.reduce_sum(tf.math.square(y_true - tf.math.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + 1e-7)"""

class ClipWeightConstraint(keras.constraints.Constraint):
    def __init__(self, min_value =0.0, max_value = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self,w):
        return K.clip(w, self.min_value, self.max_value)

class Gaussian(layers.Layer):
    def __init__(self, units=1, input_shape=(1,20,20,1)):
        super().__init__()

        self.x_0 = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(10),
            trainable=True,
            constraint = ClipWeightConstraint(min_value=0, max_value = 20)
            )

        self.y_0 = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(10),
            trainable=True,
            constraint = ClipWeightConstraint(min_value=0, max_value = 20)
            )

        self.A = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(0.01),
            trainable=True,
            constraint = ClipWeightConstraint(min_value=0, max_value = 20)
            )

        self.s_x = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(5),
            trainable=True,
            constraint = ClipWeightConstraint(min_value=0, max_value = 20)
            )

        self.s_y = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(5),
            trainable=True,
            constraint = ClipWeightConstraint(min_value=0, max_value = 20)
            )

        self.theta = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(np.pi),
            trainable=True
            )

        i, j = tf.meshgrid(tf.range(20), tf.range(20))
        self.i = tf.cast(i, tf.float32)
        self.j = tf.cast(j, tf.float32)

    def call(self, inputs):
        inputs = K.reshape(inputs, (-1, 20,20))
        

        a = tf.math.square(tf.math.cos(self.theta)) / (2 * tf.math.square(self.s_x)) + tf.math.square(tf.math.sin(self.theta)) / (2 * tf.math.square(self.s_y))
        b = tf.math.sin(2*self.theta) / (4 * tf.math.square(self.s_x)) - tf.math.sin(2*self.theta) / (4 * tf.math.square(self.s_y))
        c = tf.math.square(tf.math.sin(self.theta)) / (2 * tf.math.square(self.s_x)) + tf.math.square(tf.math.cos(self.theta)) / (2 * tf.math.square(self.s_y))
        
        return tf.math.reduce_sum(tf.math.multiply(self.A * tf.math.exp(-1 * (a * tf.math.square(self.i - self.x_0) + 2*b*(self.i-self.x_0)*(self.j-self.y_0) + c*tf.math.square(self.j-self.y_0))), inputs))

gaussian_layer = Gaussian()

model = keras.Sequential([
    layers.Input(shape=(cropped_width, cropped_width, no_lags)),
    layers.Conv2D(1, (11, 11), strides=(1, 1), kernel_initializer="glorot_uniform"),
    layers.PReLU(),
    gaussian_layer,
    layers.ReLU()
    ])

for x, y in train_dataset.take(1):
    print(x.shape)
    print(y.shape)

optimizer = keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss="mse", metrics=[r2_score])
history = model.fit(train_dataset, epochs=no_epochs, validation_data=val_dataset)
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")



#vectorize the Gaussian layer with tf.reduce_sum
#build the Gaussian once
#use pure tf.math instead of keras backend
#try xla: in model.compile, set jit_compile = True
#increase batch size
#Use conv2d instead of 3d
#use matrix multiplication