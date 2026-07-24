import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from skimage.measure import block_reduce
import h5py
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

frames_per_movie = 375
no_lags = 8
batch_size = 64
crop_coords = (0, 0, 480, 480)
downsample_factor = 16
movies_no = 20

def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

def create_cnn_dataset(crop_coords, downsample_factor, movies_no, file_type_name, resp_data_key, folder_name_end, movie_file_name_middle, batch_size, no_lags):
    trials, reps, resp = get_response(file_type_name, resp_data_key)
    n_features = no_lags * ((crop_coords[2] - crop_coords[0]) // downsample_factor) ** 2
    ds = tf.data.Dataset.from_generator(
        lambda: cnn_generator(crop_coords, downsample_factor, movies_no, resp, trials, reps, folder_name_end, movie_file_name_middle, no_lags),
        output_signature=(
            tf.TensorSpec(shape=(no_lags, (crop_coords[2] - crop_coords[0]) // downsample_factor, (crop_coords[2] - crop_coords[0]) // downsample_factor, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds

def cnn_generator(crop_coords, downsample_factor, movies_no, resp, trials, reps, folder_name_end, movie_file_name_middle, no_lags):
    processed_size = (crop_coords[2] - crop_coords[0]) // downsample_factor
    for rep in range(reps):
        for movie in range(movies_no):
            with h5py.File(f"C:/neurophysiology_data/movies/H6214.010/nguyen_clips_{folder_name_end}/McGill_clips_hc_{movie_file_name_middle}0000_{(movie+1):02d}.mat", 'r') as f:
                movie_data = f["mvMovie"]
                for frame in range(frames_per_movie):
                    current_response = resp[rep*trials + movie*frames_per_movie + frame, 0]
                    padding_layers = max(no_lags-frame-1, 0)
                    if padding_layers > 0:
                        padding = np.zeros((padding_layers, processed_size, processed_size))
                        yield np.concatenate((padding, block_reduce(movie_data[:frame+1,crop_coords[0]:crop_coords[2],crop_coords[1]:crop_coords[3]], block_size=(1, downsample_factor, downsample_factor), func=lambda block, axis: np.mean(block, axis=axis))), axis=0)[..., np.newaxis], np.array([current_response], dtype=np.float32)
                    else:
                        yield block_reduce(movie_data[frame-no_lags+1:frame+1,crop_coords[0]:crop_coords[2],crop_coords[1]:crop_coords[3]], block_size=(1, downsample_factor, downsample_factor), func=lambda block, axis: np.mean(block, axis=axis))[..., np.newaxis], np.array([current_response], dtype=np.float32)





def create_dataset(crop_coords, downsample_factor, file_type_name, resp_data_key, folder_name_end, movie_file_name_middle, batch_size, no_lags):
    trials, reps, resp = get_response(file_type_name, resp_data_key)
    dataset_creator(resp, trials, reps, folder_name_end, movie_file_name_middle, crop_coords, downsample_factor, no_lags)

    with open(f"C:/neurophysiology_data/datasets/H6214.010/{folder_name_end}_records.txt", "a") as file:
        now = str(datetime.now())
        file.write(f"Time: {now}, cropping: {crop_coords}, downsampling: {downsample_factor}\n")

    path = f"C:/neurophysiology_data/datasets/H6214.010/current_{folder_name_end}_dataset.npy"
    n_features = no_lags * ((crop_coords[2] - crop_coords[0]) // downsample_factor) ** 2

    ds = tf.data.Dataset.from_generator(
        lambda: generator(path),
        output_signature=(
            tf.TensorSpec(shape=(n_features,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("Dataset created:")
    print(folder_name_end)
    print(crop_coords, flush=True)

    return ds

def generator(path):
    data = np.load(path, mmap_mode="r")
    try:
        yield from generator_reader(data)
    finally:
        del data

def generator_reader(data):
    for row in data:
        yield row[:-1].astype(np.float32), row[-1].astype(np.float32)

def create_reg_dataset():
    reg_trials, reg_reps, reg_resp = get_response("reg", "reg_resp")
    pred_trials, pred_reps, pred_resp = get_response("pred", "pred_resp")

def get_response(file_type_name, resp_data_key):
    with h5py.File(f'C:/neurophysiology_data/responses/H6214.010/H6214.010_2_Ch44_{file_type_name}SetResp.mat', 'r') as file:
        resp = file[resp_data_key][:]
    
    (reps, trials) = resp.shape
    resp = resp.reshape(-1, 1, order="F")
 
    return trials, reps, resp

def dataset_creator(resp_data, trials, reps, folder_name_end, movie_file_name_middle, crop_coords, downsample_factor, no_lags):

    dataset = np.array([])
    for i in range(trials//frames_per_movie):
        with h5py.File(f"C:/neurophysiology_data/movies/H6214.010/nguyen_clips_{folder_name_end}/McGill_clips_hc_{movie_file_name_middle}0000_{(i+1):02d}.mat", 'r') as f:
            movie = f["mvMovie"]

            reduced = block_reduce(movie[:,crop_coords[0]:crop_coords[2],crop_coords[1]:crop_coords[3]], block_size=(1, downsample_factor, downsample_factor), func=lambda block, axis: np.mean(block, axis=axis))
        (t, h, w) = reduced.shape
        print(reduced.shape)

        zero_padding = np.zeros((no_lags-1,h,w))
        movie = np.concatenate((zero_padding, reduced), axis=0)

        trial_template = np.array([])
        for j in range(375):
            bin_data = movie[j:j+no_lags,:,:]
            bin_data = bin_data.reshape(-1)

            if j == 0:
                trial_template = bin_data
            else:
                trial_template = np.vstack((trial_template, bin_data))

        if i == 0:
            dataset = trial_template
        else:
            dataset = np.vstack((dataset, trial_template))

    dataset = np.vstack([dataset]*reps)
    dataset = np.hstack((dataset, resp_data))

    np.save(f"C:/neurophysiology_data/datasets/H6214.010/current_{folder_name_end}_dataset.npy", dataset)

if __name__ == "__main__":

    create_cnn_dataset(crop_coords, downsample_factor, movies_no, "est", "est_resp", "train", "")

    #train_ds = create_dataset((0,0,120,120), 2, "est", "est_resp", "train", "", batch_size, no_lags)
    #val_ds = create_dataset((0,0,120,120), 2, "reg", "reg_resp", "validation", "reg_", batch_size, no_lags)
    #test_ds = create_dataset((0,0,120,120), 2, "pred", "pred_resp", "test", "pred_", batch_size, no_lags)