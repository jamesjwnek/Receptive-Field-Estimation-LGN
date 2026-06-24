import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from skimage.measure import block_reduce

def create_dataset(folder_name_end, movie_file_name_middle, resp_file_name_middle, resp_data_key, no_lags, end_height_width):

	resp_data = scipy.io.loadmat(f"C:/neurophysiology_movies/H6214.010_2_Ch44/H6214.010_2_Ch44_{resp_file_name_middle}SetResp.mat")
	resp = resp_data[resp_data_key]
	(trials, reps) = resp.shape
	#stack response data vertically
	resp = resp.reshape(-1, 1, order="F")

	dataset = np.array([])
	for i in range(trials//375):
		movie_data = scipy.io.loadmat(f"C:/neurophysiology_movies/nguyen_clips_{folder_name_end}/McGill_clips_hc_{movie_file_name_middle}0000_{(i+1):02d}.mat")
		movie = movie_data["mvMovie"]

		reduced = block_reduce(movie, block_size=(movie.shape[0]//end_height_width, movie.shape[1]//end_height_width, 1), func=lambda block, axis: np.mean(block, axis=axis))
		(h, w, t) = reduced.shape

		zero_padding = np.zeros((h,w,no_lags-1))
		movie = np.concatenate((zero_padding, reduced), axis=2)

		trial_template = np.array([])
		for j in range(375):
			bin_data = movie[:,:,j:j+no_lags]
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
	dataset = np.hstack((dataset, resp))

	return dataset

train_dataset = create_dataset("train", "", "est", "est_resp", 8, 30)
np.save("C:/neurophysiology_movies/H6214_train_dataset.npy", train_dataset)
	
val_dataset = create_dataset("validation", "reg_", "reg", "reg_resp", 8, 30)
np.save("C:/neurophysiology_movies/H6214_val_dataset.npy", val_dataset)

test_dataset = create_dataset("test", "pred_", "pred", "pred_resp", 8, 30)
np.save("C:/neurophysiology_movies/H6214_test_dataset.npy", test_dataset)