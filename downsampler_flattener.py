import pandas as pd
import numpy as np
import scipy.io
from skimage.measure import block_reduce

#read the neuron's data and use only the good trials (drop 191 to the end)
df = pd.read_csv("exported_neuron.csv")
df.drop(list(range(191, len(df))), inplace=True)

#blank array with row of length 150 000 = 20 pixels * 20 pixels * 375 frames to get started
arr = np.zeros((150000,))

#iterate over each movie name
for name in df["movienames"]:
	data = scipy.io.loadmat(f"C:\\neurophysiology_movies\\CM_gabor_synthCarr_synthEnv\\{name}.mat")
	movie = data["mvMovie"]

	#block reduce to turn movie from 480 pixels * 480 pixels * 375 frames to 20 * 20 * 375, then reshape into a row
	reduced = block_reduce(movie, block_size=(24, 24, 1), func=np.mean)
	reduced = reduced.reshape(-1)
	
	#stack vertically on the empty array
	arr = np.vstack((arr, reduced))

#remove row of zeros
arr = arr[1:,:]

#save to csv or npy (npy much smaller file)

#np.savetxt("20x20flattenedmovies.csv", arr, delimiter=",")
np.save('20x20flattenedmovies.npy', arr)