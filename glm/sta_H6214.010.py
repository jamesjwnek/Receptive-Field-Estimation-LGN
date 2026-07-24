import numpy as np
from movieprepper import get_response
import h5py
import matplotlib.pyplot as plt

folder_name_end = "train"
movie_file_name_middle = ""

trials, reps, response_v = get_response("est", "est_resp")
response = response_v.reshape(-1)

frames_per_movie = 375
lags = 3


template = []
final_template = np.zeros((480,480))

for j in range(reps):
    for i in range(trials//frames_per_movie):
        with h5py.File(f"C:/neurophysiology_data/movies/H6214.010/nguyen_clips_{folder_name_end}/McGill_clips_hc_{movie_file_name_middle}0000_{(i+1):02d}.mat", 'r') as f:
            movie_section = f["mvMovie"][:-lags,:,:]
            padding = np.zeros((lags,480,480))
            full_movie = np.concatenate((padding,movie_section), axis=0)
            associated_response = response[375*i+j*7500:375*i+375+j*7500]
            stacked_slices = np.sum(full_movie * associated_response[:,None,None], axis=0)
            final_template = final_template + stacked_slices


plt.imshow(final_template/np.sum(response))
plt.show()

#testing broadcasting
"""
A = np.array([
    [[1,1,1],
    [2,2,2],
    [3,3,3]],
    [[1,1,1],
    [2,2,2],
    [3,3,3]],
    [[1,1,1],
    [2,2,2],
    [3,3,3]],
    [[1,1,1],
    [2,2,2],
    [3,3,3]],
    ])

v = np.array([1,2,3,4])

B = A * v[:,None,None]

C = np.sum(B, axis=0)
"""

