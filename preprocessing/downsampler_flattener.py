import pandas as pd
import numpy as np
import scipy.io
from skimage.measure import block_reduce
import pygame

pygame.init()
screen=pygame.display.set_mode((1000,800))

#read the neuron's data and use only the good trials (drop 191 to the end)
df = pd.read_csv("exported_neuron.csv")
df.drop(list(range(191, len(df))), inplace=True)

#blank array with row of length 150 000 = 20 pixels * 20 pixels * 375 frames to get started
arr = np.zeros((150000,))


clock = pygame.time.Clock()

count = 0
#iterate over each movie name
for name in df["movienames"]:
	count += 1
	print(count)
	data = scipy.io.loadmat(f"C:\\neurophysiology_movies\\CM_gabor_synthCarr_synthEnv\\{name}.mat")
	movie = data["mvMovie"]
	track_mean = np.mean(movie)
	movie = movie - np.mean(movie)

	#block reduce to turn movie from 480 pixels * 480 pixels * 375 frames to 20 * 20 * 375, then reshape into a row
	reduced = block_reduce(movie, block_size=(24, 24, 1), func=lambda block, axis: np.sqrt(np.mean(np.square(block), axis=axis)))
	frame = np.array([0])

	H,W,T = reduced.shape
	for t in range(T):
		slicee = reduced[:,:,t]
		s_min, s_max = np.min(slicee), np.max(slicee)

		factor = 256 / (s_max - s_min)
		reduced[:,:,t] = factor * (slicee - s_min)

		surf = pygame.surfarray.make_surface(np.stack([reduced[:,:,t]]* 3, axis=2))
		scale_surf = pygame.transform.scale(surf, (480, 480))

		frame = np.concatenate((frame, reduced[:,:,t].reshape(-1)), axis=0)

		normal = movie[:,:,t] + track_mean
		norm_surf = pygame.surfarray.make_surface(np.stack([normal]*3, axis=2))
		screen.blit(norm_surf, (480, 0))

		screen.blit(scale_surf, (0,0))
		pygame.display.flip()
		clock.tick(120)

		for event in pygame.event.get():
			if event.type==pygame.QUIT:
				break

	fslice = frame[1:]
	
	#stack vertically on the empty array
	arr = np.vstack((arr, fslice))

	

#remove row of zeros
arr = arr[1:,:]

#save to csv or npy (npy much smaller file)

#np.savetxt("20x20flattenedmovies.csv", arr, delimiter=",")
np.save('20x20flattenedmoviesrms.npy', arr)