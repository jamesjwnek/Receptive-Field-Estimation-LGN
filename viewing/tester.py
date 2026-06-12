import numpy as np
import pandas as pd
import scipy.io
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

#movies = np.load("20x20flattenedmoviesvar.npy")

#frame = movies[0,0:400]
#frame.shape = (20,20)

#frame = np.stack((frame,frame,frame), axis=2)

#print(frame.shape)

movie_name = "CM_gabor_synthCarr_synthEnv_0000_10"
data = scipy.io.loadmat(f"C:\\neurophysiology_movies\\CM_gabor_synthCarr_synthEnv\\{movie_name}.mat")
movie = data["mvMovie"]

print(movie[:,:,0].shape)

import pygame

pygame.init()

screen=pygame.display.set_mode((1500,500))
clock = pygame.time.Clock()

down = movie - np.mean(movie)
print(down.shape)
def rms(patch, axis):
	return np.sqrt(np.mean(np.square(patch)))


down = block_reduce(down, block_size=(24,24,1), func=lambda block, axis: np.sqrt(np.mean(np.square(block), axis=axis)))

cmin, cmax = np.min(down), np.max(down)

factor = 256 / (cmax - cmin)

down = factor * (down - cmin)

down = np.stack((down[:,:,0], down[:,:,0], down[:,:,0]), axis=2)

down = pygame.surfarray.make_surface(down)
down = pygame.transform.scale(down, (480,480))

moviergb = np.stack((movie[:,:,0], movie[:,:,0], movie[:,:,0]), axis=2)
normal = pygame.surfarray.make_surface(moviergb)

varss = np.load("20x20flattenedmoviesrms.npy")
print(varss.shape)
var = varss[0, 0:400]

var = var.reshape((20,20))
print(var.shape)
var = np.stack((var, var, var), axis=2)
print(var.shape)
var = pygame.surfarray.make_surface(var)
var = pygame.transform.scale(var, (480,480))


running = True
while running:
	for event in pygame.event.get():
		if event.type==pygame.QUIT:
			running = False


	screen.blit(down, (0,0))
	screen.blit(normal, (480,0))
	screen.blit(var, (1000,0))

	pygame.display.flip()
	clock.tick(60)

"""
df = pd.read_csv("exported_neuron.csv")
df.drop(list(range(191, len(df))), inplace=True)

labels = np.loadtxt("labels.csv", delimiter=",")
labels = labels[:10,:]
print(labels.shape)

frame = np.zeros((480,480))
no_frames = 0

for i in range(10):
	movie_name = df.at[i, "movienames"]
	data = scipy.io.loadmat(f"C:\\neurophysiology_movies\\CM_gabor_synthCarr_synthEnv\\{movie_name}.mat")
	movie = data["mvMovie"]
	movie = movie**2

	no_frames += 1

	frame_labels = labels[i,:]

	frame += movie @ frame_labels

import matplotlib.pyplot as plt
plt.imshow(frame/no_frames)
plt.show()

"""