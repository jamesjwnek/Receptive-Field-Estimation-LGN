# for testing different downsampling methods

import numpy as np
import pandas as pd
import scipy.io
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import pygame

#loading the first movie in the dataset
movie_name = "CM_gabor_synthCarr_synthEnv_0000_10"
data = scipy.io.loadmat(f"C:\\neurophysiology_movies\\CM_gabor_synthCarr_synthEnv\\{movie_name}.mat")
movie = data["mvMovie"]

pygame.init()
screen=pygame.display.set_mode((1500,500))
clock = pygame.time.Clock()

#shift so mean is zero, then downsample with method of choice (here root mean square)
down = movie - np.mean(movie)
down = block_reduce(down, block_size=(24,24,1), func=lambda block, axis: np.sqrt(np.mean(np.square(block), axis=axis)))

#normalize so minimum value is black, maximum is white, linear scaling in between
cmin, cmax = np.min(down), np.max(down)
factor = 256 / (cmax - cmin)
down = factor * (down - cmin)

#stack it to comply with rgb, then make surface and scale up for viewing
down = np.stack((down[:,:,0], down[:,:,0], down[:,:,0]), axis=2)
down = pygame.surfarray.make_surface(down)
down = pygame.transform.scale(down, (480,480))

#compare it to normal movie
moviergb = np.stack((movie[:,:,0], movie[:,:,0], movie[:,:,0]), axis=2)
normal = pygame.surfarray.make_surface(moviergb)

#load downsampled pictures to commpare with downsampling on this script, just first frame of first movie
varss = np.load("C:/neurophysiology_movies/20x20flattenedmoviesrms.npy")
var = varss[0, 0:400]
var = var.reshape((20,20))

#stack it to comply with rgb, then make surface and scale up for viewing
var = np.stack((var, var, var), axis=2)
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