import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from skimage.measure import block_reduce

movie_data = scipy.io.loadmat("C:/neurophysiology_movies/nguyen_clips_test/McGill_clips_hc_0000_02.mat")
movie = movie_data["mvMovie"]
reduced = block_reduce(movie, block_size=(16, 16, 1), func=lambda block, axis: np.mean(block, axis=axis))
print(reduced.shape)

down_data = scipy.io.loadmat("C:/neurophysiology_movies/DataSets_McGillClips_hc_downsampled480to30.mat")
down_movie = down_data["Training_stimuli"]
print(down_movie.shape)

import pygame

frame = 0

pygame.init()
screen=pygame.display.set_mode((1000,500))
clock = pygame.time.Clock()

def move_forward():
	global frame1_surf
	global frame1_down_surf_scaled
	global frame
	global frame1_surf_scaled



	frame1 = np.stack([reduced[:,:,frame]]*3, axis=2)
	frame1 = frame1 #* 256 / np.max(frame1)
	frame1_surf = pygame.surfarray.make_surface(frame1)
	frame1_surf_scaled = pygame.transform.scale(frame1_surf, (480, 480))
	print(frame1)

	frame1_down = np.stack([down_movie[:,:,375+frame]]*3, axis=2)
	frame1_down_surf = pygame.surfarray.make_surface(frame1_down)
	frame1_down_surf_scaled = pygame.transform.scale(frame1_down_surf, (480, 480))

	frame += 1

move_forward()
print(frame)

running = True
while running:
	for event in pygame.event.get():
		if event.type==pygame.QUIT:
			running = False

		if event.type== pygame.KEYDOWN:
			if event.key == pygame.K_SPACE:
				move_forward()
				print(frame)

	screen.blit(frame1_surf_scaled, (0,0))
	screen.blit(frame1_down_surf_scaled, (500, 0))

	pygame.display.flip()
	clock.tick(60)