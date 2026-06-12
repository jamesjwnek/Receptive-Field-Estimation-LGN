import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

just_flat_movies = np.load("20x20flattenedmoviesrms.npy")
labels = np.loadtxt("labels.csv", delimiter=",")
labels = labels[:, 7:]
stacked_labels = labels.flatten(order="F")

data = np.zeros((191*368, 3200))

curr_row = 0
timelags = 8
for x in range(timelags-1, 375):
	data[curr_row:curr_row+191,:] = just_flat_movies[:,(x-timelags+1)*400:(x+1)*400]
	curr_row += 191

data = np.column_stack((data, stacked_labels))

df = pd.DataFrame(data)
#df = df.drop(df.index[200:])

mask = df[df.columns[-1]] > 0
df_yes = df[mask]
df_no = df[~mask]

pics = []

def new_sample(frame):
	row = frame.sample()
	print(row.iloc[0, 3200])
	lis = []

	for i in range(8):
		pic = row.iloc[0, 400*i:400*i + 400].to_numpy()
		pic.shape = (20,20)
		picsurf = pygame.surfarray.make_surface(np.stack([pic]*3, axis=2))
		picsurf = pygame.transform.scale(picsurf, (400, 400))
		lis.append(picsurf)

	pics.append(lis)

pygame.init()
screen = pygame.display.set_mode((1200, 800))

slider = Slider(screen, 100, 650, 800, 40, min=1, max=8, step=1)

clock = pygame.time.Clock()
FPS = 60

new_sample(df_yes)

running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_SPACE:
				new_sample(df_yes)

	screen.fill((255,255,255))

	pygame_widgets.update(pygame.event.get())

	screen.blit(pics[-1][slider.getValue()-1], (0,0))

	pygame.display.flip()
	clock.tick(FPS)

pygame.quit()
