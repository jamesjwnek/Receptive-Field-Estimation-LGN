import json
from pathlib import Path
directory = Path("C:/neurophysiology_data/glm/runs")
files = [f for f in directory.iterdir() if f.is_file()]
files.reverse()

with open(files[0], 'r', encoding='utf-8') as file:
    data = json.load(file)

import pygame
import pygame_menu
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import numpy as np

pygame.init()

screen = pygame.display.set_mode((2000, 800))
pygame.display.set_caption("Run Reader")
clock = pygame.time.Clock()

my_font = pygame.font.SysFont('Arial', 16)

menu = pygame_menu.Menu(
    title='Past Runs',
    width=400,
    height=800,
    theme=pygame_menu.themes.THEME_BLUE
)

display = False

slider = Slider(screen, 650, 650, 800, 40, min=0, max=10, step=0.05)
slider.disable()
slider.hide()

slider_box = TextBox(screen, 1500, 650, 50, 40, fontSize=16)
slider_box.disable()
slider_box.hide()

def show_run(path):
	with open(path, 'r', encoding='utf-8') as file:
		data = json.load(file)
	global display
	display = True

	label_stuff = data.copy()
	weights = np.array(label_stuff.pop("weights", None))
	weights = weights.reshape((20,20,8))
	label_stuff.pop("training_stats", None)
	label_stuff.pop("biases", None)

	label_stuff["test_loss"] = round(label_stuff["test_loss"], 2)
	label_stuff["test_accuracy"] = round(label_stuff["test_accuracy"], 2)

	global labels_list
	labels_list = []

	slider.enable()
	slider.show()
	slider_box.show()

	for key, value in label_stuff.items():
		text_surface = my_font.render(str(key) + ": " + str(value), True, (0,0,0))
		labels_list.append(text_surface)

	global weights_list
	weights_list = []
	weights = weights * slider.getValue() / weights.max()
	for i in range(8):
		pic = weights[:,:,i]
		pic = 256 / (1 + np.exp(-1 * pic))
		picsurf = pygame.surfarray.make_surface(np.stack([pic.T]*3, axis=2))
		picsurf = pygame.transform.scale(picsurf, (200, 200))
		weights_list.append(picsurf)


for file_path in files:
    menu.add.button(file_path.stem, show_run, file_path, font_size=16)

menu.add.button('Quit', pygame_menu.events.EXIT)
menu.set_relative_position(0, 0)

stats_surface = pygame.Surface((200, 800))
stats_surface.fill((200, 200, 200))

running = True
while running:
	events = pygame.event.get()
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	screen.fill((255, 255, 255))

	if display == True:
		screen.blit(stats_surface, (400, 0))
		for i, x in enumerate(labels_list):
			screen.blit(x, (420, 20 + i*30))

		row_1_weights = weights_list[0:5]
		row_2_weights = weights_list[5:10]

		for i, x in enumerate(row_1_weights):
			screen.blit(x, (650 + 250*i, 50))

		for i, x in enumerate(row_2_weights):
			screen.blit(x, (650 + 250*i, 300))

		slider_box.setText(slider.getValue())

	

	if menu.is_enabled():
		menu.update(events)
		menu.draw(screen)

	pygame_widgets.update(events)

	pygame.display.flip()
	clock.tick(60)
pygame.quit()
