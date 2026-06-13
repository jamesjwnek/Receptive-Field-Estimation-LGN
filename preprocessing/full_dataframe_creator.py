import numpy as np
import pandas as pd

flat_movies = np.load("C:/neurophysiology_movies/20x20flattenedmoviesrms.npy")
labels = np.loadtxt("C:/neurophysiology_data/glm/labels.csv", delimiter=",")

no_lags = 8
template = np.zeros((1, 400*no_lags+1))

for t in range(no_lags, 376):
	chunk = flat_movies[:, 400*(t-no_lags):400*t]
	label_col = labels[:, t-1:t]
	chunk = np.concatenate((chunk, label_col), axis=1)

	template = np.vstack((template, chunk))

template = template[1:]

df = pd.DataFrame(template)
df.to_csv(r"C:\neurophysiology_movies\full_dataset.csv", index=False)