import numpy as np

rng = np.random.default_rng()
data = np.zeros((2000, 4001))

for i in range(2000):

	#create an array of noise, uniform distribution between -1 and 1, size (h=20, w=20, t=10)

	template = rng.uniform(low=-1.0, high=1.0, size=(20, 20, 10))
	#response is sum of small patch in frame 5

	response = np.sum(template[4:8, 14:18, 5]) - 0 * np.sum(template[4:8, 14:18, 7])
	template = template.reshape(-1)
	template = np.append(template, response)

	data[i] = template

noise_col = rng.normal(loc=0.0, scale=0.0, size=(1, 2000))

data[:, -1] = data[:, -1] + noise_col

np.savetxt("C:/neurophysiology_movies/simulation_data_lagged_rf.csv", data, delimiter=",")
