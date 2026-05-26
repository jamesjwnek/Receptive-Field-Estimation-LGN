import numpy as np
import torch
"""
#movies = np.loadtxt("20x20flattenedmovies.csv", delimiter=",")
labels = np.loadtxt("labels.csv", delimiter=",")

#print(movies.shape)
print(labels.shape)

def relu(x):
	return max(0, x)

p0 = np.full((3200,), 0.1)

timelags = 8

def residuals(p):

	for i in range(191):
		for x in range(timelags-1, 375):
			true_val = labels[i, x]
			for y in range(x-timelags+1, x+1):
				print(y)
		

residuals(p0)

x=374
for y in range(x-timelags, x):
	print(y)

"""

if torch.cuda.is_available():
	device = "cuda"
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"

torch.manual_seed(42)

w = torch.randn((3200,1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)

just_flat_movies = np.load("20x20flattenedmovies.npy")

data = np.zeros((191*368, 3200))

curr_row = 0

timelags = 8
for x in range(timelags-1, 375):
	data[curr_row:curr_row+191,:] = just_flat_movies[:,(x-timelags+1)*400:(x+1)*400]
	curr_row += 191

learning_rate = 0.1
n_epochs = 10

labels = np.loadtxt("labels.csv", delimiter=",")

labels = labels[:, timelags-1:]

y_train = labels.T.reshape(-1)
y_train = torch.FloatTensor(y_train)

data = torch.FloatTensor(data)

print((data @ w).shape)
"""
for epoch in range(n_epochs):
	y_pred = data @ w + b
	loss = ((y_pred - y_train)**2).mean()
	loss.backward()
	with torch.no_grad():
		b -= learning_rate * b.grad()
		w -= learning_rate * w.grad()
		b.grad.zero_()
		w.grad.zero_()

	print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item()}")
"""

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(data, y_train)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)

import torch.nn as nn

model = nn.Sequential(nn.linear(3200, 1), nn.ReLU())
model = model.to(device)

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

def train(model, optimizer, criterion, train_loader, n_epochs):
	model.train()
	for epoch in range(n_epochs):
		total_loss = 0
		for X_batch, y_batch in train_loader:
			X_batch, y_batch = X_batch.to(device), y_batch.to(device)
			y_pred = model(X_batch)
			loss = criterion(y_pred, y_batch)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		mean_loss = total_loss/len(train_loader)

		print(f"Epoch: {epoch + 1}, Loss: {mean_loss:.4f}")