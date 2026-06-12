import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

flat_movies_list = np.load("20x20flattenedmoviesrms.npy")
labels = np.loadtxt("labels.csv", delimiter=",")