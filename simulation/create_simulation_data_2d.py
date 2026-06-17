#code to create simulation data, stored as a csv file
#columns 1-(32x32) are for pixel darkness, column (32x32 + 1) is for the response
#response is sum of pixels in rows 20-24 and columns 10-14

#will create 1200 entries total, not split into train - validation - test yet

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

#create 2000 row (each entry) by 32*32 column (number of pixels) random uniformly distributed dataset
template = rng.uniform(low=-1.0, high=1.0, size=(2000, 32*32))


###############square patch rf###########

#get the column indices to disregard in the dataset (do not match row and column needed)
rf_column_indices = np.arange(1, 32*32+1, 1) #all column indices in the dataset
rf_column_indices = rf_column_indices.astype(np.float64)
rf_column_indices.shape = (32, 32)
rf_column_indices[20:24, 10:14] = np.nan #once reshaped make nan all pixels in rows 20-24, columns 10-14
rf_column_indices.reshape(-1)
rf_column_indices = rf_column_indices[~np.isnan(rf_column_indices)] #remove nans
rf_column_indices = rf_column_indices.astype(int) - 1 #change to integer then -1 for zero-based indexing

#delete unwanted columns in dataset to remain only with columns that are used for the response
just_rf_columns = np.delete(template, rf_column_indices, axis=1)

#response here is just the sum of these columns
label_col = np.sum(just_rf_columns, axis=1, keepdims=True)

#noise column
noise_col = rng.normal(loc=0.0, scale=1.0, size=label_col.shape)

########################


################sine wave rf############
"""
sin_row = np.sin(np.arange(0, 32*32, 1))
weights = np.tile(sin_row, [2000, 1])
label_col = np.sum(template * weights, axis=1, keepdims=True)
noise_col = np.zeros(label_col.shape)
"""
#####################################

#add label column to dataset then save as csv
final_array = np.concatenate((template, label_col+noise_col), axis=1)

