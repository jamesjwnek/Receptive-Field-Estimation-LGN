###GOAL: output 2d array, rows are the 191 trials, containing binned spike counts

import pandas as pd
import numpy as np

#import the file, this contains the following columns (each row is a trial):
#organized start times: time at which photocell turns on for that trial
#organized end times: time at which photocell turns off for that trial
#datarecord times: time at which csm file indicates the start of the trial (just before the organized end time)
#movienames: name of movie played
#emptycol_1 ---- emptycol_198: times of each spike in that trial, relative to the start of the experiment.
#                              the spikes span a 5 second time period and their number varies (so some entries are empty)
df = pd.read_csv("exported_neuron.csv")

#get just the column names that contain the times of each spike
column_names = df.columns.to_list()
just_ts_column_names = column_names[4:]

#make a new column which is a list of the spike times during each trial
df["dirty_ts"] = df[just_ts_column_names].to_numpy().tolist()

#make copy just containing movie names, start times, list of spikes
movie_ts_df = df[["movienames", "organizedstarttimes", "dirty_ts"]].copy()

#get rid of nan entries in each list, put it in new column
movie_ts_df["clean_ts"] = movie_ts_df.apply(lambda row: [i for i in row["dirty_ts"] if not pd.isna(i)], axis=1)

#change everything in that column to an array instead of a list to allow broadcasting
movie_ts_df["clean_ts"] = movie_ts_df["clean_ts"].map(np.array)

#subtract trial start time so spike times are relative to start of trial
movie_ts_df["clean_ts"] = movie_ts_df.apply(lambda row: row["clean_ts"] - row["organizedstarttimes"], axis=1)

#bin the spike times into 1/75 second bins over 5 seconds
binedges = np.arange(5*75+1)
binedges = binedges / 75
movie_ts_df["binned"] = movie_ts_df.apply(lambda row: np.histogram(row["clean_ts"], binedges)[0], axis=1)

#final value for start of range should be 191
movie_ts_df.drop(list(range(191, len(movie_ts_df))), inplace=True)

binned_2d_arr = np.stack(movie_ts_df["binned"].to_numpy())

np.savetxt("labels.csv", binned_2d_arr, delimiter=",")