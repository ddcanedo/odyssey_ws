from jakteristics import las_utils, compute_features, FEATURE_NAMES
import os
from tkinter import filedialog
from sklearn import linear_model
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pickle
import math

print(FEATURE_NAMES)

f = filedialog.askdirectory(title = "Select the folder with the LAS files")


X = []

for las in os.listdir(f):
	xyz = las_utils.read_las_xyz(f + '/' + las)

	features = compute_features(xyz, search_radius=3)
	
	if np.isnan(features).any() == False:
		stats = {}
		for i in FEATURE_NAMES:
			stats[i] = []
		
		for feature in features:
			for i in range(len(FEATURE_NAMES)):
				stats[FEATURE_NAMES[i]].append(feature[i])

		tmp = []

		for i in FEATURE_NAMES:		
			mean = np.mean(stats[i])
			median = np.median(stats[i])
			var = np.var(stats[i])
			stdev = np.std(stats[i])
			cov = np.cov(stats[i])
			tmp += [median, stdev, var, cov]

		X.append(tmp)


clf = LocalOutlierFactor(n_neighbors=1, novelty=True).fit(X)
predictions = clf.predict(X)

print('Done')

pickle.dump(clf, open('pointCloud.sav', 'wb'))

