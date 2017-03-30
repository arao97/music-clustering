# -*- coding: utf-8 -*-

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import sys
import pandas as pd

main_dir = sys.argv[1]

song_data = pd.read_csv(main_dir + 'song_pca.csv', encoding = 'latin-1')

song_data.drop('track_id', axis = 1, inplace = True)

print(song_data.values)

db = DBSCAN(eps = 10, min_samples = 1500, metric = 'euclidean').fit(song_data)
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(labels)
print(n_clusters)

kmeans = KMeans(n_clusters = 10).fit(song_data)

print(kmeans.cluster_centers_)