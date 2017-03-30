# -*- coding: utf-8 -*-

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import sys
import seaborn as sns
import pandas as pd

main_dir = sys.argv[1]

song_data = pd.read_csv(main_dir + 'song_pca.csv', encoding = 'latin-1')

song_data.drop('track_id', axis = 1, inplace = True)

'''sns.regplot('loudness','tempo',song_data)
sns.plt.show()'''

'''db = DBSCAN(eps = 2, min_samples = 1500, metric = 'euclidean').fit(song_data)
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(labels)
print(n_clusters)

kmeans = KMeans(n_clusters = 10).fit(song_data)

print(kmeans.cluster_centers_)
print(kmeans.labels_)'''

brch = Birch(n_clusters = 9, compute_labels = True)
brch.fit(song_data)
print(brch.labels_)
     

     