# -*- coding: utf-8 -*-

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering as Ward
from sklearn.neighbors import kneighbors_graph as KNN
import pandas as pd
import sys

birch_dict = {
        
        0: 'rock',
        1: 'metal',
        2: 'electronic',
        3: 'blues',
        4: 'classical',
        5: 'hip-hop',
        6: 'pop',
        7: 'rap/hip-hop',
        8: 'folk'
        
        }

ward_dict = {
        
        0: 'rock',
        1: 'country/folk',
        2: 'hip-hop',
        3: 'electronic',
        4: 'reggae/rock',
        5: 'ambient/classic',
        6: 'blues',
        7: 'metal',
        8: 'rap'
        
        }


main_dir = sys.argv[1]
artTitle = pd.read_csv(main_dir + 'song_data.csv', encoding = 'latin-1')
artist_list = artTitle['artist_name'].tolist()
title_list = artTitle['title'].tolist()
song_data = pd.read_csv(main_dir + 'song_pca.csv', encoding = 'latin-1')
tracks_list = song_data['track_id'].tolist()
song_data.drop('track_id', axis = 1, inplace = True)


brch = Birch(n_clusters = 9, compute_labels = True)
brch.fit(song_data)
print(brch.labels_)
brch_list = list(brch.labels_)

graph = KNN(song_data, n_neighbors=100, include_self=False)


ward = Ward(n_clusters =9,linkage='ward',connectivity = graph)
ward.fit(song_data)
print(ward.labels_)
ward_list = list(ward.labels_)

song_data['Ward_genre'] = ward_list
song_data['Birch_genre'] = brch_list
song_data['track_id'] = tracks_list
song_data['artist'] = artist_list
song_data['title'] = title_list
song_data = song_data[['track_id','artist','title','Birch_genre','Ward_genre','loudness','tempo','time_signature','key','mode','avg_timbre','var_timbre']]
song_data['Birch_genre'].replace(birch_dict, inplace = True)
song_data['Ward_genre'].replace(ward_dict, inplace = True)
song_data.to_csv(main_dir + 'clustered_data.csv',sep = ',',encoding = 'latin-1')