# -*- coding: utf-8 -*-

import numpy as np
import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def create_feature_list(feature_name):
    feature_list = [feature_name + str(i) for i in range(1,13)]
    return feature_list

def create_numpy_arrays(feature_name):
    return scale(song_data[create_feature_list(feature_name)].values)

def create_scaled_array(feature_name):
    return scale(song_data[feature_name].values)

def pca_transform(data_frame):
    pca.fit(data_frame)
    return pca.fit_transform(data_frame)

def convert_to_df():
    pca_transformed = pd.DataFrame(list(song_data['track_id']), columns = ['track_id'])
    
    pca_transformed['avg_timbre'] = pd.Series(list(avg_song_timbre), index = pca_transformed.index)
    pca_transformed.apply(float(pca_transformed['avg_timbre'][0]), 1)

main_dir = sys.argv[1]

song_data = pd.read_csv(main_dir + 'song_data.csv', encoding = 'latin-1')

song_loudness = create_scaled_array('loudness')
song_tempo = create_scaled_array('tempo')
avg_song_timbre = create_numpy_arrays('avg_timbre')
var_song_timbre = create_numpy_arrays('var_timbre')
avg_song_pitches = create_numpy_arrays('avg_pitches')
var_song_pitches = create_numpy_arrays('var_pitches')

pca = PCA(n_components = 1)

avg_song_timbre = pca_transform(avg_song_timbre)
var_song_timbre = pca_transform(var_song_timbre)
avg_song_pitches = pca_transform(avg_song_pitches)
var_song_pitches = pca_transform(var_song_pitches)

pca_array = [[list(song_data['track_id'])[i],
              float(list(song_loudness)[i]),
                  float(list(song_tempo)[i]),
                      list(song_data['time_signature'])[i],
                          list(song_data['key'])[i],
                              list(song_data['mode'])[i],
                                  float(list(avg_song_timbre)[i]),
                                       float(list(var_song_timbre)[i]),
                                            float(list(avg_song_pitches)[i]),
                                                 float(list(var_song_pitches)[i])] for i in range(10000)]

pca_transformed = pd.DataFrame(
        pca_array, columns = ['track_id',
                              'loudness',
                              'tempo',
                              'time_signature',
                              'key',
                              'mode',
                              'avg_timbre',
                              'var_timbre',
                              'avg_pitches',
                              'var_pitches'])

pca_transformed.to_csv(sys.argv[1] + 'song_pca.csv', index = False)