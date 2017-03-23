# -*- coding: utf-8 -*-

import os
import glob
import sys
import csv
import numpy as np
import hdf5_getters as GETTERS

def iterate_files(basedir, csvfile, writer, fieldnames, ext='.h5'):
    count = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        count+=len(files)
        for f in files:
            h5feat = feat_from_file(f)
            feats_with_names = dict(zip(fieldnames, h5feat))
            writer.writerow(feats_with_names)
            print('Transformed file %s' % (str(f)))
    
    print('Transformed %d records' % (count))
            

def feat_names():
    
    res =  ['track_id','title','artist_name','year','loudness','tempo','time_signature','key','mode','duration']
    
    for k in range(1,13):
        res.append('avg_timbre'+str(k))
    for k in range(1,13):
        res.append('var_timbre'+str(k))
    for k in range(1,13):
        res.append('avg_pitches'+str(k))
    for k in range(1,13):
        res.append('var_pitches'+str(k))
        
    return res

def feat_from_file(path):
    
    feats = []
    h5 = GETTERS.open_h5_file_read(path)
    
    feats.append( GETTERS.get_track_id(h5) )
    feats.append( GETTERS.get_title(h5) )
    feats.append( GETTERS.get_artist_name(h5) )
    feats.append( GETTERS.get_year(h5) )
    feats.append( GETTERS.get_loudness(h5) )
    feats.append( GETTERS.get_tempo(h5) )
    feats.append( GETTERS.get_time_signature(h5) )
    feats.append( GETTERS.get_key(h5) )
    feats.append( GETTERS.get_mode(h5) )
    feats.append( GETTERS.get_duration(h5) )
    
    #timbre
    timbre = GETTERS.get_segments_timbre(h5)
    avg_timbre = np.average(timbre, axis=0)
    for k in avg_timbre:
        feats.append(k)
    var_timbre = np.var(timbre, axis=0)
    for k in var_timbre:
        feats.append(k)

    #pitches
    pitches = GETTERS.get_segments_pitches(h5)
    avg_pitches = np.average(pitches, axis=0)
    for k in avg_pitches:
        feats.append(k)
    var_pitches = np.var(pitches, axis=0)
    for k in var_pitches:
        feats.append(k)
    
    h5.close()
    
    return feats

if __name__ == '__main__':
    
    if(len(sys.argv) != 2):
        print('Invalid argument length')
        sys.exit(0)
    
    main_dir = sys.argv[1]
    
    
    with open(main_dir+'/song_data.csv', 'w') as csvfile:
        fieldnames = feat_names()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        main_dir = main_dir.replace('/', '\\')
        iterate_files(main_dir + '\\data', csvfile, writer, fieldnames)
        
        
    sys.exit(0)