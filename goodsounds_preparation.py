#Monta csv
#transforma csv em dicionario com labels codificadas

#falta criar datalist
#falta fazer preprocessamento de audio


import pandas as pd
import numpy as np
import os
import splitfolders
import sqlite3
from sklearn.preprocessing import LabelEncoder


path_all = 'good-sounds/sound_files'
path_output_all =  '/data_lists/goodsounds_all_test.scp'
labels_file = 'good-sounds/good-sounds-labels_test.csv'
sqlite_path = 'good-sounds/database.sqlite'
#splitfolders.ratio(input = path, output = 'good-sounds', seed = 42, ratio=(.6,.4), group_prefix = None, move = False)


def clean_sqlite(sqlite_path):
    con = sqlite3.connect(sqlite_path)
    df = pd.read_sql_query("SELECT * from Sounds", con) 
    cols_drop = ['note', 'octave', 'dynamics', 'recorded_at', 'location',
                'player', 'bow_velocity', 'bridge_position', 'string', 
                'csv_file', 'csv_id', 'attack', 'decay', 'sustain', 'release',
                'offset', 'reference', 'comments', 'semitone', 'pitch_reference']
    con.close()
    df_sounds = df.drop(columns = cols_drop)

    con = sqlite3.connect(sqlite_path)
    df_packs = pd.read_sql_query("SELECT * from Packs", con) 
    con.close()


    df_sounds = df_sounds[df_sounds['pack_id'].notna()]
    df_sounds = df_sounds.reset_index()
    packs = list()
    for i in range(len(df_sounds)): 
        for pack in range(len(df_packs)):
            if int(df_sounds['pack_id'][i]) == df_packs['id'][pack]:
                packs.append(df_packs['name'][pack])
                break
    df_sounds['pack'] = packs
    df_sounds = df_sounds.drop(columns= 'pack_id')


    con = sqlite3.connect(sqlite_path)
    df_takes = pd.read_sql_query("SELECT * from Takes", con) 
    df_takes = df_takes[df_takes['sound_id'].notna()]
    mics = list()
    for i in range(len(df_sounds)): 
        for take in range(len(df_takes)): #testar print e ver quantas iteracoes ta tendo em df_takes
            if int(df_sounds['id'][i]) == df_takes['sound_id'][take]:
                mics.append(df_takes['microphone'][take])
                break
    df_sounds['microphone'] = mics


    con = sqlite3.connect(sqlite_path)
    df_packs = pd.read_sql_query("SELECT * from Packs", con) 
    con.close()


    paths = list()
    for row in range(len(df_sounds)):
        path_row = df_sounds['pack'][row]+'/'+df_sounds['microphone'][row]+'/'+df_sounds['pack_filename'][row]
        paths.append(path_row)
    df_sounds['path'] = paths
    df_sounds.drop(columns=['pack', 'microphone', 'pack_filename', 'index'])
    return df_sounds



def wav_filter(file):
    if '.wav' in file: return True
    else: return False


def create_datalist(path_to_data):
    open(path_to_data, 'w').close()
    with open(path_to_data, 'w') as f:
        for root, dirs, filenames in os.walk(path_to_data):
            filenames = list(filter(wav_filter, filenames))
            for i in filenames:
                f.writelines(os.path.join(root,i)+'\n')
            #  pass

df_s = clean_sqlite(sqlite_path)
df_s.to_csv(labels_file)
df = pd.read_csv(labels_file)
encoder = LabelEncoder()
instrument_labels = list(df['instrument'])
labels = encoder.fit_transform(instrument_labels)
df['label'] = labels

gs_labelfile = dict()
for file_index in range(len(df['path'])):
    gs_labelfile[df['path'][file_index]] = df['label'][file_index]
    print(df['path'][file_index], df['label'][file_index])
#print(gs_labelfile)




#a = np.load('data_lists/TIMIT_labels.npy', allow_pickle = True)
#a = a.item()
#print(a)