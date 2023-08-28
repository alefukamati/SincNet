#Monta csv
#transforma csv em dicionario com labels codificadas

#falta fazer preprocessamento de audio
#dividir em test e train (splitfolders)

import pandas as pd
import numpy as np
import os
import sox
import splitfolders
import sqlite3
from sklearn.preprocessing import LabelEncoder

dict_file_out = '/data_lists/goodsounds_labels.npy'
path_sounds = 'good-sounds/sound_files'
norm_path_all = 'norm_good-sounds/sound_files'
path_output_datalist =  '/data_lists/goodsounds_all.scp'
labels_file = 'good-sounds/good-sounds-labels.csv'
sqlite_path = 'good-sounds/database.sqlite'
#splitfolders.ratio(input = path, output = 'good-sounds', seed = 42, ratio=(.6,.4), group_prefix = None, move = False)


def clean_sqlite(sqlite_path):
    con = sqlite3.connect(sqlite_path)
    df = pd.read_sql_query("SELECT * from Sounds", con) 
    print("Starting to clean sqlite file...")
    cols_drop = ['note', 'octave', 'dynamics', 'recorded_at', 'location',
                'player', 'bow_velocity', 'bridge_position', 'string', 
                'csv_file', 'csv_id', 'attack', 'decay', 'sustain', 'release',
                'offset', 'reference', 'comments', 'semitone', 'pitch_reference']
    con.close()
    df_sounds = df.drop(columns = cols_drop)
    print("dropped SOUNDS unnecessary columns!")
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
    print("dropped pack_id!")

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
    print("add microphones to dataset")

    con = sqlite3.connect(sqlite_path)
    df_packs = pd.read_sql_query("SELECT * from Packs", con) 
    con.close()


    paths = list()
    for row in range(len(df_sounds)):
        path_row = df_sounds['pack'][row]+'/'+df_sounds['microphone'][row]+'/'+df_sounds['pack_filename'][row]
        paths.append(path_row)
    df_sounds['path'] = paths
    df_sounds.drop(columns=['pack', 'microphone', 'pack_filename', 'index'])
    print("dataset finished!")
    return df_sounds



def wav_filter(file):
    if '.wav' in file: return True
    else: return False


def pre_process(path_to_sounds):
    """ Pré processamento dos áudios, cria um diretório novo com o prefixo norm_
        contendo os arquivos normalizados.
    """
    for root, dirs, filenames in os.walk(path_to_sounds):
        filenames = list(filter(wav_filter, filenames))
        for out in filenames:
            soundpath = os.path.join(root,out)
            root_to_output = 'norm_'+root
            soundpath_out = os.path.join(root_to_output, out) #REVISAR ESSA PARTE
            tfm = sox.Transformer()
            tfm.norm()
            tfm.silence(location = 0, silence_threshold = 2, min_silence_duration = 0.2) #VERIFICAR PARAMETROS DE TEMPO
            tfm.build_file(input_filepath = soundpath,
                                output_filepath = soundpath_out)
    print(f"built preprocessing directory! example of normalized file path: {root_to_output}")


def create_datalist(path_to_datalist, path_to_sounds):
    open(path_to_datalist, 'w').close()
    with open(path_to_datalist, 'w') as f:
        for root, dirs, filenames in os.walk(path_to_sounds):
            filenames = list(filter(wav_filter, filenames))
            for i in filenames:
                f.writelines(os.path.join(root,i)+'\n')
    print("datalist created!")


pre_process(path_sounds)
df_s = clean_sqlite(sqlite_path) 
df_s.to_csv(labels_file)
df = pd.read_csv(labels_file)
encoder = LabelEncoder()
instrument_labels = list(df['instrument'])
labels = encoder.fit_transform(instrument_labels)
df['label'] = labels

create_datalist(path_output_datalist, norm_path_all) #cria o datalist

#criar dicionario .npy com labels
gs_labelfile = dict()
gs_labelfile = dict.fromkeys(df['path'])
for file_index, key in enumerate(df['path']):
    gs_labelfile[key] = df['label'][file_index]
np.save(dict_file_out, gs_labelfile)
print("finished label file!")


