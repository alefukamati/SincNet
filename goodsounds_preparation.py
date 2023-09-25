""#Monta csv
#transforma csv em dicionario com labels codificadas

#falta fazer preprocessamento de audio
#dividir em test e train (splitfolders)

import pandas as pd
import numpy as np
import os
import sox
import splitfolders
import sqlite3
import shutil
from sklearn.preprocessing import LabelEncoder
from audiomentations import AddGaussianNoise, Compose


dict_file_out = 'data_lists/goodsounds_labels.npy'

path_sounds = 'good-sounds/sound_files'
path_sounds_norm = 'new_norm_good-sounds/'

path_datalist_all =  'data_lists/newgoodsounds_all.scp'
path_datalist_train = 'data_lists/newgoodsounds_train.scp'
path_datalist_test = 'data_lists/newgoodsounds_test.scp'

path_train_sounds = 'new_norm_good-sounds/train'
path_test_sounds = 'new_norm_good-sounds/test'

labels_file = 'good-sounds/good-sounds-labels.csv'
sqlite_path = 'good-sounds/database.sqlite'
#splitfolders.ratio(input = path, output = 'good-sounds', seed = 42, ratio=(.6,.4), group_prefix = None, move = False)


def clean_sqlite(sqlite_path):
    print(f'cleaning sqlite dataframe on {sqlite_path}...')
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
        path_row = df_sounds['pack'][row]+'/'+df_sounds['microphone'][row]+'_'+df_sounds['pack_filename'][row]
        paths.append(path_row)
    df_sounds['path'] = paths
    df_sounds.drop(columns=['pack', 'microphone', 'pack_filename', 'index'])
    print("dataframe ready!")
    return df_sounds


def wav_filter(file):
    if '.wav' in file: return True
    else: return False


def pre_process(path_to_sounds):
    """ Pré processamento dos áudios, cria um diretório novo com o prefixo norm_
        contendo os arquivos normalizados.
    """
    print('starting preprocess...')

    df = pd.read_csv('good-sounds/good-sounds-labels.csv')
    for root, dirs, filenames in os.walk(path_to_sounds):
        filenames = list(filter(wav_filter, filenames))
        for out in filenames:
            soundpath = os.path.join(root,out)
            new_sp = soundpath.split('/')
            tmp = '/'.join(new_sp[2:4])+'_'+new_sp[4]
            if tmp in list(df['path']): 
                root_to_output = 'new_norm_'+root
                if not os.path.isdir(root_to_output): os.makedirs(root_to_output)
                soundpath_out = os.path.join(root_to_output, out)
                tfm = sox.Transformer()
                tfm.norm()
                tfm.silence(location = 0, silence_threshold = 2, min_silence_duration = 0.2) #VERIFICAR PARAMETROS DE TEMPO
                tfm.build_file(input_filepath = soundpath,
                                    output_filepath = soundpath_out)
    print(f"built preprocessing directory! example of normalized file path: {root_to_output}")


def create_datalist(path_to_datalist, path_to_sounds):
    print(f'creating datalist to {path_to_sounds} in path {path_to_datalist}...')
    #f = open(path_to_datalist, 'w').close()
    with open(path_to_datalist, 'w') as f:
        for root, dirs, filenames in os.walk(path_to_sounds):
            filenames = list(filter(wav_filter, filenames))
            for i in filenames:
                pth = os.path.join(root, i)
                tmp = pth.split('/')
                final = '/'.join(tmp[1:])
                f.writelines(final+'\n')
    print("datalist created!")


def concat_datalist(datalist_train, datalist_test, datalist_all):
    print(f'concatenating datalists to {datalist_all}')
    with open(datalist_test, 'r') as fp:
        data1 = fp.read()
    with open(datalist_train, 'r') as fp:
        data2 = fp.read()
    data1 += '\n'
    data1 += data2
    with open(datalist_all, 'w') as fp:
        fp.write(data1)
    print('datalist all created!')


def prepare_split(path): #diretórios exlcuindo diretorios de mics e colocando como prefixo de arquivos
    print('preparing files for split...')
    for root, dirs, filenames in os.walk(path):
        filenames = list(filter(wav_filter, filenames))
        mics = []
        for i in filenames:
            filesrc = os.path.join(root,i)
            tmp = filesrc.split('/')
            rsd = '/'.join(tmp[0:len(tmp)-2])
            mic = tmp[len(tmp)-2]
            if mic not in mics: mics.append(mic)
            filedst = rsd+'/'+mic+'_'+tmp[len(tmp)-1]
            os.rename(filesrc, filedst) 
            print(mics)
            #for m in mics: os.rmdir(m) VERIFICAR ISSO DPS
    print("files renamed for splitting!")


def split(path, output_path): 
    splitfolders.ratio(input = path, output = output_path, seed = 42, ratio=(.8, 0,.2), group_prefix = True, move = False)
    #os.removedirs('norm_good-sounds/val')
    print("folders split into test and train")

def label_dict_dir(path, train = False, test = False):
    labels = dict()
    files = list()
    instruments = list()
    print(f'creating dict for {path}')
    for root, dirs, filenames in os.walk(path):
        filenames = list(filter(wav_filter, filenames))
        for i in filenames:
            filesrc = os.path.join(root,i)
            tmp = filesrc.split('/')
            file = '/'.join(tmp[1:len(tmp)])
            print(file)
            instrument = tmp[2].split('_')[0]
            files.append(file)
            instruments.append(instrument)
    encoder = LabelEncoder()
    inst_encoded = encoder.fit_transform(instruments)
    for x in range(len(files)):
        labels[files[x]] = inst_encoded[x]
    print(f'dict created for {path}!')
    return labels
            
    
def create_labels(root, path_out):
    print('starting to create the labelfile...')
    path_train = root+'train'
    path_test = root+'test'
    labels_train = label_dict_dir(path_train)
    labels_test = label_dict_dir(path_test)
    all_labels = dict()
    all_labels = {**labels_train, **labels_test}
    np.save(path_out, all_labels)
    print('labelfile created!')


def noise_augmentation(path_in):
    augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), ])
    augmented = augment(samples = path_in)

def audio_augmentation():
    return

#LINHAS ABAIXO PRECISAM SER DESCOMENTADAS
#pre_process(path_sounds)
#prepare_split(path_sounds_norm)
#split(path_sounds_norm, path_sounds_norm)
#create_datalist(path_datalist_train, path_train_sounds)
#create_datalist(path_datalist_test, path_test_sounds)
#concat_datalist(path_datalist_train, path_datalist_test, path_datalist_all)

create_labels(path_sounds_norm, dict_file_out)


#df_s = clean_sqlite(sqlite_path) 
#df_s.to_csv(labels_file)
#df = pd.read_csv(labels_file)
#print("encoding labels...")

#instrument_labels = list(df['instrument']