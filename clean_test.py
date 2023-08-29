import os

path_datalist_all =  'data_lists/goodsounds_all.scp'
path_datalist_train = 'data_lists/goodsounds_train.scp'
path_datalist_test = 'data_lists/goodsounds_test.scp'
path_sounds_norm = 'norm_good-sounds/sound_files' #nao ta na tupla
dict_file_out = 'data_lists/goodsounds_labels.npy'

test_files = (path_datalist_all, path_datalist_train, path_datalist_test, dict_file_out)

for file in test_files:
    os.remove(file)