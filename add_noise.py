#pegar path ou datalist e utilizar para gerar novas amostras com ruido

#ouvir audios do github para comparar e verificar se os parametros de ruido estão bons
from IPython.display import Audio
import librosa
from audiomentations import AddGaussianNoise, Compose
import os
import soundfile as sf
import numpy as np

path_sounds_norm = 'new_norm_good-sounds/'
path_noise = 'backnoise_norm-good-sounds/'

def wav_filter(file):
    if '.wav' in file: return True
    else: return False

for root, dirs, filenames in os.walk(path_sounds_norm):
        filenames = list(filter(wav_filter, filenames))
        for i in filenames:
            filesrc = os.path.join(root,i)
            if 'sound_files' in filesrc: continue
            audio, sr = librosa.load(filesrc)
            augment = Compose([
                # AddBackgroundNoise() ])
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1), ])
            augmented = augment(samples = audio, sample_rate = sr)
            filedst = '/'.join(filesrc.split('/')[1:])
            filedst = path_noise+filedst
            if os.path.isfile(filedst): continue
            try:
                sf.write(filedst, augmented, sr)
            except sf.LibsndfileError:
                f = filedst.split('/')
                for i in range(2,4):
                     test = '/'.join(f[0:i])
                     if (os.path.isdir(test) == False): 
                        os.mkdir(test)
                        print(f'created directory {test}')
                sf.write(filedst, augmented, sr)
            print(f'Generated file {filedst}')


