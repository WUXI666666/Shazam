from genericpath import isfile
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle
from fingerprint import createfingerprint, createhashes
import glob

# 保存数据的函数
def save_database(server_tables, offset_times, song_names):
    with open('server_tables.pkl', 'wb') as f:
        pickle.dump(server_tables, f)
    with open('offset_times.pkl', 'wb') as f:
        pickle.dump(offset_times, f)
    with open('song_names.pkl', 'wb') as f:
        pickle.dump(song_names, f)
# 创建数据库
def create_database():
    server_tables = []
    offset_times = []
    files = []
    song_names = []
    extensions = ['*.wav']
    for ext in extensions:
        for filename in glob.glob(os.path.join('songs', ext)):
            files.append(filename)
            song_names.append(os.path.basename(filename))
    for i in range(len(files)):
        x, fs = librosa.load(files[i], sr=8192, mono=True)
        F_print = createfingerprint(x)
        T, O = createhashes(F_print, offset=True)
        server_tables.append(T)
        offset_times.append(O)
        print(i)
    save_database(server_tables, offset_times, song_names)
create_database()