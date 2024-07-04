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
from record import recordaudio
# 指纹提取参数
dist_freq = 11
dist_time = 7
tol_freq = 1
tol_time = 1

# 指纹提取功能
def compute_constellation_map(Y, dist_freq=7, dist_time=7, thresh=0.01):
    result = ndimage.maximum_filter(Y, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
    Cmap = np.logical_and(Y == result, result > thresh)
    return Cmap

def plot_constellation_map(Cmap, Y=None, xlim=None, ylim=None, title='',
                           xlabel='Time (sample)', ylabel='Frequency (bins)',
                           s=5, color='r', marker='o', figsize=(7, 3), dpi=72):
    if Cmap.ndim > 1:
        (K, N) = Cmap.shape
    else:
        K = Cmap.shape[0]
        N = 1
    if Y is None:
        Y = np.zeros((K, N))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    im = ax.imshow(Y, origin='lower', aspect='auto', cmap='gray_r', interpolation='nearest')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    Fs = 1
    if xlim is None:
        xlim = [-0.5/Fs, (N-0.5)/Fs]
    if ylim is None:
        ylim = [-0.5/Fs, (K-0.5)/Fs]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    n, k = np.argwhere(Cmap == 1).T
    ax.scatter(k, n, color=color, s=s, marker=marker)
    plt.tight_layout()
    return fig, ax, im

def compute_spectrogram(audio_path, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    x, Fs = librosa.load(audio_path, sr=Fs)
    x_duration = len(x) / Fs
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
    if bin_max is None:
        bin_max = X.shape[0]
    if frame_max is None:
        frame_max = X.shape[0]
    Y = np.abs(X[:bin_max, :frame_max])
    return Y

def match_binary_matrices_tol(C_ref, C_est, tol_freq=0, tol_time=0):
    assert C_ref.shape == C_est.shape, "Dimensions need to agree"
    N = np.sum(C_ref)
    M = np.sum(C_est)
    C_est_max = ndimage.maximum_filter(C_est, size=(2*tol_freq+1, 2*tol_time+1), mode='constant')
    C_AND = np.logical_and(C_est_max, C_ref)
    TP = np.sum(C_AND)
    FN = N - TP
    FP = M - TP
    return TP, FN, FP, C_AND

def compute_matching_function(C_D, C_Q, tol_freq=1, tol_time=1):
    L = C_D.shape[1]
    N = C_Q.shape[1]
    M = L - N
    assert M >= 0, "Query must be shorter than document"
    Delta = np.zeros(L)
    for m in range(M + 1):
        C_D_crop = C_D[:, m:m+N]
        TP, FN, FP, C_AND = match_binary_matrices_tol(C_D_crop, C_Q, tol_freq=tol_freq, tol_time=tol_time)
        Delta[m] = TP
    shift_max = np.argmax(Delta)
    return Delta, shift_max



# 加载数据的函数
def load_database():
    with open('server_tables.pkl', 'rb') as f:#with用于上下文管理，确保文件在使用完毕后自动关闭。
        server_tables = pickle.load(f)#反序列化
    with open('offset_times.pkl', 'rb') as f:
        offset_times = pickle.load(f)
    with open('song_names.pkl', 'rb') as f:
        song_names = pickle.load(f)
    return server_tables, offset_times, song_names

# 比较两个指纹表
def comparetables(table):
    server_tables, offset_times, _ = load_database()
    matched_pairs_list = np.zeros((len(server_tables), 1))
    hist_list = np.zeros((len(server_tables), 1))
    for h in range(len(server_tables)):
        matched_pairs = 0
        add_to_hist = []#偏移时间
        for i in range(513):#遍历每个频率位置
            if table[i]:
                for j in table[i]:#每个频率的指纹
                    if j in server_tables[h][i]:
                        indices = [i for i, x in enumerate(server_tables[h][i]) if x == j]
                        #enumerate 函数用于将这个列表生成一个枚举对象，每个元素都是一个 (索引, 值) 对。
                        matched_pairs += len(indices)
                        for o in indices:
                            add_to_hist.append(offset_times[h][i][o])
        matched_pairs_list[h] = matched_pairs
        if add_to_hist:
            maxh = np.max(add_to_hist)
            hist, edges = np.histogram(add_to_hist, bins=range(0, (maxh + (maxh + 1) % 2050), 2050))
            hist_list[h] = np.max(hist)#一个区间中最多匹配次数
    return matched_pairs_list, hist_list

# 查找歌曲ID，通过衡量匹配比率和集中度，如果 max_hist 值高，说明在某个时间偏移上有大量的匹配，意味着在这个时间偏移上，两个音频的相似度非常高。
def fetchID(fprint):
    matched_pairs, hists = comparetables(fprint)
    num_pairs = sum(map(len, fprint))
    
    # 检查是否有匹配
    if num_pairs == 0 or len(matched_pairs) == 0:
        return "Not found!"
    
    ratios = matched_pairs / num_pairs
    
    # 检查 ratios 是否为空
    if len(ratios) == 0 or np.all(ratios == 0):
        return "Not found!"
    
    max_match = np.max(ratios)
    max_hist = np.max(hists)
    id_value = np.where(ratios == max_match)
    id_hist = np.where(hists == max_hist)
    
    # 检查 id_value 和 id_hist 是否为空，并提取单个元素
    if id_value[0].size == 0 or id_hist[0].size == 0 or id_value[0][0] != id_hist[0][0]:
        id = -1
    else:
        id = id_value
    
    _, _, song_names = load_database()
    if id == -1:
        name = "Not found!"
    else:
        name = song_names[id[0][0]]
    
    print('Number of Matches\t' + 'Ratio\t\t' + 'Hist max\t\t' 'Song Name')
    for i in range(min(10, len(matched_pairs))):
        matched_pairs_value = matched_pairs[i][0] if matched_pairs[i].size > 0 else 0
        ratios_value = ratios[i][0] if ratios[i].size > 0 else 0
        hists_value = hists[i][0] if hists[i].size > 0 else 0
        song_name = song_names[i] if len(song_names) > i else "Unknown"
        print(f"{matched_pairs_value}\t\t\t{ratios_value:.4f}\t\t{hists_value:.4f}\t\t{song_name}")
    
    return name


# 听歌识曲功能
def recognize_song_from_path(query_path):
    x, fs = librosa.load(query_path, sr=8192, mono=True)
    F_print = createfingerprint(x)
    T = createhashes(F_print, offset=False)
    song_name = fetchID(T)
    return song_name
def recognize_song(x):
    F_print = createfingerprint(x)
    T = createhashes(F_print, offset=False)
    song_name = fetchID(T)
    return song_name
# 比较两个音频文件
def compare_2songs(path1, path2):
    Y1 = compute_spectrogram(path1)
    Y2 = compute_spectrogram(path2)
    CM1 = compute_constellation_map(Y1, dist_freq, dist_time)
    CM2 = compute_constellation_map(Y2, dist_freq, dist_time)
    Delta, shift_max = compute_matching_function(CM1, CM2, tol_freq=tol_freq, tol_time=tol_time)
    print(Delta[shift_max])
    plot_constellation_map(CM1, np.log(1 + 1 * Y1), color='r', s=30, title=path1)
    plot_constellation_map(CM2, np.log(1 + 1 * Y2), color='r', s=30, title=path2)

# 比较目录中的音频文件
def compare_dir(path, fn_query):
    Y_q = compute_spectrogram(fn_query)
    CMP_q = compute_constellation_map(Y_q, dist_freq, dist_time)
    for fn in os.listdir(path):
        if os.path.isfile(os.path.join(path, fn)):
            if fn.endswith(".wav"):
                fn = os.path.join(path, fn)
                print(fn)
                Y_d = compute_spectrogram(fn)
                CMP_d = compute_constellation_map(Y_d, dist_freq, dist_time)
                Delta, shift_max = compute_matching_function(CMP_d, CMP_q, tol_freq=0, tol_time=0)
                print(Delta[shift_max])
                plot_constellation_map(CMP_d, np.log(1 + 1 * Y_d), color='r', s=30, title=fn)

# 创建数据库
# create_database()

# 比较两个音频文件
# compare_2songs("./songs/NationalAnthemIndia.wav", "./tests/test_3.wav")

# 听歌识曲
song_name = recognize_song_from_path("./tests/test_3.wav")
# song_name=recognize_song (recordaudio())
print(f"Recognized song: {song_name}")

plt.show()
