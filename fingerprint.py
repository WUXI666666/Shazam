import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import scipy
import librosa
from scipy import ndimage
#####################################################################
# createfingerprint
# 创建简单指纹，即频谱峰值在矩阵上的散点图
# --- 输入 ---
# x - 时间域音频片段样本
# --- 输出 ---
# output - 指纹矩阵
#####################################################################
def createfingerprint(x, plot=False):
    # 生成频谱图并转换为对数刻度
    X = lr.stft(x, n_fft=1024, hop_length=32, window="blackman")
    X = np.abs(X)
    L, W = np.shape(X)

    # 提取峰值并使用最大滤波器选择邻域中的最高峰
    output = peakextract(X)
    output = scipy.ndimage.maximum_filter(output, size=25)
    max_peak = np.max(output)
    output = np.where(output == 0 , -1 , output)
    output = np.where(X == output, 1, 0)

    # 如果启用，显示包含原始频谱图的指纹图像
    if plot:
        plt.imshow(X)
        y_ind, x_ind = np.where(output != 0)
        plt.scatter(x=x_ind, y=y_ind, c='r', s=8.0)
        plt.gca().invert_yaxis()
        plt.xlabel('Frames')
        plt.ylabel('Bins')
        plt.draw()

    return output
# 创建指纹的函数
# def compute_constellation_map(Y, dist_freq=7, dist_time=7, thresh=0.01):
#     result = ndimage.maximum_filter(Y, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
#     Cmap = np.logical_and(Y == result, result > thresh)
#     return Cmap
# def createfingerprint(x, plot=False):
#     # 生成频谱图并转换为对数刻度
#     X = librosa.stft(x, n_fft=1024, hop_length=32, window="blackman")
#     X = np.abs(X)
#     L, W = np.shape(X)

#     # 提取峰值
#     output = compute_constellation_map(X, dist_freq=7, dist_time=7, thresh=0.01)

#     # 如果启用，显示包含原始频谱图的指纹图像
#     if plot:
#         plt.imshow(np.log1p(X), origin='lower', aspect='auto', cmap='gray_r', interpolation='nearest')
#         y_ind, x_ind = np.where(output != 0)
#         plt.scatter(x=x_ind, y=y_ind, c='r', s=8.0)
#         plt.gca().invert_yaxis()
#         plt.xlabel('Time (frames)')
#         plt.ylabel('Frequency (bins)')
#         plt.title('Constellation Map')
#         plt.show()

#     return output
#####################################################################
# peakextract
# 从频谱图中的每个时间帧中提取峰值
# --- 输入 ---
# S - 频谱图
# --- 输出 ---
# peaks - 从每个频段中提取峰值的矩阵
#####################################################################
def peakextract(S):
    # 初始化峰值矩阵
    row, col = np.shape(S)
    peaks = np.zeros((row, col))

    # 将频率谱分为对数频段
    bands = np.array([[1, 11], [12, 21], [22, 41], [42, 81], [82, 161], [162, 513]])

    # 在每个频段中的每个时间帧中找到最大值并更新峰值矩阵
    for i in range(col):
        for j in range(6):
            q1 = bands[j, 0]
            q2 = bands[j, 1]
            frame_band = S[q1:q2, i]
            localpeak = np.max(frame_band)
            index = np.where(frame_band == localpeak)
            peaks[q1 + index[0], i] = localpeak

    return peaks

#####################################################################
# createhashes
# 从峰值矩阵中创建哈希并存储在表中。哈希由峰值矩阵中的时间频率对组成。
# --- 输入 ---
# peaks - 包含峰值位置的二进制矩阵
# offset - 服务器端的锚点帧号
# --- 输出 ---
# T - 哈希表
# O - 偏移量（仅适用于服务器端）
#####################################################################
def createhashes(peaks, offset=False):
    # 转置以使帧按顺序排列
    peaks = np.transpose(peaks)
    # 找到峰值位置并计算峰值数量
    points = np.where(peaks != 0)#非0元素索引[x,y]
    num_points = np.shape(points[0])[0]#length

    # 创建包含空列表的空列表，索引为 0-512
    T = [[] for i in range(513)]#存储哈希值傅里叶变换的对称性质导致我们只需要正频率分量。

    # 仅在服务器端使用
    if offset:
        O = [[] for i in range(513)]#偏移

    # 更新哈希表
    for i in range(num_points):
        for j in range(num_points - i):
            # 如果帧的时间差在 1-50 之间，则生成浮点数 f1.delta_t 并添加到列表中
            if abs(points[0][i] - points[0][i+j]) != 0 and abs(points[0][i] - points[0][i+j]) < 51:
                if abs(points[0][i] - points[0][i+j]) < 10:
                    T[points[1][i]].append(float(str(points[1][i+j]) + '.0' + \
                    str(abs(points[0][i] - points[0][i+j]))))#锚点f，点f,时间差
                else:
                    T[points[1][i]].append(float(str(points[1][i+j]) + '.' + \
                    str(abs(points[0][i] - points[0][i+j]))))

                # 仅在服务器端使用
                if offset:
                    O[points[1][i]].append(points[0][i])#f->t频率下对应时间

    # 服务器端
    if offset:
        return T, O
    # 客户端
    else:
        return T

# def createhashes(peaks, offset=False):
#     # 转置以使帧按顺序排列
#     peaks = np.transpose(peaks)
#     # 找到峰值位置并计算峰值数量
#     points = np.where(peaks != 0)
#     num_points = np.shape(points[0])[0]

#     # 创建包含空列表的空列表，索引为 0-512
#     T = [[] for _ in range(513)]

#     # 仅在服务器端使用
#     if offset:
#         O = [[] for _ in range(513)]

#     # 更新哈希表
#     for i in range(num_points):
#         anchor_time = points[0][i]  # 锚点时间
#         anchor_freq = points[1][i]  # 锚点频率
        
#         # 选择后续的目标点，最多选择5个点
#         for j in range(1, 6):
#             if i + j < num_points:
#                 target_time = points[0][i + j]
#                 target_freq = points[1][i + j]
#                 delta_time = target_time - anchor_time

#                 # 生成哈希值，并确保时间差在1到50之间
#                 if 1 <= delta_time < 51:
#                     hash_value = f"{anchor_freq}-{target_freq}-{delta_time}"
#                     T[anchor_freq].append(hash_value)
                    
#                     # 仅在服务器端使用
#                     if offset:
#                         O[anchor_freq].append(anchor_time)

#     # 服务器端
#     if offset:
#         return T, O
#     # 客户端
#     else:
#         return T

