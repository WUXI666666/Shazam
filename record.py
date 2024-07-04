import sounddevice as sd
import numpy as np
import librosa as lr

def recordaudio():
    # 设置录音参数
    DURATION = 8  # 录音时长（秒）
    SAMPLE_RATE = 44100  # 采样率
    CHANNELS = 2  # 立体声

    print("* recording *")

    # 录音
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()  # 等待录音结束

    print("* done recording *")

    # 转换录音数据为单声道
    recording_mono = np.mean(recording, axis=1)

    # 重新采样为8192 Hz
    x = lr.resample(recording_mono.astype(float), orig_sr=SAMPLE_RATE, target_sr=8192)

    return x