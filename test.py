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

# 示例：录制音频并绘制波形
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # audio_data = recordaudio()
    audio_data=[20,4,40,8]
    plt.figure(figsize=(14, 5))
    plt.plot(audio_data)
    plt.title("Recorded Audio Waveform")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()
