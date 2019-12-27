import math

import cv2
import numpy as np
import pyaudio as pyaudio
from moviepy.editor import *
import matplotlib.pyplot as plt

def piecewise(data, winfunc=lambda x: np.ones((x,))):
    """
    处理音频数据，将其分成part_num部分
    :param winfunc:
    :param data:
    :return:
    """
    frame_time = 10  # 多少ms一帧(ms)
    frame_step = frame_time / 2  # 帧的步长
    framerate, nframes, wave_data = data
    signal_length = nframes  # 信号总长度
    frame_length = int(round(framerate / 1000 * frame_time))  # 以帧帧时间长度
    frame_length = frame_length
    frame_step = int(round(framerate / 1000 * frame_step))  # 相邻帧之间的步长
    if signal_length <= frame_length:  # 若信号长度小于一个帧的长度，则帧数定义为1
        frames_num = 1
    else:  # 否则，计算帧的总长度
        frames_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
    pad_length = int((frames_num - 1) * frame_step + frame_length)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((wave_data, zeros))  # 填补后的信号记为pad_signal
    x = np.arange(0, frame_length)
    y = np.arange(0, frames_num * frame_step, frame_step)
    a = np.tile(x, (frames_num, 1))
    b = np.tile(y, (frame_length, 1))
    bt = b.T
    indices = a + bt  # 相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    t = winfunc(frame_length)
    win = np.tile(t, (frames_num, 1))  # window窗函数，这里默认取1
    return frames * win  # 返回帧信号矩阵


def audioToSpectrogram(frames, n=2000):
    """
    音频信号转为语谱图
    :param n:
    :param frames:
    :return:
    """
    complex_spectrum = np.fft.rfft(frames, n=n * 2)
    amp_spectrum = np.absolute(complex_spectrum)
    phase = np.angle(complex_spectrum)
    spec = np.log1p(amp_spectrum)
    return amp_spectrum, spec, phase

def show(wave_data, name="spec"):
    matrix = piecewise((framerate, len(wave_data), wave_data))
    amp_spectrum, spec, phase = audioToSpectrogram(matrix)
    image1 = (amp_spectrum - amp_spectrum.min()) / (amp_spectrum.max() - amp_spectrum.min())
    image1 = cv2.resize(image1.T, (800, 200))

    image2 = (spec - spec.min()) / (spec.max() - spec.min())
    image2 = cv2.resize(image2.T, (800, 200))
    cv2.imshow(f"{name}_log", np.vstack((image2, image1)))
    return image2

if __name__ == '__main__':
    root = "E:\\DATA\\dfdc\\dfdc_train_part_00"
    real = "fnslimfagb.mp4"
    fake = "sufvvwmbha.mp4"
    framerate = 48000
    audio_real = AudioFileClip(os.path.join(root, real), fps=framerate)
    audio_fake = AudioFileClip(os.path.join(root, fake), fps=framerate)
    wave_data_real = audio_real.to_soundarray()
    wave_data_real = np.asarray(wave_data_real * 32768, np.short)
    wave_data_real = wave_data_real[:, 0]

    wave_data_fake = audio_fake.to_soundarray()
    wave_data_fake = np.asarray(wave_data_fake * 32768, np.short)
    wave_data_fake = wave_data_fake[:, 0]

    show(wave_data_fake, "fake_s")
    show(wave_data_real, "real_s")
    print()
