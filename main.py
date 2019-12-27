import json
import math
import os
import glob
import time
import wave
import threading
import face_recognition
import numpy as np
import cv2
import pyaudio as pyaudio
from moviepy.editor import *
import matplotlib.pyplot as plt

frame = None
cap_real = None


def get_face():
    global frame
    down_scale_factor = 0.25
    while True:
        if frame is None:
            time.sleep(0.2)
            continue
        frame_bak = np.copy(frame)
        cam_h, cam_w, _ = frame_bak.shape
        small_frame = cv2.resize(frame_bak, (0, 0), fx=down_scale_factor, fy=down_scale_factor)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        for index, (y1_sm, x2_sm, y2_sm, x1_sm) in enumerate(face_locations):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            x1 = int(x1_sm / down_scale_factor)
            x2 = int(x2_sm / down_scale_factor)
            y1 = int(y1_sm / down_scale_factor)
            y2 = int(y2_sm / down_scale_factor)

            x1_rltv = x1 / cam_w
            x2_rltv = x2 / cam_w
            y1_rltv = y1 / cam_h
            y2_rltv = y2 / cam_h

            _face_area = frame_bak[y1:y2, x1:x2, :]
            if _face_area.size == 0:
                continue
            cv2.imshow(f'faces{index}', frame_bak[y1:y2, x1:x2, :])
            cv2.waitKey(1)


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
    root2 = "E:\\DATA\\dfdc\\deepfake-detection-challenge\\train_sample_videos"
    root = "E:\\DATA\\dfdc\\dfdc_train_part_00\\dfdc_train_part_0"
    file_list = os.listdir(root2)
    label_file = "metadata.json"
    video_list = glob.glob(os.path.join(root, "*.mp4"))
    file_list = file_list + [os.path.basename(name) for name in video_list]
    labels = json.load(open(os.path.join(root, label_file)))
    t = threading.Thread(target=get_face)
    t.setDaemon(True)
    # t.start()
    for index, (key, value) in enumerate(labels.items()):
        if value["label"] == "FAKE":
            real_video = value["original"]
            if real_video in file_list:
                cap_real = cv2.VideoCapture(os.path.join(root, real_video))
                video = VideoFileClip(os.path.join(root, real_video), audio_fps=48000)
                audio = video.audio
                wave_data = audio.to_soundarray()
                nchannels = audio.nchannels
                framerate = audio.fps
                wave_data = np.asarray(wave_data * 32768, np.short)
                # plt.plot(wave_data[:, 0])
                wave_data = wave_data[:, 0]
                image_real_s = show(wave_data, "real_s")
                # cv2.waitKey(0)
                print(key, real_video)
            else:
                cap_real = None
                continue
        else:
            continue
        # print(value["label"])
        if index < 2:
            continue
        cap = cv2.VideoCapture(os.path.join(root, key))
        frame_size = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        video = VideoFileClip(os.path.join(root, key), audio_fps=48000)
        # audio2 = AudioFileClip(os.path.join(root, key))

        audio = video.audio
        wave_data = audio.to_soundarray()
        nchannels = audio.nchannels
        buffersize = audio.buffersize
        framerate = audio.fps
        sampwidth = 2

        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(sampwidth),
                        channels=nchannels,
                        rate=framerate,
                        output=True)
        volume = 1  # 音量大小（正常响度的倍数）
        wave_data = wave_data * volume
        wave_data = np.asarray(wave_data * 32768, np.short)
        # plt.plot(wave_data[:, 0])
        plt.show()
        image_fake_s = show(wave_data[:, 0], "fake_s")
        print(np.sum(image_fake_s-image_real_s))

        wave_data = np.maximum(np.minimum(wave_data, 32767), -32768)
        wave_data = wave_data.flatten()
        # wave_data = np.reshape(wave_data, (-1))
        audio_frame_len = int(nchannels * framerate / fps)
        data = wave_data[:audio_frame_len].tobytes()
        i = 1
        if np.sum(np.abs(image_fake_s-image_real_s))>0:
            while data != b'':
                s = time.time()
                stream.write(data)
                data = wave_data[int(i * audio_frame_len):int(i * audio_frame_len + audio_frame_len)].tobytes()
                su, frame = cap.read()
                if not su:
                    break
                if not cap_real is None:
                    su, frame_real = cap_real.read()
                    height, width, _ = frame_real.shape
                    h = 480
                    w = int(width * h / height)
                    # cv2.imshow("frame_real", cv2.resize(frame_real, (w, h)))
                else:
                    cv2.destroyWindow("frame_real")
                height, width, _ = frame.shape
                h = 480
                w = int(width * h / height)
                cv2.putText(frame, value["label"], (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                # cv2.imshow("image", cv2.resize(frame, (w, h)))
                cv2.waitKey(1)
                i += 1
