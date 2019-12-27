import numpy as np
import pyaudio as pyaudio
from moviepy.editor import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    root = "E:\\DATA\\dfdc\\dfdc_train_part_00"
    real = "fnslimfagb.mp4"
    fake = "sufvvwmbha.mp4"

    audio_real = AudioFileClip(os.path.join(root, real), fps=48000)
    audio_fake = AudioFileClip(os.path.join(root, fake), fps=48000)
    wave_data_real = audio_real.to_soundarray()
    wave_data_fake = audio_fake.to_soundarray()
    plt.plot(wave_data_fake[:, 0])
    plt.plot(wave_data_real[:, 0])
    plt.show()
    print()
