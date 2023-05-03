from scipy.io import wavfile
from scipy.signal import sosfreqz
import numpy as np
import _pickle
import h5py
import math
from keras.models import load_model
import os
import tensorflow as tf
import soundfile
import librosa
import wave
import random
from scipy import signal as sg
import matplotlib.pyplot as plt

Num_of_filter = 10
up_Q = 60
up_F = 7000
up_G = 35
down_Q = 0.001
down_F = 25
down_G = -35
Num_of_all = 1000
Num_of_iteration = 1000
alpha_alive = 0.1
num_of_son = 8
alpha_new = 0.1
x0 = np.zeros([Num_of_filter, 4])
x_all = np.ones([Num_of_all, Num_of_filter, 4])
ipsino = 0.000000001
best_x = np.zeros([Num_of_filter, 4])
best_value = 0
n_fft = 256
f0 = 1000
Q = 5
fs = 16000
gain = -10
#sos = peak_filter_iir(f0, gain, Q, fs)
#w, h = sg.sosfreqz(sos, worN=n_fft, fs=fs)
up_Hz = 3000
target = np.ones([n_fft])



def low_shelf_filter_iir(f0, gain=0., Q=1., fs=16000):
    """
    根据PEQ参数设计二阶IIR数字low shelf滤波器，默认采样率192k
    :param f0: 中心频率
    :param gain: 峰值增益
    :param Q: 峰值带宽
    :param fs: 系统采样率
    :return: 双二阶滤波器系数
    """
    A = np.sqrt(10 ** (gain / 20))
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])

    h = np.hstack((b / a[0], a / a[0]))

    return h


def peak_filter_iir(f0, gain=0., Q=1., fs=16000):
    """
    根据PEQ参数设计二阶IIR数字peak滤波器，默认采样率192k
    :param f0: 中心频率
    :param gain: 峰值增益，正值为peak filter,负值为notch filter
    :param Q: 峰值带宽
    :param fs: 系统采样率
    :return: 双二阶滤波器系数
    """
    A = np.sqrt(10 ** (gain / 20))
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])

    h = np.hstack((b / a[0], a / a[0]))

    return h


def high_shelf_filter_iir(f0, gain=0., Q=1., fs=16000):
    """
    根据PEQ参数设计二阶IIR数字high shelf滤波器，默认采样率192k
    :param f0: 中心频率
    :param gain: 峰值增益
    :param Q: 峰值带宽
    :param fs: 系统采样率
    :return: 双二阶滤波器系数
    """
    A = np.sqrt(10 ** (gain / 20))
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
    a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])

    h = np.hstack((b / a[0], a / a[0]))

    return h


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


def f(x):
    loss = 0
    filter_h = np.ones_like(target)
    for i in range(Num_of_filter):
        fliter_n = x[i, :]
        if fliter_n[0] == 0:
            sos = low_shelf_filter_iir(fliter_n[2], fliter_n[3], fliter_n[1], fs)
        if fliter_n[0] == 1:
            sos = peak_filter_iir(fliter_n[2], fliter_n[3], fliter_n[1], fs)
        if fliter_n[0] == 2:
            sos = high_shelf_filter_iir(fliter_n[2], fliter_n[3], fliter_n[1], fs)
        w, h = sg.sosfreqz(sos, worN=256, fs=fs)
        h_mag = np.abs(h)
        filter_h = filter_h * h_mag
    for i in range(n_fft):
        if w[i] < up_Hz:
            loss = loss + np.abs(target[i] - filter_h[i])
    loss_updown = 1 / (loss + ipsino)
    return loss_updown

'''audio, fs = read_audio("test.wav")
sos = peak_filter_iir(f0, gain, Q, fs)
length = len(audio)
audio_out = np.zeros_like(audio)
in2 = 0
in1 = 0
out1 = 0
out2 = 0
b0 = sos[0]
b1 = sos[1]
b2 = sos[2]
a0 = sos[3]
a1 = sos[4]
a2 = sos[5]
for i in range(length):
    audio_out[i] = b0 / a0 * audio[i] + b1 / a0 * in1 + b2 / a0 * in2 - a1 / a0 * out1 - a2 / a0 * out2
    in2 = in1
    in1 = audio[i]
    out2 = out1
    out1 = audio_out[i]
write_audio("test_out.wav", audio_out, fs)
sos = peak_filter_iir(f0, gain, Q, fs)
w, h = sg.sosfreqz(sos, worN=256, fs=fs)
fig, ax1 = plt.subplots()
ax1.semilogx(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency')
ax1.grid()
ax2 = ax1.twinx()
ax2.semilogx(w, np.angle(h, deg=True), 'r')
ax2.set_ylabel('Angle [deg]', color='r')
ax2.axis('tight')
plt.show()'''

a=1






for i in range(Num_of_all):
    for j in range(Num_of_filter):
        x_all[i, j, 0] = int(random.randint(0, 2))
        x_all[i, j, 1] = random.uniform(down_Q, up_Q)
        x_all[i, j, 2] = random.uniform(down_F, up_F)
        x_all[i, j, 3] = random.uniform(down_G, up_G)

f_value = np.zeros([Num_of_all])
idex_alive = np.zeros([int(Num_of_all * alpha_alive)])
derta_Q = (up_Q - down_Q) / 2
derta_F = (up_F - down_F) / 2
derta_G = (up_G - down_G) / 2

for iter in range(Num_of_iteration):
    for i in range(Num_of_all):
        f_value[i] = f(x_all[i, :, :])
    max_value = np.max(f_value)
    max_idex = np.array(np.where(f_value == max_value))
    best_value = max_value
    print(iter, best_value)
    best_x = x_all[max_idex, :, :]
    tmp_num = 0
    for i in range(int(Num_of_all * alpha_alive)):
        max_value = np.max(f_value)
        max_idex = np.array(np.where(f_value == max_value))
        for j in range(len(max_idex.T)):
            idex_alive[tmp_num] = max_idex[0, j]
            tmp_num = tmp_num + 1
            f_value[max_idex[0, j]] = 0
            if tmp_num == Num_of_all * alpha_alive:
                break
        if tmp_num == Num_of_all * alpha_alive:
            break
    for i in range(int(Num_of_all * alpha_alive)):
        x_all[i, :, :] = x_all[int(idex_alive[i]), :, :]
##############################################################################
    for i in range(int(Num_of_all * alpha_alive)):
        for j in range(num_of_son):
            for k in range(Num_of_filter):
                x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 0] = int(random.randint(0, 2))

                x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] = x_all[i, k, 1] + 0.01 * random.uniform(-derta_Q, derta_Q)
                if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] > up_Q:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] = up_Q
                if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] < down_Q:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] = down_Q

                x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] = x_all[i, k, 2] + 0.001 * random.uniform(-derta_F, derta_F)
                if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] > up_F:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] = up_F
                if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] < down_F:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] = down_F

                x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] = x_all[i, k, 3] + 0.01 * random.uniform(-derta_G, derta_G)
                if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] > up_G:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] = up_G
                if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] < down_G:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] = down_G
#########################################################################################3
    for i in range(int(Num_of_all * alpha_new)):
        for j in range(Num_of_filter):
            x_all[Num_of_all - i - 1, j, 0] = int(random.randint(0, 2))
            x_all[Num_of_all - i - 1, j, 1] = random.uniform(down_Q, up_Q)
            x_all[Num_of_all - i - 1, j, 2] = random.uniform(down_F, up_F)
            x_all[Num_of_all - i - 1, j, 3] = random.uniform(down_G, up_G)


print(best_x)
print(1 / best_value)

a=1





a=1
