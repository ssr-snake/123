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
import csv
from scipy import signal as sg
import matplotlib.pyplot as plt

Num_of_filter = 15
up_Q = 12
up_F = 3100
down_Q = 0.1
down_F = 50
Num_of_all = 1000
Num_of_iteration = 2000
alpha_alive = 0.1
num_of_son = 6
num_of_son_single = 6
num_of_son_double = 6
alpha_new = 0.3
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
up_Hz = 3001
down_Hz = 99

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


def f(x, target_n):
    loss = 0
    filter_h = np.ones_like(target_n)
    for im in range(Num_of_filter):
        fliter_n = x[im, :]
        if fliter_n[0] == 0:
            sos = low_shelf_filter_iir(fliter_n[2], fliter_n[3], fliter_n[1], fs)
            # sos = peak_filter_iir(fliter_n[2], fliter_n[3], fliter_n[1], fs)
        if fliter_n[0] == 1:
            sos = peak_filter_iir(fliter_n[2], fliter_n[3], fliter_n[1], fs)
        if fliter_n[0] == 2:
            sos = high_shelf_filter_iir(fliter_n[2], fliter_n[3], fliter_n[1], fs)
            # sos = peak_filter_iir(fliter_n[2], fliter_n[3], fliter_n[1], fs)
        w, h = sg.sosfreqz(sos, worN=128, fs=fs)
        h_mag = np.abs(h)
        filter_h = filter_h * h_mag
    for im in range(1, int(n_fft/2)):
        if w[im] < up_Hz:
            if w[im] > down_Hz:
                # loss = loss + np.abs(target_n[im] - filter_h[im])
                loss = loss + np.abs(target_n[im] - 20 * np.log10(filter_h[im]))
    loss_updown = 1 / (loss + ipsino)
    return loss_updown


def filter_on(audio_in, filter_in):
    length = len(audio_in)
    if filter_in[0] == 0:
        # sos = peak_filter_iir(filter_in[2], filter_in[3], filter_in[1], fs)
        sos = low_shelf_filter_iir(filter_in[2], filter_in[3], filter_in[1], fs)
    if filter_in[0] == 1:
        sos = peak_filter_iir(filter_in[2], filter_in[3], filter_in[1], fs)
    if filter_in[0] == 2:
        # sos = peak_filter_iir(filter_in[2], filter_in[3], filter_in[1], fs)
        sos = high_shelf_filter_iir(filter_in[2], filter_in[3], filter_in[1], fs)
    audio_out = np.zeros_like(audio_in)
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
    for it in range(length):
        audio_out[it] = b0 / a0 * audio_in[it] + b1 / a0 * in1 + b2 / a0 * in2 - a1 / a0 * out1 - a2 / a0 * out2
        in2 = in1
        in1 = audio_in[it]
        out2 = out1
        out1 = audio_out[it]
    return audio_out


def read_Q(name):
    with open(name, newline='') as csvfile:
        Q_va = np.zeros([128])
        num = 0
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            Q_va[num] = float(''.join(row))
            # Q_va[num] = pow(10, Q_va[num]/20)
            num = num + 1
    return Q_va


Q_target = read_Q('target.csv')
print(Q_target)
Q_now = read_Q('now.csv')
print(Q_now)
target = Q_target - Q_now
print(target)
up_G = np.max(target) * 1.1
down_G = np.min(target) * 1.1
print(up_G)
print(down_G)

for i in range(Num_of_all):
    for j in range(Num_of_filter):
        x_all[i, j, 0] = 1
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
        f_value[i] = f(x_all[i, :, :], target)
    max_value = np.max(f_value)
    max_idex = np.array(np.where(f_value == max_value))
    best_value = max_value
    print(iter, (1 / best_value) / 47)
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
        # print(i)
        x_all[i, :, :] = x_all[int(idex_alive[i]), :, :]
##############################################################################
    for i in range(int(Num_of_all * alpha_alive)):
        for j in range(num_of_son):
            idex_1 = int(random.randint(0, int(Num_of_all * alpha_alive) - 1))
            idex_2 = int(random.randint(0, int(Num_of_all * alpha_alive) - 1))
            # print(idex_1, idex_2)
            for k in range(Num_of_filter):
                if int(random.randint(0, 1)) == 0:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, :] = x_all[idex_1, k, :]
                else:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, :] = x_all[idex_2, k, :]

            for k in range(Num_of_filter):
                # x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 0] = int(random.randint(0, 2))

                if int(random.randint(0, Num_of_iteration)) - 0.1 * Num_of_iteration < iter:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] = x_all[i, k, 1] + 0.01 * random.uniform(-derta_Q, derta_Q)
                    if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] > up_Q:
                        x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] = up_Q
                    if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] < down_Q:
                        x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] = down_Q
                    
                if int(random.randint(0, Num_of_iteration)) - 0.1 * Num_of_iteration < iter:
                    x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] = x_all[i, k, 2] + 0.001 * random.uniform(-derta_F, derta_F)
                    if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] > up_F:
                        x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] = up_F
                    if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] < down_F:
                        x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] = down_F
                    
                if int(random.randint(0, Num_of_iteration)) - 0.1 * Num_of_iteration < iter:
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

audio, fs = read_audio("now.wav")
for i in range(Num_of_filter):
    audio = filter_on(audio, best_x[0, 0, i, :])
write_audio("EQ_out.wav", audio, fs)
