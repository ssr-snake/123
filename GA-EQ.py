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

# def f(x):
#     loss = 0
#     for i in range(10):
#         loss = loss - np.max(x[:, :] * x[:, :])
#     return loss
#
# up_Q = 30
# up_F = 8000
# up_G = 10
# down_Q = -30
# down_F = 0
# down_G = 0
# Num_of_all = 100
# Num_of_iteration = 100
# alpha_alive = 0.1
# num_of_son = 8
# alpha_new = 0.1
# x0 = np.zeros([4, 10])
# x_all = np.ones([Num_of_all, 10, 4])
#
# f0 = 1000
# Q = 1
# fs = 48000
# gain = 10


import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

sr, x = wavfile.read('test.wav')

x = signal.decimate(x, 4)
# a = x[1:5]
# x = x[48000*3:48000*3+8192]
x = x[: 256]
x *= np.hamming(256)

X = abs(np.fft.rfft(x))
X_db = 20 * np.log10(X)
freqs = np.fft.rfftfreq(256, 1/44100)
plt.plot(freqs, X_db)
plt.show()


# def high_shelf_filter_iir(f0, gain=0., Q=1., fs=192000):
#     """
#     根据PEQ参数设计二阶IIR数字high shelf滤波器，默认采样率192k
#     :param f0: 中心频率
#     :param gain: 峰值增益
#     :param Q: 峰值带宽
#     :param fs: 系统采样率
#     :return: 双二阶滤波器系数
#     """
#     A = np.sqrt(10 ** (gain / 20))
#     w0 = 2 * np.pi * f0 / fs
#     alpha = np.sin(w0) / (2 * Q)
#
#     b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
#     b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
#     b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
#     a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
#     a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
#     a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
#
#     b = np.array([b0, b1, b2])
#     a = np.array([a0, a1, a2])
#
#     h = np.hstack((b / a[0], a / a[0]))
#
#     return h
#
# sos = high_shelf_filter_iir(f0, gain, Q, fs)
# w, h = sg.sosfreqz(sos, worN=4096, fs=fs)
#
# fig, ax1 = plt.subplots()
# ax1.semilogx(w, 20 * np.log10(abs(h)), 'b')
# ax1.set_ylabel('Amplitude [dB]', color='b')
# ax1.set_xlabel('Frequency')
# ax1.grid()
# ax2 = ax1.twinx()
# ax2.semilogx(w, np.angle(h, deg=True), 'r')
# ax2.set_ylabel('Angle [deg]', color='r')
# ax2.axis('tight')
# plt.show()

# for i in range(Num_of_all):
#     for j in range(10):
#         x_all[i, j, 0] = int(random.randint(0, 2))
#         x_all[i, j, 1] = random.uniform(down_Q, up_Q)
#         x_all[i, j, 2] = random.uniform(down_F, up_F)
#         x_all[i, j, 3] = random.uniform(down_G, up_G)
#
# # for i in range(Num_of_all):
# #     for j in range(10):
# #         x_all[i, j, 1] = random.uniform(down_Q, up_Q)
# #
# # for i in range(Num_of_all):
# #     for j in range(10):
# #         x_all[i, j, 2] = random.uniform(down_F, up_F)
# #
# # for i in range(Num_of_all):
# #     for j in range(10):
# #         x_all[i, j, 3] = random.uniform(down_G, up_G)
#
# f_value = np.zeros([Num_of_all])
# idex_alive = np.zeros([int(Num_of_all * alpha_alive)])
# derta_Q = (up_Q - down_Q) / 2
# derta_F = (up_F - down_F) / 2
# derta_G = (up_G - down_G) / 2
#
# for iter in range(Num_of_iteration):
#     for i in range(Num_of_all):
#         f_value[i] = f(x_all[i, :, :])
#     tmp_num = 0
#     for i in range(int(Num_of_all * alpha_alive)):
#         max_value = np.max(f_value)
#         max_idex = np.array(np.where(f_value == max_value))
#         for j in range(len(max_idex.T)):
#             idex_alive[tmp_num] = max_idex[0, j]
#             tmp_num = tmp_num + 1
#             f_value[max_idex[0, j]] = 0
#             if tmp_num == Num_of_all * alpha_alive:
#                 break
#         if tmp_num == Num_of_all * alpha_alive:
#             break
#     for i in range(int(Num_of_all * alpha_alive)):
#         x_all[i, :, :] = x_all[int(idex_alive[i]), :, :]
# ##############################################################################
#     for i in range(int(Num_of_all * alpha_alive)):
#         for j in range(num_of_son):
#             for k in range(10):
#                 x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 0] = int(random.randint(0, 2))
#                 x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] = x_all[i, k, 1] + 0.01 * random.uniform(-derta_Q, derta_Q)
#                 if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] > up_Q:
#                     x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] = up_Q
#                 if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] < down_Q:
#                     x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 1] = down_Q
#                 x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] = x_all[i, k, 2] + 0.01 * random.uniform(-derta_F, derta_F)
#                 if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] > up_F:
#                     x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] = up_F
#                 if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] < down_F:
#                     x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 2] = down_F
#                 x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] = x_all[i, k, 3] + 0.01 * random.uniform(-derta_G, derta_G)
#                 if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] > up_G:
#                     x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] = up_G
#                 if x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] < down_G:
#                     x_all[int(Num_of_all * alpha_alive) + i * num_of_son + j, k, 3] = down_G
# #########################################################################################3
#     for i in range(int(Num_of_all * alpha_new)):
#         for j in range(10):
#             x_all[Num_of_all - i - 1, j, 0] = int(random.randint(0, 2))
#             x_all[Num_of_all - i - 1, j, 1] = random.uniform(down_Q, up_Q)
#             x_all[Num_of_all - i - 1, j, 2] = random.uniform(down_F, up_F)
#             x_all[Num_of_all - i - 1, j, 3] = random.uniform(down_G, up_G)
#
#
#
#     a=1





a=1
