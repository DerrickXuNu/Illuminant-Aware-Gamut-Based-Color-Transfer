import numpy as np
import cv2
from fill_border import fill_border
import math
from scipy import signal

np.set_printoptions(precision=15)


def gDer(f, sigma, iorder, jorder):
    break_off_sigma = 3.0
    filtersize = math.floor(break_off_sigma * sigma + 0.5)
    f = fill_border(f, filtersize)
    x = np.arange(-filtersize, filtersize + 1)
    Gauss = 1 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp((x ** 2) / (-2.0 * sigma * sigma))
    if iorder == 0:
        Gx = Gauss / (1.0 * np.sum(Gauss))
    elif iorder == 1:
        Gx = -(x / (1.0 * sigma ** 2)) * Gauss
        Gx = Gx / (1.0 * np.sum(np.sum(x * Gx)))
    elif iorder == 2:
        Gx = (x ** 2 / sigma ** 4.0 - 1 / sigma ** 2.0) * Gauss
        Gx = Gx - np.sum(Gx) / (1.0 * x.shape[0])
        Gx = Gx / (1.0 * np.sum(0.5 * x * x * Gx))
    Gx = Gx.reshape(1, 13)
    H = -signal.convolve2d(f, Gx, mode='same')

    if jorder == 0:
        Gy = Gauss / (1.0 * np.sum(Gauss))
    elif jorder == 1:
        Gy = -(x / (1.0 * sigma ** 2)) * Gauss
        Gy = Gy / (1.0 * (np.sum(np.sum(x * Gy))))
    elif jorder == 2:
        Gy = (x ** 2 / sigma ** 4.0 - 1 / sigma ** 2.0) * Gauss
        Gy = Gy - np.sum(Gy) / (1.0 * x.shape[0])
        Gy = Gy / (1.0 * np.sum(0.5 * x * x * Gy))
    Gy = Gy.reshape(1, 13)
    H = signal.convolve2d(H, Gy.conj().T, mode='same')
    filtersize = int(filtersize)
    H = H[filtersize:H.shape[0] - filtersize, filtersize: H.shape[1] - filtersize]
    return H
