import numpy as np
import cv2
import math
from scipy import signal

np.set_printoptions(precision=15)


def set_board(in_, width, method=1):
    temp = np.ones(in_.shape)
    y, x = np.mgrid[1:(in_.shape[0] + 1), 1:(in_.shape[1] + 1)]
    temp = temp * ((x < temp.shape[1] - width + 1) & (x > width))
    temp = temp * ((y < temp.shape[0] - width + 1) & (y > width))
    out = temp * in_
    if method == 1:
        out = out + (np.sum(out[:]) / (1.0 * np.sum(temp[:]))) * (np.ones(np.shape(in_)) - temp)
    return out
