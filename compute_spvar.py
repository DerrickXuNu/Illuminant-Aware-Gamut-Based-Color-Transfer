import numpy as np
import cv2
from gDer import gDer
import math
from scipy import signal

np.set_printoptions(precision=15)


def compute_spvar(im, sigma):
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    Rx = gDer(R, sigma, 1, 0)
    Ry = gDer(R, sigma, 0, 1)
    Rw = np.sqrt(Rx ** 2 + Ry ** 2)

    Gx = gDer(G, sigma, 1, 0)
    Gy = gDer(G, sigma, 0, 1)
    Gw = np.sqrt(Gx ** 2 + Gy ** 2)

    Bx = gDer(B, sigma, 1, 0)
    By = gDer(B, sigma, 0, 1)
    Bw = np.sqrt(Bx ** 2 + By ** 2)

    # Opponent_der
    O3_x = (Rx + Gx + Bx) / np.sqrt(3)
    O3_y = (Ry + Gy + By) / np.sqrt(3)
    sp_var = np.sqrt(O3_x ** 2 + O3_y ** 2)

    return sp_var, Rw, Gw, Bw
