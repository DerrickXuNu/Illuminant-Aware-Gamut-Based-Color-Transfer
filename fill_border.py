import numpy as np
import cv2

np.set_printoptions(precision=15)


def fill_border(in_, bw):
    hh = in_.shape[0]
    ww = in_.shape[1]
    dd = 1 if len(in_.shape) == 2 else in_.shape[2]
    bw = int(bw)
    if dd == 1:
        out = np.zeros((hh + bw * 2, ww + bw * 2))
        out[:bw, :bw] = np.ones((bw, bw)) * in_[0, 0]
        out[bw + hh:2 * bw + hh, :bw] = np.ones((bw, bw)) * in_[hh - 1, 0]
        out[: bw, bw + ww: 2 * bw + ww] = np.ones((bw, bw)) * in_[0, ww - 1]
        out[bw + hh:2 * bw + hh, bw + ww: 2 * bw + ww] = np.ones((bw, bw)) * in_[hh - 1, ww - 1]
        out[bw: bw + hh, bw: bw + ww] = in_
        out[: bw, bw: bw + ww] = np.ones((bw, 1)).dot(in_[0, :].reshape(1, -1))
        out[bw + hh: 2 * bw + hh, bw: bw + ww] = np.ones((bw, 1)).dot(in_[hh - 1, :].reshape(1, -1))
        out[bw: bw + hh, : bw] = in_[:, 0].reshape(-1, 1).dot(np.ones((1, bw)))
        out[bw: bw + hh, bw + ww: 2 * bw + ww] = in_[:, ww - 1].reshape(-1, 1).dot(np.ones((1, bw)))
    else:
        out = np.zeros((hh + bw * 2, ww + bw * 2, dd))
        for ii in range(dd):
            out[:bw, :bw, ii] = np.ones((bw, bw)) * in_[0, 0, ii]
            out[bw + hh: 2 * bw + hh, : bw, ii] = np.ones((bw, bw)) * in_[hh - 1, 0, ii]
            out[: bw, bw + ww: 2 * bw + ww, ii] = np.ones((bw, bw)) * in_[0, ww - 1, ii]
            out[bw + hh: 2 * bw + hh, bw + ww: 2 * bw + ww, ii] = np.ones((bw, bw)) * in_[hh - 1, ww - 1, ii]
            out[bw: bw + hh, bw: bw + ww, ii] = in_[:, :, ii]
            out[: bw, bw: bw + ww, ii] = np.ones((bw, 1)).dot(in_[0, :, ii].reshape(1, -1))
            out[bw + hh: 2 * bw + hh, bw: bw + ww, ii] = np.ones((bw, 1)).dot(in_[hh - 1, :, ii].reshape(1, -1))
            out[bw: bw + hh, : bw, ii] = in_[:, 0, ii].reshape(-1, 1).dot(np.ones((1, bw)))
            out[bw: bw + hh, bw + ww: 2 * bw + ww, ii] = in_[:, ww - 1, ii].reshape(-1, 1).dot(np.ones((1, bw)))
    return out
