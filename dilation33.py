import numpy as np

np.set_printoptions(precision=15)


def dilation33(in_, it=1):
    in_ = np.asarray(in_)
    hh = in_.shape[0]
    ll = in_.shape[1]
    out = np.zeros((hh, ll, 3))

    while it > 0:
        it = it - 1
        out[:hh - 1, :, 0] = in_[1:hh, :]
        out[hh - 1, :, 0] = in_[hh - 1, :]
        out[:, :, 1] = in_
        out[0, :, 2] = in_[0, :]
        out[1:, :, 2] = in_[0:hh - 1, :]
        out2 = out.max(2)
        out[:, :ll - 1, 0] = out2[:, 1:ll]
        out[:, ll - 1, 0] = out2[:, ll - 1]
        out[:, :, 1] = out2
        out[:, 0, 2] = out2[:, 0]
        out[:, 1:, 2] = out2[:, :ll - 1]
        out = out.max(2)
        in_ = out
    return out
