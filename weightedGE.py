import numpy as np
import math
from set_board import set_board
from dilation33 import dilation33
from compute_spvar import compute_spvar


np.set_printoptions(precision=15)


def im2double(im):
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    return im.astype(np.float) / (1.0 * info.max)


def weightedGE(input_im, kappa=1, mink_norm=1, sigma=1):
    iter = 10
    mask_cal = np.zeros((input_im.shape[0], input_im.shape[1]))
    tmp_ill = np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])
    final_ill = tmp_ill.copy()
    tmp_image = input_im.copy()
    flag = 1
    while iter > 0 and flag == 1:
        iter -= 1
        tmp_image[:, :, 0] = tmp_image[:, :, 0] / (np.sqrt(3) * (tmp_ill[0]))
        tmp_image[:, :, 1] = tmp_image[:, :, 1] / (np.sqrt(3) * (tmp_ill[1]))
        tmp_image[:, :, 2] = tmp_image[:, :, 2] / (np.sqrt(3) * (tmp_ill[2]))
        sp_var, Rw, Gw, Bw = compute_spvar(tmp_image, sigma)
        mask_zeros = np.maximum(Rw, np.maximum(Gw, Bw)) < np.finfo(float).eps
        mask_pixels = dilation33(((tmp_image.max(2)) == 255))
        mask = set_board((np.logical_or(np.logical_or(mask_pixels, mask_zeros), mask_cal) == 0).astype(float),
                         sigma + 1, 0)
        mask[mask != 0] = 1
        grad_im = np.sqrt(Rw ** 2 + Gw ** 2 + Bw ** 2)
        weight_map = (sp_var / (1.0 * grad_im)) ** kappa
        weight_map[weight_map > 1] = 1
        data_Rx = np.power(Rw * weight_map, mink_norm)
        data_Gx = np.power(Gw * weight_map, mink_norm)
        data_Bx = np.power(Bw * weight_map, mink_norm)

        tmp_ill[0] = np.power(np.sum(data_Rx * mask), 1 / (1.0 * mink_norm))
        tmp_ill[1] = np.power(np.sum(data_Gx * mask), 1 / (1.0 * mink_norm))
        tmp_ill[2] = np.power(np.sum(data_Bx * mask), 1 / (1.0 * mink_norm))

        tmp_ill = tmp_ill / (1.0 * np.linalg.norm(tmp_ill))
        final_ill = final_ill * tmp_ill
        final_ill = final_ill / (1.0 * np.linalg.norm(final_ill))
        if np.arccos(tmp_ill.dot(1 / math.sqrt(3) * np.array([1, 1, 1]).T)) / np.pi * 180 < 0.05:
            flag = 0
    white_R = final_ill[0]
    white_G = final_ill[1]
    white_B = final_ill[2]
    output_im = np.zeros((input_im.shape[0], input_im.shape[1], input_im.shape[2]))
    output_im[:, :, 0] = input_im[:, :, 0] / (np.sqrt(3) * (final_ill[0]))
    output_im[:, :, 1] = input_im[:, :, 1] / (np.sqrt(3) * (final_ill[1]))
    output_im[:, :, 2] = input_im[:, :, 2] / (np.sqrt(3) * (final_ill[2]))

    return white_R, white_G, white_B

