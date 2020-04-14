import numpy as np
import cv2
from weightedGE import weightedGE, im2double
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy.matlib
from scipy.spatial import ConvexHull
from scipy.optimize import fmin_bfgs

np.set_printoptions(precision=15)


def myfunoptimal(x):
    global pi, pt, Vt
    T = np.array(
        [[x[1] * np.cos(x[0]), -x[1] * np.sin(x[0]), 0], [x[2] * np.sin(x[0]), x[2] * np.cos(x[0]), 0], [0, 0, 1]])
    Io = pi.dot(T.T)
    hull = ConvexHull(Io)
    Vo = hull.volume
    hull_2 = ConvexHull(np.vstack((Io, pt)))
    Vtotal = hull_2.volume
    f = (Vtotal - Vt) + (Vtotal - Vo)
    return f


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= 1.0 * s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= 1.0 * t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def rotate_to_Zaxis(I, a):
    b = np.sqrt(a[0] ** 2 + a[1] ** 2)
    Txz = np.array([[a[0] / (b * 1.0), a[1] / (b * 1.0), 0], [-a[1] / (b * 1.0), a[0] / (b * 1.0), 0], [0, 0, 1.0]])
    c = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    Tz = np.array([[a[2] / (c * 1.0), 0, -b / (c * 1.0)], [0, 1.0, 0], [b / (c * 1.0), 0, a[2] / (c * 1.0)]])
    T = Tz.dot(Txz)
    I = T.dot(I)
    return I


def rotate_back_from_Zaxis(I, a):
    b = np.sqrt(a[0] ** 2 + a[1] ** 2)
    iTxz = np.array([[a[0] / (b * 1.0), -a[1] / (b * 1.0), 0], [a[1] / (b * 1.0), a[0] / (b * 1.0), 0], [0, 0, 1.0]])
    c = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    iTz = np.array([[a[2] / (c * 1.0), 0, b / (c * 1.0)], [0, 1.0, 0], [-b / (c * 1.0), 0, a[2] / (c * 1.0)]])
    iT = iTxz.dot(iTz)
    I = iT.dot(I)
    return I


def createDXForward(N, M):
    K = N * M
    a = np.array([-np.ones((K, 1)), -2 * np.ones((K, 1)), -np.ones((K, 1)), np.ones((K, 1)), 2 * np.ones((K, 1)),
                  np.ones((K, 1))]).reshape(6, -1)
    Dx = sparse.spdiags(a, np.array([-N, -N + 1, -N + 2, N, N + 1, N + 2]), K, K)
    Dx = Dx.tolil()
    Dx[: N, :] = 0
    Dx[K - N:, :] = 0
    return Dx


def createDYForward(N, M):
    K = N * M
    a = np.array([-np.ones((K, 1)), np.ones((K, 1)), -2 * np.ones((K, 1)), 2 * np.ones((K, 1)), -np.ones((K, 1)),
                  np.ones((K, 1))]).reshape(6, -1)
    Dy = sparse.spdiags(a, np.array([-N, -N + 2, -1, 1, N, N + 2]), K, K)
    Dy = Dy.tolil()
    Dy[: N, :] = 0
    Dy[K - N:, :] = 0
    return Dy


def normalizeIntensity(Is, It, ws, hs):
    lambda_ = 1
    DX = createDXForward(ws, hs)
    DX = DX.T.dot(DX)
    DY = createDYForward(ws, hs)
    DY = DY.T.dot(DY)
    D = lambda_ * (DX + DY)
    A = sparse.eye(ws * hs) + D
    A = A.tolil()
    t = np.sqrt(3)
    Is = Is / (t * 1.0)
    It = It / (t * 1.0)
    If = hist_match(Is, It)

    b = If + D.dot(Is)
    A = A.tocsr()
    Io = spsolve(A, b)
    Io = t * Io
    return Io


def color_transfer(Is, It):
    global pi, pt, Vt
    H, W, _ = Is.shape
    Ht, Wt, _ = It.shape
    mink_norm = 5
    sigma = 2
    kappa = 10
    # step 1:White-balancing and rotating
    # Grey-Egde algorithm to estimate illuminations of the source and target
    print('calculating the white balance')
    [wRs, wGs, wBs] = weightedGE(Is, kappa, mink_norm, sigma)
    WBs = np.array([wRs, wGs, wBs])
    [wRt, wGt, wBt] = weightedGE(It, kappa, mink_norm, sigma)
    WBt = np.array([wRt, wGt, wBt])

    WBs = np.sqrt(3) * WBs / (np.sqrt(np.sum(WBs ** 2)) * 1.0)
    WBt = np.sqrt(3) * WBt / (np.sqrt(np.sum(WBt ** 2)) * 1.0)
    Is = Is.reshape(-1, 3, order='F').T
    It = It.reshape(-1, 3, order='F').T
    Is = np.diag(1.0 / WBs).dot(Is)  # pass
    It = np.diag(1.0 / WBt).dot(It)  # pass
    Is = rotate_to_Zaxis(Is, np.array([1, 1, 1]))  # pass
    It = rotate_to_Zaxis(It, np.array([1, 1, 1]))  # pass

    # Step 2: Luminance Matchingnce
    print('matching the Luminance')
    Is = Is.T
    It = It.T
    Is[:, 2] = normalizeIntensity(Is[:, 2], It[:, 2], H, W)

    # Step 3: Color Gamut Aligning
    print('Color Gamut Aligning')
    Ms = np.mean(Is, 0)
    Mt = np.mean(It, 0)
    Is = Is - np.matlib.repmat(Ms, H * W, 1)
    It = It - np.matlib.repmat(Mt, Ht * Wt, 1)
    hull_s = ConvexHull(Is)
    Chi = hull_s.simplices
    hull_t = ConvexHull(It)
    Cht = hull_t.simplices
    Vt = hull_t.volume
    idi = np.unique(Chi[:])
    idt = np.unique(Cht[:])
    pi = Is[idi - 1, :]
    pt = It[idt - 1, :]
    # compute the optimal     matrix
    x0 = np.array([0, 1, 1])
    x = fmin_bfgs(myfunoptimal, x0, maxiter=50, disp=1)
    T = np.array(
        [[x[1] * np.cos(x[0]), -x[1] * np.sin(x[0]), 0], [x[2] * np.sin(x[0]), x[2] * np.cos(x[0]), 0], [0, 0, 1]])
    # Align two gamuts
    Io = T.dot(Is.T)
    Mt[2] = Ms[2]
    Io = Io + np.matlib.repmat(Mt.reshape(1, 3).T, 1, H * W)

    # STEP 4: Rotate back and undo white-balancing
    print('Rotate back and undo white-balancing')
    Io = rotate_back_from_Zaxis(Io, np.array([1, 1, 1]))
    Io = np.diag(WBt).dot(Io)
    Io[Io < 0] = 0
    Io[Io > 1] = 1
    Io = Io.T
    Io_ = np.reshape(Io, (H, W, 3), order='F')
    Io_2 = np.asarray(Io_ * 255, dtype=np.uint8)
    Io_2 = cv2.cvtColor(Io_2, cv2.COLOR_RGB2BGR)

    return Io_2
