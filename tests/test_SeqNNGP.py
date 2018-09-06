import numpy as np
import pyNNGP


def make_nnIndx(coords, m):
    nnIndx = []
    nnDist = []
    for i, coord in enumerate(coords):
        if i == 0:
            continue
        # Find the m nearest neighbors out of first i-1 points
        dists = np.sqrt(np.sum((coords[:i] - coord)**2, axis=1))
        s = np.argsort(dists)
        neighbors = s[:min(i, m)]
        nnIndx.extend(list(neighbors))
        nnDist.extend(list(dists[neighbors]))
    return np.array(nnIndx), np.array(nnDist)


def make_uIndx(n, m, nnIndx):
    uIndx = []
    uIndxU = []
    uiIndx = []
    for i in range(n):
        isANeighbor = []
        which = []
        for j in range(i+1, n):
            # Look through the jth part of nnIndx for i
            if j < m:
                nnStart = j*(j-1)//2
                nnEnd = j*(j+1)//2
            else:
                nnStart = m*(m-1)//2 + m*(j-m)
                nnEnd = m*(m-1)//2 + m*(j+1-m)
            try:
                k = list(nnIndx[nnStart:nnEnd]).index(i)
            except:
                continue
            else:
                isANeighbor.append(j)
                which.append(k)
        uIndx.extend(list(np.sort(isANeighbor)))
        uIndxU.append(len(isANeighbor))
        uiIndx.extend(which)
    uIndxL = [0]+list(np.cumsum(uIndxU)[:-1])
    return np.array(uIndx), np.hstack([uIndxL, uIndxU]), np.array(uiIndx)


def test_indices():
    np.random.seed(57)
    n = 50
    m = 10

    y = np.random.normal(size=n)
    X = np.random.normal(size=(n, 1))
    coords = np.random.uniform(size=(n, 2))

    phi = 6.0
    phiA, phiB = 3.0, 3./0.01
    phiTuning = 0.5
    sigmaSq = 5.0
    sigmaSqIGa, sigmaSqIGb = 2.0, 5.0

    covModel = pyNNGP.Exponential(sigmaSq, phi, phiA, phiB, phiTuning, sigmaSqIGa, sigmaSqIGb)
    tausqr = 1.0
    snngp = pyNNGP.SeqNNGP(y, X, coords, m, covModel, tausqr)
    nnIndx, nnDist = make_nnIndx(coords, m)
    np.testing.assert_array_equal(snngp._SeqNNGP.nnIndx, nnIndx)
    np.testing.assert_array_equal(snngp._SeqNNGP.nnDist, nnDist)
    uIndx, uIndxLU, uiIndx = make_uIndx(n, m, nnIndx)
    np.testing.assert_array_equal(snngp._SeqNNGP.uIndx, uIndx)
    np.testing.assert_array_equal(snngp._SeqNNGP.uIndxLU, uIndxLU)
    np.testing.assert_array_equal(snngp._SeqNNGP.uiIndx, uiIndx)


if __name__ == '__main__':
    test_indices()
