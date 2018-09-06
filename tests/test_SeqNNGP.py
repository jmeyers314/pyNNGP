import numpy as np
import pyNNGP

def test_SeqNNGP():
    np.random.seed(5)
    size = 10000
    m = 15
    y = np.random.normal(size=size)
    X = np.random.normal(size=size)
    coords = np.random.uniform(size=(size, 2))

    phi = 6.0
    phiA, phiB = 3.0, 3./0.01
    phiTuning = 0.5
    sigmaSq = 5.0
    sigmaSqIGa, sigmaSqIGb = 2.0, 5.0
    covModel = pyNNGP.Exponential(sigmaSq, phi, phiA, phiB, phiTuning, sigmaSqIGa, sigmaSqIGb)

    tausqr = 1.0
    snngp = pyNNGP.SeqNNGP(y, X, coords, m, covModel, tausqr)
    snngp.sample(10)


def test_nnIndx():
    np.random.seed(5)
    size = 5
    m = 3
    # y = np.random.normal(size=size)
    # X = np.random.normal(size=size)
    # coords = np.random.uniform(size=(size, 2))

    X = np.array([[1,1,1,1,1], [-0.3053884, -0.8204684, 0.4874291, 0.7383247, 0.5757814]])
    y = np.array([2.5739998, 0.2331699, 5.8600358, 4.6630348, -0.3214800])

    coords = np.array([
        [0.2016819, 0.06178627],
        [0.2655087, 0.89838968],
        [0.3721239, 0.94467527],
        [0.5728534, 0.66079779],
        [0.9082078, 0.62911404]
    ])

    phi = 6.0
    phiA, phiB = 3.0, 3./0.01
    phiTuning = 0.5
    sigmaSq = 5.0
    sigmaSqIGa, sigmaSqIGb = 2.0, 5.0

    covModel = pyNNGP.Exponential(sigmaSq, phi, phiA, phiB, phiTuning, sigmaSqIGa, sigmaSqIGb)
    tausqr = 1.0
    snngp = pyNNGP.SeqNNGP(y, X, coords, m, covModel, tausqr)

    print(coords)
    for i, coord in enumerate(coords):
        dists = np.sum((coords-coord)**2, axis=1)[:i]
        closest = np.argsort(dists)
        if i < m:
            print(i, closest)
        else:
            print(i, closest[:m])

    print()
    print(snngp._SeqNNGP.nnIndx)
    print(snngp._SeqNNGP.nnIndxLU)
    print(snngp._SeqNNGP.nnDist)
    print(snngp._SeqNNGP.uIndx)
    print(snngp._SeqNNGP.uIndxLU)
    print(snngp._SeqNNGP.uiIndx)
    print(snngp._SeqNNGP.CIndx)


if __name__ == '__main__':
    test_SeqNNGP()
    test_nnIndx()
