from . import _pyNNGP
import numpy as np

class SeqNNGP:
    def __init__(self, y, X, coords, nNeighbors, covModel, tausqr):
        self.X = np.ascontiguousarray(np.atleast_2d(X))
        self.y = np.ascontiguousarray(np.atleast_1d(y))
        # Sort by coords[:, 0] first?
        self.coords = np.ascontiguousarray(np.atleast_2d(coords))
        self.nNeighbors = nNeighbors
        self.covModel = covModel
        self.tausqr = tausqr

        self._SeqNNGP = _pyNNGP.SeqNNGP(
            self.y.ctypes.data, self.X.ctypes.data, self.coords.ctypes.data,
            # self.X.shape[0], self.X.shape[1], self.nNeighbors,
            self.X.shape[1], self.X.shape[0], self.nNeighbors,
            self.covModel, self.tausqr
        )

    def sample(self, N):
        self._SeqNNGP.sample(N)

    def updateW(self):
        self._SeqNNGP.updateW()

    def updateBeta(self):
        self._SeqNNGP.updateBeta()

    def updateTauSq(self):
        self._SeqNNGP.updateTauSq()

    @property
    def w(self):
        return self._SeqNNGP.w

    @property
    def beta(self):
        return self._SeqNNGP.beta

    @property
    def tauSq(self):
        return self._SeqNNGP.tauSq

    @property
    def B(self):
        return self._SeqNNGP.B

    @property
    def F(self):
        return self._SeqNNGP.F
