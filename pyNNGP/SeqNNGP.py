from . import _pyNNGP
import numpy as np

class SeqNNGP:
    def __init__(self, y, X, coords, nNeighbors, covModel, tausqr):
        self.X = np.ascontiguousarray(np.atleast_2d(X))
        self.y = np.ascontiguousarray(np.atleast_1d(y))
        # Sort by coords[:, 0] first?
        self.coords = np.ascontiguousarray(np.atleast_1d(coords))
        self.nNeighbors = nNeighbors
        self.covModel = covModel
        self.tausqr = tausqr

        self._SeqNNGP = _pyNNGP.SeqNNGP(
            self.y.ctypes.data, self.X.ctypes.data, self.coords.ctypes.data,
            self.X.shape[0], self.X.shape[1], self.nNeighbors,
            self.covModel, self.tausqr
        )

    def sample(self, N):
        self._SeqNNGP.sample(N)
