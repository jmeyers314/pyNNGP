import numpy as np
from past.builtins import basestring
from sklearn.neighbors import KNeighborsRegressor, KDTree

class NNGP(object):
    def __init__(self, t, y, eps, refType, m, cov):
        self.t = t  # ordinates
        self.y = y  # abscissae
        self.eps = eps  # measurement uncertainties in y
        self.refType = refType # type of reference set to construct.
        self.m = m  # number of reference neighbors to associate with each point t.
        self.cov = cov # covariance function of parent GP.

        self._init_s()
        self._init_wt()
        self._init_ws()
        self._make_s_neighbor_sets()
        self._make_t_neighbor_sets()


    def _init_s(self):
        # Options for refType are:
        # 1) string with 'S=T', sets reference set equal to input ordinates
        # 2) tuple with ('subset', nRef):
        #    Chooses a random subset of T of size nRef as S
        # 3) tuple with ('random', nRef, bounds)
        #    Chooses nRef uniformly distributed points within hyper-rectangle defined by bounds
        #    Bounds should be a tuple of 2-tuples, indicating bounds in each dimension.
        if isinstance(self.refType, basestring):
            if self.refType == 'S=T':
                self.s = self.t
        elif isinstance(self.refType, tuple):
            typ = self.refType[0]
            if self.typ == 'subset':
                nRef = self.refType[1]
                choice = np.random.choice(len(self.t), size=nRef)
                self.s = self.t[choice]
            elif typ == 'random':
                nRef, bounds = self.refType[1], self.refType[2]
                self.s = np.vstack([np.random.uniform(lo, hi, nRef) for lo, hi in bounds]).T

    def _init_wt(self):
        self.wt = np.copy(self.y)

    def _init_ws(self):
        knn = KNeighborsRegressor(n_neighbors=5, weights='uniform')
        self.ws = knn.fit(self.t, self.y).predict(self.s)

    def _make_s_neighbor_sets(self):
        self.Ns = []
        for i, si in enumerate(self.s):
            if i == 0:
                self.Ns.append([])
                continue
            kdtree = KDTree(self.s[0:i])
            self.Ns.append(
                kdtree.query(
                    si.reshape(1, -1),
                    k=min(self.m, i),
                    return_distance=False
                )[0]
            )

    def _make_t_neighbor_sets(self):
        if self.refType == 'S=T':
            self.Nt = self.Ns
            return
        self.Nt = []
        kdtree = KDTree(self.s)
        for i, ti in enumerate(self.t):
            self.Nt.append(kdtree.query(ti.reshape(1, -1), self.m))

    def _Bsi(self, i):
        """B_{s_i}
        """
        ...

    def _CNs(self, i):
        """ C_{N(s_i)}
        """
        Nsi = self.Ns[i]
        return self.cov(Nsi, Nsi)

    def _Ccross(i):
        """ C_{s_i, N(s_i)}
        """

    def _Fsi(self, i):
        """ F_{s_i}
        """

    def _Cs(self, i):
        """ C_{si, si}
        """
        si = self.s[i]
        return self.cov(si, si)

    def oneSample(self):
        self.update_wt()
        self.update_ws()
        self.update_y_unobserved()
