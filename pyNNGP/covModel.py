class CovModel:
    def cov(self, dist):
        raise NotImplementedError

class Exponential(CovModel):
    def __init__(self, sigmaSq, sigmaSqPrior, phi, phiPrior):
        ...
    def cov(self):
        ...
