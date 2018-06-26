import pyNNGP
import numpy as np

def test_init():
    # Let's try a scalar response to a bivariate ordinate
    n = 200
    tx = np.random.uniform(size=n)
    ty = np.random.uniform(size=n)
    t = np.vstack([tx, ty]).T

    y = np.zeros_like(t)
    eps = np.ones_like(t)*0.001

    refType = 'S=T'

    m = 3

    cov = None

    nngp = pyNNGP.NNGP(t, y, eps, refType, m, cov)

    for i in range(n):
        assert i not in nngp.Ns[i]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i, si in enumerate(nngp.s):
        ax.scatter(*si, c='k')
        for sj in nngp.s[nngp.Ns[i]]:
            ax.arrow(*si, *(sj-si), color='b', length_includes_head=True, alpha=0.2)
    plt.show()

if __name__ == '__main__':
    test_init()
