import numpy as np
from matplotlib import pyplot as plt


def vasicek(r0, alpha, mu, sigma, step_size):
    w = np.random.normal(size=40)
    res = [r0]
    for i in range(0, 40):
        res.append(res[-1] + (alpha * (mu - res[-1]) + sigma * w[i]))
    return res

def lmm_fwd_rate(l0, sigma):
    w = np.random.normal(size=40)
    res = [l0]
    for i in range(0, 40):
        res.append(res[-1] + (sigma * res[-1] * w[i]))
    return res

def lmm2(l0, sigma):
    res2 = np.array(lmm_fwd_rate(l0, sigma))
    res2 += 1
    res2 = res2.cumprod()
    return np.power(res2, [1/x for x in range(1, 1+len(res2))]) - 1


if __name__ == '__main__':
    res = [0.04] * 41
    for i in range(1, 11):
        res = lmm2(res[1], 0.05)
        plt.plot(range(0, 41), res, label=i)
        plt.legend()
        plt.show()
