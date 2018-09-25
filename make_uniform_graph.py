import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams["figure.figsize"] = (7, 7)

n = 100 # size of sample
m = 100 # number of samples on each step
maxk = 50 # max k in x^k

theta = 1

k_axis = [i for i in range(1, maxk + 1)]
d_axis = [0] * maxk

for k in range(1, maxk + 1):
    for j in range(0, m):
        sample = np.random.uniform(0, theta, n)
        mean = sum(map(lambda x: x**k, sample)) / n
        delta = (theta - ((k+1) * mean)**(1/k))**2
        #print(mean, delta)
        d_axis[k - 1] += delta
    d_axis[k - 1] /= m

plt.cla()
plt.plot(k_axis, d_axis, linestyle='None', marker='o', color="black")
plt.xlim(-0.001, maxk + 1)
plt.ylim(0, 0.005)
plt.xlabel("k")
plt.ylabel("MSE")
plt.savefig("uniform.png")
