import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as mpatches
from auxilary import Hy_step, E_step, mod_helmholtz, Hx_step, create_Ds
import multiprocessing as mp
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import ray
from scipy.sparse import csc_matrix
import timeit


# ray.init()
# @ray.remote
# def fHx(dt, x, y, h, E0, Hx0, BHx):
#      return Hx_step(dt, x, y, h, E0, Hx0, BHx)
#
# @ray.remote
# def fHy(dt, x, y, h, E0, Hy0, BHy):
#      return Hy_step(dt, x, y, h, E0, Hy0, BHy)


def max_solver(dt, x, y, h, time_steps, E_a, Hx_a, Hy_a, DxE, DyE, DxHx, DyHx, DxHy, DyHy, AE, AHx, AHy):
    E_tot = []
    Hx_tot = []
    Hy_tot = []
    E0 = E_a[0]
    Hx0 = Hx_a[0]
    Hy0 = Hy_a[0]
    for t in range(time_steps - 1):
        print(t)
        E_tot.append(E0)
        Hx_tot.append(Hx0)
        Hy_tot.append(Hy0)
        start = timeit.default_timer()
        E0 = E_step(dt, x, y, h, E0.copy(), Hx0.copy(), Hy0.copy(), DxE, DyE, AE)


        # ret_id1 = fHx.remote(dt, x, y, h, E0.copy(), Hx0.copy(), (Hx_a[t+1]-Hx_a[t])/dt)
        # ret_id2 = fHy.remote(dt, x, y, h, E0.copy(), Hy0.copy(), (Hy_a[t+1]-Hy_a[t])/dt)
        # Hx0, Hy0 = ray.get([ret_id1, ret_id2])
        Hx0 = Hx_step(dt, x, y, h, E0.copy(), Hx0.copy(), (Hx_a[t + 1] - Hx_a[t]) / dt, DxHx, DyHx, AHx)
        Hy0 = Hy_step(dt, x, y, h, E0.copy(), Hy0.copy(), (Hy_a[t + 1] - Hy_a[t]) / dt, DxHy, DyHy, AHy)
        stop = timeit.default_timer()

    errE = np.array([np.mean(abs(E_tot[i][1:-1, 1:-1] - E_a[i][1:-1, 1:-1])) for i in range(len(E_tot))])
    errHx = np.array([np.mean(abs(Hx_tot[i][1:-1, :] - Hx_a[i][1:-1, :])) for i in range(len(Hx_tot))])
    errHy = np.array([np.mean(abs(Hy_tot[i][:, 1:-1] - Hy_a[i][:, 1:-1])) for i in range(len(Hy_tot))])
    return ((np.mean(errE) + np.mean(errHx) + np.mean(errHy)) / 3)


kx = 4
ky = 3
omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)

err = []
ns = [64,84, 128]
# N=[21,41,81]

for i, n in enumerate(ns):
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    h = x[1] - x[0]
    DxE, DyE = create_Ds(x, y)
    DxHx, DyHx = create_Ds(x, y[1:])
    DxHy, DyHy = create_Ds(x[1:], y)
    start= timeit.default_timer()
    AE = DxE + DyE + ((h ** 2) / 6) * DxE@DyE
    # stop = timeit.default_timer()
    # print(stop-start)
    # print(q)
    AHx = DxHx + DyHx + ((h ** 2) / 6) * DxHx@DyHx
    AHy = DxHy + DyHy + ((h ** 2) / 6) * DxHy@DyHy
    # AE=DxE + DyE + ((h ** 2) / 6) * np.matmul(DxE, DyE)
    # AHx = DxHx + DyHx + ((h ** 2) / 6) * np.matmul(DxHx, DyHx)
    #
    # AHy = DxHy + DyHy + ((h ** 2) / 6) * np.matmul(DxHy, DyHy)







    time_steps = 2 * ns[i] + 1
    T = 1
    dt = T / (time_steps - 1)

    t, X, Y = np.meshgrid(np.linspace(0, T, time_steps), x, y, indexing='ij')
    E_a = np.cos(omega * t) * np.sin(math.pi * kx * X) * np.sin(math.pi * ky * Y)
    # BF=-(omega**2+2)*E_a[0]
    # F=BF[1:-1,1:-1].flatten()
    Hx_a = -(np.sin(omega * (t + dt / 2)) * np.sin(math.pi * kx * X) * np.cos(
        math.pi * ky * (Y + h / 2)) * math.pi * ky / omega)[:, :, :-1]
    Hy_a = (np.sin(omega * (t + dt / 2)) * np.cos(math.pi * kx * (X + h / 2)) * np.sin(
        math.pi * ky * Y) * math.pi * kx / omega)[:, :-1, :]
    # err.append(np.mean(abs(Hy_step(dt, x, y, h, E_a[1], Hy_a[0])[:,1:-1]-Hy_a[1][:,1:-1])))
    # err.append(np.mean(abs(Hx_step(dt, x, y, h, E_a[1], Hx_a[0])[1:-1, :] - Hx_a[1][1:-1,:])))
    err.append(max_solver(dt, x, y, h, time_steps, E_a, Hx_a, Hy_a, DxE, DyE,
                          DxHx, DyHx, DxHy, DyHy,
                          AE, AHx, AHy))


    # err.append(max_solver(dt, x, y, h, time_steps, E_a, Hx_a, Hy_a, DxE.toarray(), DyE.toarray(),
    #                       DxHx.toarray(), DyHx.toarray(), DxHy.toarray(), DyHy.toarray(),
    #                       AE.toarray(), AHx.toarray(), AHy.toarray()))
    # err.append(np.mean(abs(E_step(dt, x, y, h, E_a[0], Hx_a[0], Hy_a[0]) - E_a[1])))
    # err.append(np.mean(abs(mod_helmholtz(x, y,BF*0,F,BF ,2) - E_a[0])))

#
x = np.log(1 / np.array(ns))
y = np.log(np.array(err))
plt.plot(ns, err)
print(np.diff(y) / np.diff(x))
#
# x=np.log(1/np.array(nT))
# y=np.log(np.array(err))
# plt.plot(nT,err)
# print(np.diff(y) / np.diff(x))

# x=np.log(np.array(Ts))
# y=np.log(np.array(err))
# plt.plot(1/np.array(Ts),err)
# print(np.diff(y) / np.diff(x))

plt.show()
