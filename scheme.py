import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

import matplotlib.patches as mpatches
from auxilary import Hy_step2, Hx_step2, create_Ds2, f_Hy, f_E, f_Hx, create_P1, create_P2, E_step2, create_lap
import multiprocessing as mp
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import ray
from scipy.sparse import csc_matrix
import timeit
import pickle
from decimal import Decimal
from DRP_multiple_networks.utils import amper, faraday
from DRP_multiple_networks.constants import Constants


def max_solver(fourth, omega, kx, ky, dt, x, y, h, time_steps, DxE, DyE, DxHx, DyHx, DxHy, DyHy, AE, AHx, AHy, usual):
    print(time_steps)
    errE = []
    errHx = []
    errHy = []
    energy = []

    E0 = f_E(omega, kx, ky, x, y, 0)
    LE0 = -math.pi**2*(kx**2+ky**2)*E0
    Hx0 = f_Hx(omega, kx, ky, x, y, 0, dt, h)
    LHx0=-math.pi**2*(kx**2+ky**2)*Hx0
    Hy0 = f_Hy(omega, kx, ky, x, y, 0, dt, h)
    LHy0=-math.pi**2*(kx**2+ky**2)*Hy0

    p1x = create_P1(x[1:-1], x[1:], h, dt, 'x')
    p2x = create_P2(x[1:-1], x[1:], h, dt, 'x')
    p1y = create_P1(x[1:], x[1:-1], h, dt, 'y')
    p2y = create_P2(x[1:], x[1:-1], h, dt, 'y')
    p1e = create_P1(x[1:-1], x[1:-1], h, dt, 'e')
    p2e = create_P2(x[1:-1], x[1:-1], h, dt, 'e')
    plap=create_lap(x[1:-1], x[1:-1])

    for i, t in enumerate(tqdm(range(time_steps - 1))):
        errE.append(
            np.mean(abs(E0[1:-1, 1:-1] - f_E(omega, kx, ky, x, y, i*dt)[1:-1, 1:-1])))
        errHx.append(
            np.mean(abs(Hx0[1:-1, :] - f_Hx(omega, kx, ky, x, y, i*dt, dt, h)[1:-1, :])))
        errHy.append(
            np.mean(abs(Hy0[:, 1:-1] - f_Hy(omega, kx, ky, x, y, i*dt, dt, h)[:, 1:-1])))


        LE0, E0 = E_step2(dt, x, y, h, E0.copy(), LE0.copy(),
                            Hx0.copy(), Hy0.copy(), p1e, p2e, plap, usual)
        Hx0 = Hx_step2(dt, x, y, h, E0.copy(),
                        LE0.copy(), Hx0.copy(), p1x, p2x)
        Hy0 = Hy_step2(dt, x, y, h, E0.copy(),
                        LE0.copy(), Hy0.copy(), p1y, p2y)


    errE = np.array(errE)
    errHx = np.array(errHx)
    errHy = np.array(errHy)
    # print(abs(errE))
    return ((np.mean(errE) + np.mean(errHx) + np.mean(errHy)) / 3, errE+errHx+errHy)


def run_scheme(C, usual):
    kx = C.kx
    ky = C.ky
    T = C.T
    cfl = C.cfl
    n = C.ns
    fourth = C.fourth_order
    ky = kx
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)

    h = x[1] - x[0]
    dt = h*cfl
    time_steps = int(T/dt)

    DxE, DyE = create_Ds2(x, y)

    DxHx, DyHx = create_Ds2(x, y[1:])
    DxHy, DyHy = create_Ds2(x[1:], y)
    # start = timeit.default_timer()
    AE = DxE + DyE + ((h ** 2) / 6) * DxE @ DyE

    AHx = DxHx + DyHx + ((h ** 2) / 6) * DxHx @ DyHx
    AHy = DxHy + DyHy + ((h ** 2) / 6) * DxHy @ DyHy

    res = max_solver(fourth, omega, kx, ky,  dt, x, y, h, time_steps, DxE, DyE,
                     DxHx, DyHx, DxHy, DyHy,
                     AE, AHx, AHy, usual)

    return res


def yee_solver(beta, delta, gamma, omega, kx, ky, dt, x, y, h, time_steps, cfl, n, t):
    C = Constants(n+1, 1, t, cfl, kx, ky)

    errE = []
    errHx = []
    errHy = []

    E = f_E(omega, kx, ky, x, y, 0)
    Hx = f_Hx(omega, kx, ky, x, y, 0, dt, h)[1:-1, :]
    Hy = f_Hy(omega, kx, ky, x, y, 0, dt, h)[:, 1:-1]

    E0 = np.expand_dims(E, axis=(0, -1))
    Hx0 = np.expand_dims(Hx, axis=(0, -1))
    Hy0 = np.expand_dims(Hy, axis=(0, -1))

    for i, t in enumerate(tqdm(range(time_steps - 1))):

        errE.append(np.mean(
            abs(E0[0, 1:-1, 1:-1, 0] - f_E(omega, kx, ky, x, y, (i)*dt)[1:-1, 1:-1])))
        errHx.append(np.mean(
            abs(Hx0[0, :, :, 0] - f_Hx(omega, kx, ky, x, y, (i)*dt, dt, h)[1:-1, :])))
        errHy.append(np.mean(
            abs(Hy0[0, :, :, 0] - f_Hy(omega, kx, ky, x, y, (i)*dt, dt, h)[:, 1:-1])))

        E0 = amper(E0, Hx0, Hy0, beta, delta, gamma, C)
        Hx0, Hy0 = faraday(E0, Hx0, Hy0, beta, delta, gamma, C)

    errE = np.array(errE)
    errHx = np.array(errHx)
    errHy = np.array(errHy)
    return ((np.mean(errE) + np.mean(errHx) + np.mean(errHy)) / 3, errE+errHx+errHy)
