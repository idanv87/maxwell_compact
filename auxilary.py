import numpy as np
import matplotlib.pyplot as plt
import math

import scipy
from scipy.sparse import csr_matrix
from scipy.linalg import circulant
from scipy.sparse import block_diag
from scipy.sparse import vstack

from scipy.sparse.linalg import cg
import pylops
import timeit

import ray

from scipy import signal
import matplotlib.pyplot as plt


def create_A_2(n):
    kernely = np.zeros((n, 1))
    kernely[-1] = 1
    kernely[0] = 22
    kernely[1] = 1
    A = circulant(kernely)
    A[0, -1] = 0
    A[-1, 0] = 0
    A[0, 0] = 26
    A[0, 1] = -5
    A[0, 2] = 4
    A[0, 3] = -1
    A[-1, -1] = 26
    A[-1, -2] = -5
    A[-1, -3] = 4
    A[-1, -4] = -1

    return csr_matrix(A) / 24


def E_step(dt, x, y, h, E, Hx, Hy, Dx, Dy,A):

    start = timeit.default_timer()
    assert E.shape[0] == E.shape[1]

    Uy = Hy[1:, 1:-1] - Hy[:-1, 1:-1]
    Ux = Hx[1:-1, 1:] - Hx[1:-1, :-1]

    dHy_dx = scipy.sparse.linalg.inv(create_A_2(Uy.shape[0]))@(Uy / h)
    dHx_dy = np.transpose((scipy.sparse.linalg.inv(create_A_2(Ux.shape[1]))@np.transpose(Ux) / h))

    dHy_dx_minus_dHx_dy = np.zeros((E.shape[0], E.shape[1]))  # dHx/dy
    dHy_dx_minus_dHx_dy[1:-1, 1:-1] = dHy_dx - dHx_dy

    k = 24 / (dt ** 2)

    #
    BF = k * (dHy_dx_minus_dHx_dy)
    F = BF[1:-1, 1:-1].flatten()
    BE = np.zeros((E.shape[0], E.shape[1]))
    # stop = timeit.default_timer()
    # print('E_step Time: ', stop - start)
    # start = timeit.default_timer()

    E_next = mod_helmholtz(x, y, BE.copy(), -F, -BF, k, Dx, Dy,A)
    stop = timeit.default_timer()
    # print(stop-start)
    # print(q)




    return E + dt * E_next


def Hy_step(dt, x, y, h, E, Hy, BHy, Dx, Dy, A):
    BHy = None
    assert E.shape[0] == E.shape[1]

    # Your statements here

    Ex = E[1:, :] - E[:-1, :]

    dE_dx = (scipy.sparse.linalg.inv(create_A_2(Hy.shape[0]))@ Ex / h)

    # dE_dx= (E[1:, :] - E[:-1, :])/h

    k = (24 / (dt ** 2))
    BF = k * dE_dx
    F = BF[1:-1, 1:-1].flatten()

    # BHy = (E[1:, :] - E[:-1, :]) / h
    lapx1 = (dE_dx[0, 2:] + dE_dx[0, 0:-2] - 2 * dE_dx[0, 1:-1]) / h ** 2
    lapx2 = (dE_dx[-1, 2:] + dE_dx[-1, :-2] - 2 * dE_dx[-1, 1:-1]) / h ** 2
    lapy1 = (2 * dE_dx[0, 1:-1] - 5 * dE_dx[1, 1:-1] + 4 * dE_dx[2, 1:-1] - dE_dx[3, 1:-1]) / h ** 2
    lapy2 = (2 * dE_dx[-1, 1:-1] - 5 * dE_dx[-2, 1:-1] + 4 * dE_dx[-3, 1:-1] - dE_dx[-4, 1:-1]) / h ** 2

    BHy = dE_dx
    BHy[0, 1:-1] += (dt ** 2 / 24) * (lapx1 + lapy1)
    BHy[-1, 1:-1] += (dt ** 2 / 24) * (lapx2 + lapy2)
    BHy[:, 0] = 0
    BHy[:, -1] = 0

    Hy_next = mod_helmholtz(x[1:], y, BHy.copy(), -F, -BF, k, Dx, Dy, A)

    return Hy + dt * Hy_next


def Hx_step(dt, x, y, h, E, Hx, BHx, Dx, Dy, A):
    BHx = None
    assert E.shape[0] == E.shape[1]
    # A = np.linalg.inv(create_A_2(Hx.shape[1]))

    Ey = E[:, 1:] - E[:, :-1]
    dE_dy = np.transpose((scipy.sparse.linalg.inv(create_A_2(Hx.shape[1]))@np.transpose(Ey) / h))

    k = 24 / (dt ** 2)
    BF = -k * dE_dy
    F = BF[1:-1, 1:-1].flatten()

    lapx1 = -(dE_dy[2:, 0] + dE_dy[0:-2, 0] - 2 * dE_dy[1:-1, 0]) / h ** 2
    lapx2 = -(dE_dy[2:, -1] + dE_dy[:-2, -1] - 2 * dE_dy[1:-1, -1]) / h ** 2
    lapy1 = -(2 * dE_dy[1:-1, 0] - 5 * dE_dy[1:-1, 1] + 4 * dE_dy[1:-1, 2] - dE_dy[1:-1, 3]) / h ** 2
    lapy2 = -(2 * dE_dy[1:-1, -1] - 5 * dE_dy[1:-1, -2] + 4 * dE_dy[1:-1, -3] - dE_dy[1:-1, -4]) / h ** 2

    BHx = -dE_dy
    BHx[1:-1, 0] += ((dt ** 2) / 24) * (lapx1 + lapy1)
    BHx[1:-1, -1] += ((dt ** 2) / 24) * (lapx2 + lapy2)
    BHx[0, :] = 0
    BHx[-1, :] = 0

    Hx_next = mod_helmholtz(x, y[1:], BHx.copy(), -F, -BF, k, Dx, Dy, A)

    return Hx + dt * Hx_next


def blocks(A, n):
    B = A
    for j in range(n - 1):
        # B = block_diag(B, A)
        B=block_diag((B,A))

    return B




def create_B(x, y, B):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    B[1:-1, 1:-1] = 0
    Nx = len(x) - 2
    Ny = len(y) - 2
    assert B.shape[0] == len(x) and B.shape[1] == len(y)

    Bx = (np.delete(np.delete(B[:, 1:-1], 1, 0), -2, 0)).flatten() / dx ** 2
    By = (np.delete(np.delete(B[1:-1, :], 1, 1), -2, 1)).flatten() / dy ** 2

    return Bx, By


def mod_helmholtz(x, y, B, F, BF, k, Dx, Dy,C):
    start = timeit.default_timer()

    DxF = Dx
    DyF = Dy

    # Your statements here

    assert k >= 0

    assert (x[1] - x[0]) == (y[1] - y[0])
    h = x[1] - x[0]

    '''
    y''-ky=0
    '''
    Bx, By = create_B(x, y, B.copy())
    BxF, ByF = create_B(x, y, BF.copy())


    la = -k * (1 + k * (h ** 2) / 12)
    # start = timeit.default_timer()
    A = C + la * scipy.sparse.identity(Dx.shape[0])
    stop = timeit.default_timer()
    v = (1 + k * (h ** 2) / 12) * F \
        + ((h ** 2) / 12) * (Dx@ F + BxF +DyF@F + ByF) \
        - Bx - By - (h ** 2 / 6) * (Dx@By + Bx)

    x, exit_code = cg(A, v, tol=1e-09)

    B1 = B.copy()
    B1[1:-1, 1:-1] = x.reshape((B.shape[0] - 2, B.shape[1] - 2))

    # print('helm'+ str(stop-start))
    # print(q)



    return B1


def create_Ds(x, y):

    start = timeit.default_timer()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Nx = len(x) - 2
    Ny = len(y) - 2

    N = Nx * Ny

    kernely = np.zeros((Ny, 1))
    kernely[-1] = 1
    kernely[0] = -2
    kernely[1] = 1

    Dy = circulant(kernely)



    Dy[-1, 0] = 0
    Dy[0, -1] = 0
    Dy = blocks(csr_matrix(Dy), Nx) / (dy ** 2)




    kernelx1 = np.zeros((Nx * Ny, 1))
    kernelx1[0] = -2
    kernelx1[-Ny] = 1
    Dx1 = csr_matrix(circulant(kernelx1)[0:Ny])

    kernelx2 = np.zeros((Nx * Ny, 1))
    kernelx2[Ny] = 1
    kernelx2[0] = -2
    Dx2 = csr_matrix(circulant(kernelx2)[-Ny:])


    kernelx3 = np.zeros((Nx * Ny, 1))
    kernelx3[0] = 1
    kernelx3[Ny] = -2
    kernelx3[2 * Ny] = 1
    Dx3 = csr_matrix(circulant(kernelx3)[2 * Ny:])
    # Dx = np.concatenate([Dx1, Dx3, Dx2]) / (dx ** 2)
    Dx = vstack([Dx1, Dx3, Dx2]) / (dx ** 2)






    return csr_matrix(Dx), csr_matrix(Dy)
