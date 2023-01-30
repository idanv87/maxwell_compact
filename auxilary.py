import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import multiprocessing
import timeit
import datetime
from scipy import sparse

import scipy
from scipy.sparse import csr_matrix, kron, identity
from scipy.linalg import circulant
from scipy.sparse import block_diag
from scipy.sparse import vstack

from scipy.sparse.linalg import cg, spsolve
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


def E_step(dt, x, y, h, E, Hx, Hy, Dx, Dy, A):
    start = timeit.default_timer()
    assert E.shape[0] == E.shape[1]

    Uy = Hy[1:, 1:-1] - Hy[:-1, 1:-1]
    Ux = Hx[1:-1, 1:] - Hx[1:-1, :-1]

    dHy_dx = spsolve(create_A_2(Uy.shape[0]), (Uy / h))

    dHx_dy = np.transpose((spsolve(create_A_2(Ux.shape[1]), np.transpose(Ux) / h)))


    # dHy_dx = scipy.sparse.linalg.inv(create_A_2(Uy.shape[0])) @ (Uy / h)
    # dHx_dy = np.transpose((scipy.sparse.linalg.inv(create_A_2(Ux.shape[1])) @ np.transpose(Ux) / h))

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

    E_next = mod_helmholtz(x, y, BE.copy(), -F, -BF, k, Dx, Dy, A)
    # print(stop-start)
    # print(q)

    return E + dt * E_next


def Hy_step(dt, x, y, h, E, Hy, Dx, Dy, A):
    BHy = None
    assert E.shape[0] == E.shape[1]

    # Your statements here

    Ex = E[1:, :] - E[:-1, :]

    dE_dx = spsolve(create_A_2(Hy.shape[0]) ,Ex / h)

    # dE_dx = (scipy.sparse.linalg.inv(create_A_2(Hy.shape[0])) @ Ex / h)



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


def Hx_step(dt, x, y, h, E, Hx, Dx, Dy, A):
    BHx = None
    assert E.shape[0] == E.shape[1]
    # A = np.linalg.inv(create_A_2(Hx.shape[1]))

    Ey = E[:, 1:] - E[:, :-1]
    dE_dy = np.transpose(spsolve(create_A_2(Hx.shape[1]),np.transpose(Ey) / h))


    # dE_dy = np.transpose((scipy.sparse.linalg.inv(create_A_2(Hx.shape[1])) @ np.transpose(Ey) / h))

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
        B = block_diag((B, A))

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


def mod_helmholtz(x, y, B, F, BF, k, Dx, Dy, C):
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
    A = C + la * identity(Dx.shape[0])

    v = (1 + k * (h ** 2) / 12) * F \
        + ((h ** 2) / 12) * (Dx @ F + BxF + DyF @ F + ByF) \
        - Bx - By - (h ** 2 / 6) * (Dx @ By + Bx)


    x, exit_code = cg(A, v, tol=1e-09)


    B1 = B.copy()
    B1[1:-1, 1:-1] = x.reshape((B.shape[0] - 2, B.shape[1] - 2))

    # print('helm'+ str(stop-start))
    # print(q)

    return B1


def create_Ds(x, y):
    pass
    # start = timeit.default_timer()
    # dx = x[1] - x[0]
    # dy = y[1] - y[0]
    # Nx = len(x) - 2
    # Ny = len(y) - 2
    #
    # N = Nx * Ny
    #
    # kernely = np.zeros((Ny, 1))
    # kernely[-1] = 1
    # kernely[0] = -2
    # kernely[1] = 1
    #
    # Dy = circulant(kernely)
    # Dy[-1, 0] = 0
    # Dy[0, -1] = 0
    # Dy = blocks(csr_matrix(Dy), Nx) / (dy ** 2)
    #
    # kernelx1 = np.zeros((Nx * Ny, 1))
    # kernelx1[0] = -2
    # kernelx1[-Ny] = 1
    # Dx1 = csr_matrix(circulant(kernelx1)[0:Ny])
    #
    # kernelx2 = np.zeros((Nx * Ny, 1))
    # kernelx2[Ny] = 1
    # kernelx2[0] = -2
    # Dx2 = csr_matrix(circulant(kernelx2)[-Ny:])
    #
    # kernelx3 = np.zeros((Nx * Ny, 1))
    # kernelx3[0] = 1
    # kernelx3[Ny] = -2
    # kernelx3[2 * Ny] = 1
    # Dx3 = csr_matrix(circulant(kernelx3)[2 * Ny:])
    # # Dx = np.concatenate([Dx1, Dx3, Dx2]) / (dx ** 2)
    # Dx = vstack([Dx1, Dx3, Dx2]) / (dx ** 2)




# def create_fd(ns, path):
#     for i, n in enumerate(ns):
#         x = np.linspace(0, 1, n + 1)
#         y = np.linspace(0, 1, n + 1)
#         DxE, DyE = create_Ds(x, y)
#         DxHx, DyHx = create_Ds(x, y[1:])
#         DxHy, DyHy = create_Ds(x[1:], y)
#         pickle.dump([DxE,DyE,DxHx,DyHx,DxHy,DyHy], open(path + 'fd_matrices/' + str(n) + '.pkl', "wb"))

def create_D2(x):
    Nx=len(x[1:-1])
    dx = x[1] - x[0]
    kernel = np.zeros((Nx, 1))
    kernel[-1] = 1
    kernel[0] = -2
    kernel[1] = 1
    D2 = circulant(kernel)
    D2[0,-1]=0
    D2[-1,0]=0
    D2=csr_matrix(D2)
    return D2/dx/dx


def create_Ds2(x,y):
    return kron(create_D2(x), identity(len(y)-2) ), kron( identity(len(x)-2),create_D2(y))


def create_D1(x):
    Nx = len(x)
    dx = x[1] - x[0]
    kernel = np.zeros((Nx, 1))
    kernel[-1] = 1
    kernel[0] = -1
    Dx = circulant(kernel)
    Dx[0,-1]=0
    Dx[-1,0]=0
    return csr_matrix(Dx[:-1,:]/dx)
def f_E(omega,kx,ky,x,y,t):
    X,Y=np.meshgrid( x, y, indexing='ij')
    return  np.cos(omega * t) * np.sin(math.pi * kx * X) * np.sin(math.pi * ky * Y)
    # BF=-(omega**2+2)*E_a[0]
    # F=BF[1:-1,1:-1].flatten()
def f_Hx(omega, kx, ky, x, y, t, dt, h):
    X, Y = np.meshgrid(x, y, indexing='ij')
    return  -(np.sin(omega * (t + dt / 2)) * np.sin(math.pi * kx * X) * np.cos(
        math.pi * ky * (Y + h / 2)) * math.pi * ky / omega)[:,:-1]
def f_Hy(omega, kx, ky, x, y, t, dt, h):
    X, Y = np.meshgrid(x, y, indexing='ij')
    return  (np.sin(omega * (t + dt / 2)) * np.cos(math.pi * kx * (X + h / 2)) * np.sin(
        math.pi * ky * Y) * math.pi * kx / omega)[:-1,:]






# def create_circulant():
#   N = 7
#
#   vals = np.array([2.0 / 3, -1.0 / 12, 1.0 / 12, -2.0 / 3])
#
#   offsets = np.array([1, 2, N - 2, N - 1])
#
#   dupvals = np.concatenate((vals, vals[::-1]))
#
#   dupoffsets = np.concatenate((offsets, -offsets))
#
#   a = sparse.diags(dupvals, dupoffsets, shape=(N, N))
#   return a


