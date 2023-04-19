import jax
from jax.experimental import sparse
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import multiprocessing
import timeit
import datetime
import os

import scipy
from scipy.sparse import csr_matrix, kron, identity
from scipy.linalg import circulant
from scipy.sparse import block_diag
from scipy.sparse import vstack

# from jax.scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve, cg

import pylops
import timeit

import ray

from scipy import signal
import matplotlib.pyplot as plt


def solve_cg(A, B):
    x, e = scipy.sparse.linalg.cg(A, B, tol=1e-13, atol=1e-13)
    
    return np.array(x)


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

    return csr_matrix(A/24)




def blocks(A, n):
    B = A
    for j in range(n - 1):
        # B = block_diag(B, A)
        B = block_diag((B, A))

    return B







def create_D2(x):
    Nx = len(x[1:-1])
    dx = x[1] - x[0]
    kernel = np.zeros((Nx, 1))
    kernel[-1] = 1
    kernel[0] = -2
    kernel[1] = 1
    D2 = circulant(kernel)
    D2[0, -1] = 0
    D2[-1, 0] = 0

    return D2/dx/dx


def create_second(x):
    Nx = len(x)
    dx = x[1] - x[0]
    kernel = np.zeros((Nx, 1))
    kernel[-1] = 1
    kernel[0] = -2
    kernel[1] = 1
    D2 = circulant(kernel)
    D2[0, -1] = 0
    D2[-1, 0] = 0

    return D2/dx/dx


def create_Ds2(x, y):
    return csr_matrix(kron(create_D2(x), identity(len(y)-2))), csr_matrix(kron(identity(len(x)-2), create_D2(y)))





def f_E(omega, kx, ky, x, y, t):
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.cos(omega * t) * np.sin(math.pi * kx * X) * np.sin(math.pi * ky * Y)
    # BF=-(omega**2+2)*E_a[0]
    # F=BF[1:-1,1:-1].flatten()


def f_Hx(omega, kx, ky, x, y, t, dt, h):
    X, Y = np.meshgrid(x, y, indexing='ij')
    -(np.sin(omega * (t + dt / 2)) * np.sin(math.pi * kx * X) * np.cos(
        math.pi * ky * (Y + h / 2)) * math.pi * ky / omega)[:, :-1]
    return -(np.sin(omega * (t + dt / 2)) * np.sin(math.pi * kx * X) * np.cos(
        math.pi * ky * (Y + h / 2)) * math.pi * ky / omega)[:, :-1]


def f_Hy(omega, kx, ky, x, y, t, dt, h):
    X, Y = np.meshgrid(x, y, indexing='ij')
    return (np.sin(omega * (t + dt / 2)) * np.cos(math.pi * kx * (X + h / 2)) * np.sin(
        math.pi * ky * Y) * math.pi * kx / omega)[:-1, :]


def conv_rate(N, err):
    x = np.log(1 / np.array(N))
    y = np.log(np.array(err))

    log_err = np.diff(y) / np.diff(x)

    log_err = ['%.2f' % log_err[i] for i in range(len(log_err))]
    return(log_err)

def create_lap(x,y):
    Dxx = create_second(x)
    Dyy = create_second(y)
    Dxx = csr_matrix(kron(Dxx, np.eye(len(y))))
    Dyy = csr_matrix(kron(np.eye(len(x)), Dyy))
    return Dxx+Dyy

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
def create_P2(x, y, h, dt, x_or_y):
    r = dt/h
    k = 24/dt**2

    Dxx = create_second(x)
    Dyy = create_second(y)
    # if x_or_y!='e':
    #         Dyy[0,0]=2/h**2
    #         Dyy[0,1]=-5/h**2
    #         Dyy[0,2]=4/h**2
    #         Dyy[0,3]=-1/h**2

    #         Dyy[-1,-1]=2/h**2
    #         Dyy[-1,-2]=-5/h**2
    #         Dyy[-1,-3]=4/h**2
    #         Dyy[-1,-4]=-1/h**2

    #         Dxx[0,0]=2/h**2
    #         Dxx[0,1]=-5/h**2
    #         Dxx[0,2]=4/h**2
    #         Dxx[0,3]=-1/h**2

    #         Dxx[-1,-1]=2/h**2
    #         Dxx[-1,-2]=-5/h**2
    #         Dxx[-1,-3]=4/h**2
    #         Dxx[-1,-4]=-1/h**2

    Dxx = csr_matrix(kron(Dxx, np.eye(len(y))))
    Dyy = csr_matrix(kron(np.eye(len(x)), Dyy))

    if x_or_y == 'e':
        p2 = k*(1+k*h**2/12)*csr_matrix(kron(np.eye(len(x)),
                                             np.eye(len(y))))+k*h**2/12*(Dxx+Dyy)
    else:
        p2 = k*(1+k*h**2/12)*csr_matrix(kron(np.eye(len(x)), np.eye(len(y))))

    return p2


def create_P1(x, y, h, dt, x_or_y):
    r = dt/h
    k = 24/dt**2
    Dxx = create_second(x)
    Dyy = create_second(y)

    if x_or_y == 'x':  # for Hx
        assert (len(y)-len(x)) == 1
        Dyy[0, 0] = Dyy[0, 0]/2
        Dyy[-1, -1] = Dyy[-1, -1]/2
    if x_or_y == 'y':
        Dxx[0, 0] = Dxx[0, 0]/2
        Dxx[-1, -1] = Dxx[-1, -1]/2

    Dxx = csr_matrix(kron(Dxx, np.eye(len(y))))
    Dyy = csr_matrix(kron(np.eye(len(x)), Dyy))
    p1 = -(Dxx+Dyy+(h**2/6)*Dxx@Dyy)+k*(1+k*h**2/12) * \
        csr_matrix(kron(np.eye(len(x)), np.eye(len(y))))
    return p1


def mod_helmholtz_2(x, y, h, dt, x_or_y, F, LF, p1, p2):
    k = 24/dt**2
    start = timeit.default_timer()

    if x_or_y == 'x':
        start = timeit.default_timer()
        # p1=create_P1(x[1:-1],x[1:],h,dt,'x')
        # p2=create_P2(x[1:-1],x[1:],h,dt,'x')

        # v=p2@F

        v = p2@F + \
            k*h**2/12*LF
        A = p1
        sol = solve_cg(A, v)
        sol = sol.reshape(len(x)-2, len(x)-1)
        B = np.zeros((len(x), len(x)-1))
        B[1:-1, :] = sol

    if x_or_y == 'y':
        # p1=create_P1(x[1:],x[1:-1],h,dt,'y')
        # p2=create_P2(x[1:],x[1:-1],h,dt,'y')

        # v=p2@F
        v = p2@F + \
            k*h**2/12*LF
        A = p1

        sol = solve_cg(A, v)

        sol = sol.reshape(len(x)-1, len(x)-2)

        B = np.zeros((len(x)-1, len(x)))
        B[:, 1:-1] = sol

    if x_or_y == 'e':
        # p1=create_P1(x[1:],x[1:-1],h,dt,'y')
        # p2=create_P2(x[1:],x[1:-1],h,dt,'y')

        v = p2@F
        A = p1

        sol = solve_cg(A, v)

        sol = sol.reshape(len(x)-2, len(x)-2)

        B = np.zeros((len(x), len(x)))
        B[1:-1:, 1:-1] = sol
    #  print( timeit.default_timer()-start)
    return B



def Hy_step2(dt, x, y, h, E, LE, Hy, p1, p2):

    assert E.shape[0] == E.shape[1]

    Ex = E[1:, :] - E[:-1, :]
    dE_dx = spsolve(create_A_2(Hy.shape[0]), Ex / h)
    F = dE_dx[:, 1:-1].ravel()

    Ex = LE[1:, :] - LE[:-1, :]
    dE_dx = spsolve(create_A_2(Hy.shape[0]), Ex / h)
    LF = dE_dx[:, 1:-1].ravel()

    Hy_next = mod_helmholtz_2(x, y, h, dt, 'y', F, LF, p1, p2)

    return Hy + dt * Hy_next


def Hx_step2(dt, x, y, h, E, LE, Hx, p1, p2):
    assert E.shape[0] == E.shape[1]
    # A = np.linalg.inv(create_A_2(Hx.shape[1]))

    Ey = E[:, 1:] - E[:, :-1]
    dE_dy = np.transpose(
        spsolve(create_A_2(Hx.shape[1]), np.transpose(Ey) / h))
    F = -dE_dy[1:-1, :].ravel()

    Ey = LE[:, 1:] - LE[:, :-1]
    dE_dy = np.transpose(
        spsolve(create_A_2(Hx.shape[1]), np.transpose(Ey) / h))
    LF = -dE_dy[1:-1, :].ravel()
    Hx_next = mod_helmholtz_2(x, y, h, dt, 'x', F, LF, p1, p2)

    return Hx + dt * Hx_next


def E_step2(dt, x, y, h, E, le, Hx, Hy, p1, p2, plap,usual):
    
    k = 24/dt**2
    Uy = Hy[1:, 1:-1] - Hy[:-1, 1:-1]
    Ux = Hx[1:-1, 1:] - Hx[1:-1, :-1]

    dHy_dx = spsolve(create_A_2(Uy.shape[0]), (Uy / h))

    dHx_dy = np.transpose(
        (spsolve(create_A_2(Ux.shape[1]), np.transpose(Ux) / h)))

    F = (dHy_dx - dHx_dy).ravel()
    E_next = E+dt*mod_helmholtz_2(x, y, h, dt, 'e', F, le, p1, p2)

    # here we claculate the laplacian of E_n+1
    LE = E_next*0
    # change this line to switch to usual 2 order laplacian:
    if usual:
        LE[1:-1, 1:-1] =-( dt*k*(dHy_dx - dHx_dy)-k * \
        (E_next[1:-1, 1:-1]-E[1:-1, 1:-1])-le[1:-1, 1:-1])
    else:
        LE[1:-1,1:-1]=(plap@(E_next[1:-1,1:-1].ravel())).reshape(len(x)-2,len(x)-2)



    return LE.copy(), E_next






def mod_helmholtz_3(x, y, h, dt, x_or_y, F, LF, p1, p2):
    k = 24/dt**2
    start = timeit.default_timer()

    if x_or_y == 'x':
        start = timeit.default_timer()
        # p1=create_P1(x[1:-1],x[1:],h,dt,'x')
        # p2=create_P2(x[1:-1],x[1:],h,dt,'x')

        # v=p2@F

        v = p2@F + \
            k*h**2/12*LF
        A = p1
        sol = solve_cg(A, v)
        sol = sol.reshape(len(x)-2, len(x)-1)
        B = np.zeros((len(x), len(x)-1))
        B[1:-1, :] = sol

    if x_or_y == 'y':
        # p1=create_P1(x[1:],x[1:-1],h,dt,'y')
        # p2=create_P2(x[1:],x[1:-1],h,dt,'y')

        # v=p2@F
        v = p2@F + \
            k*h**2/12*LF
        A = p1

        sol = solve_cg(A, v)

        sol = sol.reshape(len(x)-1, len(x)-2)

        B = np.zeros((len(x)-1, len(x)))
        B[:, 1:-1] = sol

    if x_or_y == 'e':
        # p1=create_P1(x[1:],x[1:-1],h,dt,'y')
        # p2=create_P2(x[1:],x[1:-1],h,dt,'y')

        v = p2@F + \
            k*h**2/12*LF
        A = p1

        sol = solve_cg(A, v)

        sol = sol.reshape(len(x)-2, len(x)-2)

        B = np.zeros((len(x), len(x)))
        B[1:-1:, 1:-1] = sol
    #  print( timeit.default_timer()-start)
    return B




