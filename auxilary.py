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

    # x, e = jax.scipy.sparse.linalg.cg(
    # sparse.BCOO.fromdense(A.toarray()), jnp.array(B), tol=1e-9)
    x, e = scipy.sparse.linalg.cg(A, B, tol=1e-9)
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


def E_step(dt, x, y, h, E, Hx, Hy, Dx, Dy, A):
    # start = timeit.default_timer()
    assert E.shape[0] == E.shape[1]

    Uy = Hy[1:, 1:-1] - Hy[:-1, 1:-1]
    Ux = Hx[1:-1, 1:] - Hx[1:-1, :-1]

    dHy_dx = spsolve(create_A_2(Uy.shape[0]), (Uy / h))

    dHx_dy = np.transpose(
        (spsolve(create_A_2(Ux.shape[1]), np.transpose(Ux) / h)))

    # dHy_dx = scipy.sparse.linalg.inv(create_A_2(Uy.shape[0])) @ (Uy / h)
    # dHx_dy = np.transpose((scipy.sparse.linalg.inv(create_A_2(Ux.shape[1])) @ np.transpose(Ux) / h))

    dHy_dx_minus_dHx_dy = np.zeros((E.shape[0], E.shape[1]))  # dHx/dy
    dHy_dx_minus_dHx_dy[1:-1, 1:-1] = dHy_dx - dHx_dy

    k = 24 / (dt ** 2)

    #
    BF = k * (dHy_dx_minus_dHx_dy)
    F = BF[1:-1, 1:-1].ravel()
    BE = np.zeros((E.shape[0], E.shape[1]))

    E_next = mod_helmholtz(x, y, BE.copy(), -F, -BF, k, Dx, Dy, A, True)

    # print(stop-start)
    # print(q)

    return E + dt * E_next


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


def mod_helmholtz(x, y, B, F, BF, k, Dx, Dy, C, fourth=True):
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
    if fourth == True:
        B1[1:-1, 1:-1] = x.reshape((B.shape[0] - 2, B.shape[1] - 2))
    else:
        B1[2:-2, 2:-2] = x.reshape((B.shape[0] - 2,
                                   B.shape[1] - 2))[1:-1, 1:-1]

    # print('helm'+ str(stop-start))
    # print(q)

    return B1


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


def create_D1(x):
    Nx = len(x)
    dx = x[1] - x[0]
    kernel = np.zeros((Nx, 1))
    kernel[-1] = 1
    kernel[0] = -1
    Dx = circulant(kernel)
    Dx[0, -1] = 0
    Dx[-1, 0] = 0
    return csr_matrix(Dx[:-1, :]/dx)


def f_E(omega, kx, ky, x, y, t):
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.cos(omega * t) * np.sin(math.pi * kx * X) * np.sin(math.pi * ky * Y)
    # BF=-(omega**2+2)*E_a[0]
    # F=BF[1:-1,1:-1].flatten()


def f_Hx(omega, kx, ky, x, y, t, dt, h):
    X, Y = np.meshgrid(x, y, indexing='ij')
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

# def mod_helmholtz_2(x,y, h,dt, x_or_y,F):


# x=np.linspace(0,1,400)
# h=x[1]-x[0]
# X,Y=np.meshgrid(x,x[:-1]+h/2,indexing='ij')

# u=np.sin(math.pi*X)*np.cos(math.pi*Y)
# F=(u*(1+2*math.pi**2))[1:-1,:]
# g=mod_helmholtz_2(x,x, h, 100, 'x',F.flatten())
# print(np.mean(abs(u-g)) )


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


def E_step2(dt, x, y, h, E, le, Hx, Hy, p1, p2):
    k = 24/dt**2
    Uy = Hy[1:, 1:-1] - Hy[:-1, 1:-1]
    Ux = Hx[1:-1, 1:] - Hx[1:-1, :-1]

    dHy_dx = spsolve(create_A_2(Uy.shape[0]), (Uy / h))

    dHx_dy = np.transpose(
        (spsolve(create_A_2(Ux.shape[1]), np.transpose(Ux) / h)))

    F = (dHy_dx - dHx_dy).ravel()
    E_next = E+dt*mod_helmholtz_2(x, y, h, dt, 'e', F, le, p1, p2)
    LE = E_next*0
    LE[1:-1, 1:-1] = dt*k*(dHy_dx - dHx_dy)-k * \
        (E_next[1:-1, 1:-1]-E[1:-1, 1:-1])-le[1:-1, 1:-1]

    return -LE.copy(), E_next

# x=np.linspace(0,1,10)
# y=x
# h=x[2]-x[1]
# dt=0.1
# x_or_y='y'
# E=np.random.rand(len(x),len(x))
# Hy=np.random.rand(len(x)-1,len(x))
# Hx=np.random.rand(len(x),len(x)-1)
# B=E_step(dt, x, y, h, E, Hx)


# p1,p2=create_P1(x[1:-1],y[1:],h,dt,x_or_y)


def Hy_step(dt, x, y, h, E, Hy, Dx, Dy, A, fourth=True):
    BHy = None
    assert E.shape[0] == E.shape[1]

    # Your statements here

    Ex = E[1:, :] - E[:-1, :]

    dE_dx = spsolve(create_A_2(Hy.shape[0]), Ex / h)

    # dE_dx = (scipy.sparse.linalg.inv(create_A_2(Hy.shape[0])) @ Ex / h)
    k = (24 / (dt ** 2))
    BF = k * dE_dx
    F = BF[1:-1, 1:-1].flatten()

    # BHy = (E[1:, :] - E[:-1, :]) / h
    BHy = dE_dx
    if fourth == True:
        lapx1 = (dE_dx[0, 2:] + dE_dx[0, 0:-2] - 2 * dE_dx[0, 1:-1]) / h ** 2
        lapx2 = (dE_dx[-1, 2:] + dE_dx[-1, :-2] - 2 * dE_dx[-1, 1:-1]) / h ** 2
        lapy1 = (2 * dE_dx[0, 1:-1] - 5 * dE_dx[1, 1:-1] +
                 4 * dE_dx[2, 1:-1] - dE_dx[3, 1:-1]) / h ** 2
        lapy2 = (2 * dE_dx[-1, 1:-1] - 5 * dE_dx[-2, 1:-1] +
                 4 * dE_dx[-3, 1:-1] - dE_dx[-4, 1:-1]) / h ** 2
        BHy[0, 1:-1] += (dt ** 2 / 24) * (lapx1 + lapy1)
        BHy[-1, 1:-1] += (dt ** 2 / 24) * (lapx2 + lapy2)
    BHy[:, 0] = 0
    BHy[:, -1] = 0

    Hy_next = mod_helmholtz(x[1:], y, BHy.copy(), -
                            F, -BF, k, Dx, Dy, A, fourth)

    return Hy + dt * Hy_next


def Hx_step(dt, x, y, h, E, Hx, Dx, Dy, A, fourth=True):
    BHx = None
    assert E.shape[0] == E.shape[1]
    # A = np.linalg.inv(create_A_2(Hx.shape[1]))

    Ey = E[:, 1:] - E[:, :-1]
    dE_dy = np.transpose(
        spsolve(create_A_2(Hx.shape[1]), np.transpose(Ey) / h))

    # dE_dy = np.transpose((scipy.sparse.linalg.inv(create_A_2(Hx.shape[1])) @ np.transpose(Ey) / h))

    k = 24 / (dt ** 2)
    BF = -k * dE_dy
    F = BF[1:-1, 1:-1].flatten()
    BHx = -dE_dy
    if fourth == True:
        lapx1 = -(dE_dy[2:, 0] + dE_dy[0:-2, 0] - 2 * dE_dy[1:-1, 0]) / h ** 2
        lapx2 = -(dE_dy[2:, -1] + dE_dy[:-2, -1] -
                  2 * dE_dy[1:-1, -1]) / h ** 2
        lapy1 = -(2 * dE_dy[1:-1, 0] - 5 * dE_dy[1:-1, 1] +
                  4 * dE_dy[1:-1, 2] - dE_dy[1:-1, 3]) / h ** 2
        lapy2 = -(2 * dE_dy[1:-1, -1] - 5 * dE_dy[1:-1, -2] +
                  4 * dE_dy[1:-1, -3] - dE_dy[1:-1, -4]) / h ** 2
        BHx[1:-1, 0] += ((dt ** 2) / 24) * (lapx1 + lapy1)
        BHx[1:-1, -1] += ((dt ** 2) / 24) * (lapx2 + lapy2)
    BHx[0, :] = 0
    BHx[-1, :] = 0

    Hx_next = mod_helmholtz(
        x, y[1:], BHx.copy(), -F, -BF, k, Dx, Dy, A, fourth)

    return Hx + dt * Hx_next
