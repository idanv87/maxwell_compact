import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import circulant
from scipy.linalg import block_diag
from scipy.sparse.linalg import cg
import pylops
import timeit
import scipy

from scipy import signal
import matplotlib.pyplot as plt


# t = np.linspace(-2, 2, 200, endpoint=False)
# sig  =signal.gausspulse(t , fc=5)
# sig=np.sin(math.pi*t)
# # plt.plot(sig)
# # plt.show()
# # print(ww)
# widths = np.linspace(0.01, 0.6,70)
# cwtmatr = signal.cwt(sig, signal.ricker, widths)
# plt.imshow(abs(cwtmatr), extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
# vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
# plt.show()
# print(qq)


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

    return A / 24


def E_step(dt, x, y, h, E, Hx, Hy):
    assert E.shape[0] == E.shape[1]
    # A = np.linalg.inv(create_A(E.shape[0]))
    # Uy = np.zeros((E.shape[0], E.shape[1]))  # dHy/dx
    # Uy[1:-1, :] = Hy[1:, :] - Hy[:-1, :]
    Uy = Hy[1:, 1:-1] - Hy[:-1, 1:-1]
    Ux = Hx[1:-1, 1:] - Hx[1:-1, :-1]
    # Uy[0, :] = -31/8*Hy[0,:]+229/24*Hy[1,:]-75/8*Hy[2,:]+37/8*Hy[3,:]-11/12*Hy[4,:]
    # Uy[-1, :] = 31/8*Hy[-1,:]-229/24*Hy[-2,:]+75/8*Hy[-3,:]-37/8*Hy[-4,:]+11/12*Hy[-5,:]
    dHy_dx = np.matmul(np.linalg.inv(create_A_2(Uy.shape[0])), Uy / h)
    dHx_dy = np.transpose(np.matmul(np.linalg.inv(create_A_2(Ux.shape[1])), np.transpose(Ux) / h))

    dHy_dx_minus_dHx_dy = np.zeros((E.shape[0], E.shape[1]))  # dHx/dy
    dHy_dx_minus_dHx_dy[1:-1, 1:-1] = dHy_dx - dHx_dy

    k = 24 / (dt ** 2)

    #
    BF = k * (dHy_dx_minus_dHx_dy)
    F = BF[1:-1, 1:-1].flatten()
    BE = np.zeros((E.shape[0], E.shape[1]))

    E_next = mod_helmholtz(x, y, BE.copy(), -F, -BF, k)
    return E + dt * E_next


def Hy_step(dt, x, y, h, E, Hy, BHy=None):
    assert E.shape[0] == E.shape[1]

    # Your statements here

    # A = create_A_2(Hy.shape[0])
    # Ex = np.zeros((Hy.shape[0], Hy.shape[1]))  # dE/dx
    Ex = E[1:, :] - E[:-1, :]
    # Ex[0, :] = -71/24*E[0,:]+47/8*E[1,:]-31/8*E[2,:]+23/24*E[3,:]
    # Ex[-1, :] = 71/24*E[-1,:]-47/8*E[-2,:]+31/8*E[-3,:]-23/24*E[-4,:]
    dE_dx = np.matmul(np.linalg.inv(create_A_2(Hy.shape[0])), Ex / h)
    # s=np.array([ cg(A, Ex[:,j] / h, tol=1e-09)[0] for j in range(Ex.shape[1])])
    #

    # dE_dx= (E[1:, :] - E[:-1, :])/h

    k = (24 / (dt ** 2))
    BF = k * dE_dx
    F = BF[1:-1, 1:-1].flatten()

    # BHy = (E[1:, :] - E[:-1, :]) / h
    lapx1 = (dE_dx[0,2:]+dE_dx[0,0:-2]-2*dE_dx[0,1:-1])/h**2
    lapx2 = (dE_dx[-1,2:] + dE_dx[-1,:-2] - 2 * dE_dx[-1,1:-1]) / h ** 2
    lapy1=    (2*dE_dx[0,1:-1]-5*dE_dx[1,1:-1]+4*dE_dx[2,1:-1]-dE_dx[3,1:-1])/h**2
    lapy2 = (2*dE_dx[-1,1:-1]-5*dE_dx[-2,1:-1]+4*dE_dx[-3,1:-1]-dE_dx[-4,1:-1])/h**2

    BHy = dE_dx
    BHy[0,1:-1]+=(dt**2/24)*(lapx1+lapy1)
    BHy[-1,1:-1] += (dt**2/24)*(lapx2 + lapy2)
    BHy[:,0]=0
    BHy[:,-1]=0

    Hy_next = mod_helmholtz(x[1:], y, BHy.copy(), -F, -BF, k)

    return Hy + dt * Hy_next


def Hx_step(dt, x, y, h, E, Hx, BHx=None):
    assert E.shape[0] == E.shape[1]
    # A = np.linalg.inv(create_A_2(Hx.shape[1]))

    Ey = E[:, 1:] - E[:, :-1]
    # Ex[0, :] = -71/24*E[0,:]+47/8*E[1,:]-31/8*E[2,:]+23/24*E[3,:]
    # Ex[-1, :] = 71/24*E[-1,:]-47/8*E[-2,:]+31/8*E[-3,:]-23/24*E[-4,:]
    # dE_dy=Ey/h
    dE_dy = np.transpose(np.matmul(np.linalg.inv(create_A_2(Hx.shape[1])), np.transpose(Ey) / h))

    k = 24 / (dt ** 2)
    BF = -k * dE_dy
    F = BF[1:-1, 1:-1].flatten()
    # BHx = -(E[:, 1:] - E[:, :-1]) / h

    lapx1 = -(dE_dy[2:,0]+dE_dy[0:-2,0]-2*dE_dy[1:-1,0])/h**2
    lapx2 = -(dE_dy[2:, -1] + dE_dy[:-2, -1] - 2 * dE_dy[1:-1, -1]) / h ** 2
    lapy1=   -(2*dE_dy[1:-1,0]-5*dE_dy[1:-1,1]+4*dE_dy[1:-1,2]-dE_dy[1:-1,3])/h**2
    lapy2 =-(2*dE_dy[1:-1,-1]-5*dE_dy[1:-1,-2]+4*dE_dy[1:-1,-3]-dE_dy[1:-1,-4])/h**2


    BHx = -dE_dy
    BHx[1:-1,0]+=((dt**2)/24)*(lapx1+lapy1)
    BHx[1:-1, -1] +=((dt**2)/24)*(lapx2 + lapy2)
    BHx[0,:]=0
    BHx[-1,:]=0





    Hx_next = mod_helmholtz(x, y[1:], BHx.copy(), -F, -BF, k)

    return Hx + dt * Hx_next


def blocks(A, n):
    B = A
    for j in range(n - 1):
        B = block_diag(B, A)
    return B


def create_D(x, y, B):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    B[1:-1, 1:-1] = 0
    Nx = len(x) - 2
    Ny = len(y) - 2
    assert B.shape[0] == len(x) and B.shape[1] == len(y)

    Bx = (np.delete(np.delete(B[:, 1:-1], 1, 0), -2, 0)).flatten() / dx ** 2
    By = (np.delete(np.delete(B[1:-1, :], 1, 1), -2, 1)).flatten() / dy ** 2

    N = Nx * Ny

    kernely = np.zeros((Ny, 1))
    kernely[-1] = 1
    kernely[0] = -2
    kernely[1] = 1
    Dy = circulant(kernely)
    Dy[-1, 0] = 0
    Dy[0, -1] = 0
    Dy = blocks(Dy, Nx) / (dy ** 2)

    kernelx1 = np.zeros((Nx * Ny, 1))
    kernelx1[0] = -2
    kernelx1[-Ny] = 1
    Dx1 = circulant(kernelx1)[0:Ny]

    kernelx2 = np.zeros((Nx * Ny, 1))
    kernelx2[Ny] = 1
    kernelx2[0] = -2
    Dx2 = circulant(kernelx2)[-Ny:]

    kernelx3 = np.zeros((Nx * Ny, 1))
    kernelx3[0] = 1
    kernelx3[Ny] = -2
    kernelx3[2 * Ny] = 1
    Dx3 = circulant(kernelx3)[2 * Ny:]

    Dx = np.concatenate([Dx1, Dx3, Dx2]) / (dx ** 2)

    return Dx, Dy, Bx, By


def mod_helmholtz(x, y, B, F, BF, k):
    # start = timeit.default_timer()

    # Your statements here

    assert k >= 0

    assert (x[1] - x[0]) == (y[1] - y[0])
    h = x[1] - x[0]

    '''
    y''-ky=0
    '''
    Dx, Dy, Bx, By = create_D(x, y, B.copy())
    DxF, DyF, BxF, ByF = create_D(x, y, BF.copy())

    la = -k * (1 + k * (h ** 2) / 12)

    A = Dx + Dy + ((h ** 2) / 6) * np.matmul(Dx, Dy) + la * np.eye(Dx.shape[0])
    v = (1 + k * (h ** 2) / 12) * F \
        + ((h ** 2) / 12) * (np.matmul(DxF, F) + BxF + np.matmul(DyF, F) + ByF) \
        - Bx - By - (h ** 2 / 6) * (np.matmul(Dx, By) + Bx)

    x, exit_code = cg(A, v, tol=1e-05)
    B1 = B.copy()
    B1[1:-1, 1:-1] = x.reshape((B.shape[0] - 2, B.shape[1] - 2))
    # stop = timeit.default_timer()

    # print('Time: ', stop - start)
    return B1

# x = np.linspace(0, 1, 21)
# y = np.linspace(0, 1, 21)
# h = x[1]-x[0]
#
# kx=1
# ky=1
# omega=math.pi*np.sqrt(kx**2+ky**2)
#
# T=0.01
# time_steps=120
# dt=T/(time_steps-1)
# t, X, Y = np.meshgrid(np.linspace(0,T,time_steps),x, y, indexing='ij')
# E_a = np.cos(omega*t)*np.sin(math.pi*kx*X) * np.sin(math.pi*ky*Y)
#
# Hx_a =-(np.sin(omega*(t+dt/2))*np.sin(math.pi*kx*X) * np.cos(math.pi*ky*(Y+h/2))*math.pi*ky/omega)[:,:,:-1]
#
# Hy_a = (np.sin(omega*(t+dt/2))*np.cos(math.pi*kx*(X+h/2)) * np.sin(math.pi*ky*Y)*math.pi*kx/omega)[:,:-1, :]
#
# E0=E_a[0]
# Hx0=Hx_a[0]
# Hy0=Hy_a[0]
# E_tot=[]
# Hx_tot=[]
# Hy_tot=[]
# for t in range(time_steps):
#     E_tot.append(E0.copy())
#     Hx_tot.append(Hx0.copy())
#     Hy_tot.append(Hy0.copy())
#     E0=E_step(dt, x, y, h, E0.copy(),Hx0.copy(), Hy0.copy()).copy()
#     Hx0=Hx_step(dt, x, y, h, E0.copy(),Hx0.copy()).copy()
#     Hy0=Hy_step(dt, x, y, h, E0.copy(),Hy0.copy()).copy()
# errE=[np.mean(abs(E_tot[i]-E_a[i])) for i in range(time_steps)]
# errHx=[np.mean(abs(Hx_tot[i][1:-1,1:-1]-Hx_a[i,1:-1,1:-1])) for i in range(time_steps)]
# errHy=[np.mean(abs(Hy_tot[i]-Hy_a[i])) for i in range(time_steps)]
# plt.plot(errE)
# plt.plot(errHx)
# plt.plot(errHy)
# plt.show()
# print(q)
# # print(np.max(abs(E_step(dt, x, y, h, E, Hy)[:,:]-(np.cos(Y+h/2)*np.sin(X))[:,:-1]
#                  # )))
# # N=3
# #
#
# # print(X.shape)
# # F=2*X+Y
# # print(F[1, 2] - (2 * x[1] + y[2]))
# # plt.show()
# # print(q)
#
# # x = np.linspace(0, math.pi, 13)
# # y = np.linspace(0, math.pi, 13)
# #
# # # dx = x[1] - x[0]
# # # dy = y[1] - y[0]
# # X, Y = np.meshgrid(x, y, indexing='ij')
# # B = np.sin(X) * np.sin(Y / 2)
# # BF = -13 / 4 * (np.sin(X) * np.sin(Y / 2))
# # F = BF[1:-1, 1:-1]
# #
# # F = F.flatten()
# #
# # phi = mod_helmholtz(x, y, B.copy(), F.copy(), BF.copy(), k=2)
# #
# # print(max(abs(phi - B[1:-1, 1:-1].flatten())))
# # print(q)
# # Dx, Dy, Bx, By = create_D(x,y,B)
# # B[1:-1,1:-1]=0
# # F_tag_y2 = (-np.cos(X) * np.cos(Y))[1:-1, 1:-1]
# # F_tag_x2 = (-np.cos(X) * np.cos(Y))[1:-1, 1:-1]
# # Nx=F.shape[0]
# # Ny=F.shape[1]
#
# # B=np.random.random((Nx+2 ,Ny+2)) #b.c matrix
#
#
# # Bx=(np.delete(np.delete(B[:,1:-1],1,0),-2,0)).flatten()
# # By=(np.delete(np.delete(B[1:-1,:],1,1),-2,1)).flatten()
# #
# # N = Nx*Ny
#
# #
#
# print(max(abs(np.matmul(Dx, F) + Bx - F_tag_x2.flatten())))
#
# print(q)
#
# ## Diagonal
# # for i in range(0, N ):
# #         # Dx[i + (N - 1) * j, i + (N - 1) * j] = -2
# #         Dy[i,i] = -2
# #         Dx[i,i] = -2
# #
# # # LOWER DIAGONAL
# # for i in range(1, Nx - 1):
# #     for j in range(0, Ny - 1):
# #         Dy[i + (N - 1) * j, i + (N - 1) * j - 1] = 1
# #     # UPPPER DIAGONAL
# # for i in range(0, N - 2):
# #     for j in range(0, N - 1):
# #         Dy[i + (N - 1) * j, i + (N - 1) * j + 1] = 1
# #
# #     # LOWER IDENTITY MATRIX
# # for i in range(0, N - 1):
# #     for j in range(1, N - 1):
# #         Dx[i + (N - 1) * j, i + (N - 1) * (j - 1)] = 1
# #
# #     # UPPER IDENTITY MATRIX
# # for i in range(0, N - 1):
# #     for j in range(0, N - 2):
# #         Dx[i + (N - 1) * j, i + (N - 1) * (j + 1)] = 1
# print(Bx)
