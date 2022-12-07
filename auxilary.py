import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import circulant
from scipy.linalg import block_diag
from scipy.sparse.linalg import cg
import pylops



def blocks(A,n):
    B=A
    for j in range(n-1):
        B=block_diag(B,A)
    return B
def create_D(x,y,B):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    B[1:-1, 1:-1] = 0
    Nx=len(x)-2
    Ny=len(y)-2
    assert B.shape[0]==len(x) and  B.shape[1]==len(y)

    Bx = (np.delete(np.delete(B[:, 1:-1], 1, 0), -2, 0)).flatten()/dx**2
    By = (np.delete(np.delete(B[1:-1, :], 1, 1), -2, 1)).flatten()/dy**2

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

    return Dx,Dy,Bx,By
# def der2x(Dx,Bx,y):
#     return np.matmul(Dx,y)+Bx
# def der2y(Dy,By,y):
#     return np.matmul(Dy,y)+By



def mod_helmholtz(x,y,B,F,BF, k):
    assert k >=0
    assert (x[1]-x[0])==(y[1]-y[0])
    h=x[1]-x[0]

    '''
    y''-ky=0
    '''

    Dx, Dy, Bx, By = create_D(x, y, B.copy())
    DxF, DyF, BxF, ByF = create_D(x, y, BF.copy())

    la=-k*(1+k*(h**2)/12)

    A=Dx+Dy+ ((h**2)/6)*np.matmul(Dx,Dy)+ la*np.eye(Dx.shape[0])

    v=(1+k*(h**2)/12)*F\
      +((h**2)/12)*(np.matmul(DxF,F)+BxF+np.matmul(DyF,F)+ByF) \
      -Bx-By -(h**2/6)*(np.matmul(Dx,By)+Bx)

    x, exit_code = cg(A, v, tol=1e-03)
    # print(x.shape)


    return x

# N=3
#

# print(X.shape)
# F=2*X+Y
# print(F[1, 2] - (2 * x[1] + y[2]))
# plt.show()
# print(q)

x=np.linspace(0,math.pi,13)
y=np.linspace(0,math.pi,13)

# dx = x[1] - x[0]
# dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='ij')
B=np.sin(X)*np.sin(Y/2)
BF=-13/4*(np.sin(X)*np.sin(Y/2))
F=BF[1:-1,1:-1]

F=F.flatten()

phi=mod_helmholtz(x,y,B.copy(),F.copy(),BF.copy(), k=2)


print(max(abs(phi-B[1:-1,1:-1].flatten())))
print(q)
# Dx, Dy, Bx, By = create_D(x,y,B)
# B[1:-1,1:-1]=0
F_tag_y2=(-np.cos(X)*np.cos(Y))[1:-1,1:-1]
F_tag_x2=(-np.cos(X)*np.cos(Y))[1:-1,1:-1]
# Nx=F.shape[0]
# Ny=F.shape[1]

# B=np.random.random((Nx+2 ,Ny+2)) #b.c matrix



# Bx=(np.delete(np.delete(B[:,1:-1],1,0),-2,0)).flatten()
# By=(np.delete(np.delete(B[1:-1,:],1,1),-2,1)).flatten()
#
# N = Nx*Ny

#

print(max(abs(np.matmul(Dx, F) +Bx- F_tag_x2.flatten())))


print(q)


## Diagonal
# for i in range(0, N ):
#         # Dx[i + (N - 1) * j, i + (N - 1) * j] = -2
#         Dy[i,i] = -2
#         Dx[i,i] = -2
#
# # LOWER DIAGONAL
# for i in range(1, Nx - 1):
#     for j in range(0, Ny - 1):
#         Dy[i + (N - 1) * j, i + (N - 1) * j - 1] = 1
#     # UPPPER DIAGONAL
# for i in range(0, N - 2):
#     for j in range(0, N - 1):
#         Dy[i + (N - 1) * j, i + (N - 1) * j + 1] = 1
#
#     # LOWER IDENTITY MATRIX
# for i in range(0, N - 1):
#     for j in range(1, N - 1):
#         Dx[i + (N - 1) * j, i + (N - 1) * (j - 1)] = 1
#
#     # UPPER IDENTITY MATRIX
# for i in range(0, N - 1):
#     for j in range(0, N - 2):
#         Dx[i + (N - 1) * j, i + (N - 1) * (j + 1)] = 1
print(Bx)
