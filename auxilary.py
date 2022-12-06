import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import circulant
from scipy.linalg import block_diag
from itertools import repeat

def blocks(A,n):
    B=A
    for j in range(n-1):
        B=block_diag(B,A)
    return B
# N=3
#

# print(X.shape)
# F=2*X+Y
# print(F[1, 2] - (2 * x[1] + y[2]))
# plt.show()
# print(q)

x=np.arange(0,6,1)
y=np.arange(0,5,1)
X, Y = np.meshgrid(x, y, indexing='ij')
F=np.sin(X)
Nx=F.shape[0]
Ny=F.shape[1]

B=np.random.random((Nx+2 ,Ny+2))
B[1:-1,1:-1]=0
F=F.flatten()
Bx=(np.delete(np.delete(B[:,1:-1],1,0),-2,0)).flatten()
By=(np.delete(np.delete(B[1:-1,:],1,1),-2,1)).flatten()

N = Nx*Ny

kernely=np.zeros((Ny,1))
kernely[-1]=1
kernely[0]=-2
kernely[1]=1
Dy=circulant(kernely)
Dy[-1,0]=0
Dy[0,-1]=0
Dy=blocks(Dy,Nx)

kernelx1=np.zeros((Nx*Ny,1))
kernelx1[0]=-2
kernelx1[-Ny]=1
Dx1=circulant(kernelx1)[0:Ny]

kernelx2=np.zeros((Nx*Ny,1))
kernelx2[Ny]=1
kernelx2[0]=-2
Dx2=circulant(kernelx2)[-Ny:]

kernelx3=np.zeros((Nx*Ny,1))
kernelx3[0]=1
kernelx3[Ny]=-2
kernelx3[2*Ny]=1
Dx3=circulant(kernelx3)[2*Ny:]


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
