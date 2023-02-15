import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as mpatches
from auxilary import Hy_step, E_step, mod_helmholtz, Hx_step, create_Ds2,f_Hy,f_E,f_Hx
import multiprocessing as mp
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import ray
from scipy.sparse import csc_matrix
import timeit
import pickle
from decimal import Decimal



# ray.init()
# @ray.remote
# def fHx(dt, x, y, h, E0, Hx0, BHx):
#      return Hx_step(dt, x, y, h, E0, Hx0, BHx)
#
# @ray.remote
# def fHy(dt, x, y, h, E0, Hy0, BHy):
#      return Hy_step(dt, x, y, h, E0, Hy0, BHy)


def max_solver(fourth, omega,kx,ky, dt, x, y, h, time_steps, DxE, DyE, DxHx, DyHx, DxHy, DyHy, AE, AHx, AHy):

    errE=[]
    errHx=[]
    errHy=[]
    energy=[]

    E0 = f_E(omega, kx, ky, x, y, 0)
    Hx0 = f_Hx(omega, kx, ky, x, y, 0, dt, h)
    Hy0 = f_Hy(omega, kx, ky, x, y, 0, dt, h)
    for i,t in enumerate(range(time_steps - 1)):
        errE.append(np.mean(abs(E0[1:-1, 1:-1] - f_E(omega, kx, ky, x, y, i*dt)[1:-1, 1:-1])))
        errHx.append(np.mean(abs(Hx0[1:-1, :] -f_Hx(omega, kx, ky, x, y, i*dt, dt, h)[1:-1, :])))
        errHy.append(np.mean(abs(Hy0[:, 1:-1] - f_Hy(omega, kx, ky, x, y, i*dt, dt, h)[:, 1:-1])))

        E0 = E_step(dt, x, y, h, E0.copy(), Hx0.copy(), Hy0.copy(), DxE, DyE, AE)
        Hx0 = Hx_step(dt, x, y, h, E0.copy(), Hx0.copy(), DxHx, DyHx, AHx, fourth)
        Hy0 = Hy_step(dt, x, y, h, E0.copy(), Hy0.copy(), DxHy, DyHy, AHy, fourth)
      

    # errE = np.array([np.mean(abs(E_tot[i][1:-1, 1:-1] - E_a[i][1:-1, 1:-1])) for i in range(len(E_tot))])
    # errE = np.array([np.mean(abs(E_tot[i][1:-1, 1:-1] - f_E(omega, kx, ky, x, y, i*dt)[1:-1, 1:-1])) for i in range(len(E_tot))])
    # errHx = np.array([np.mean(abs(Hx_tot[i][1:-1, :] - f_Hx(omega, kx, ky, x, y, i*dt, dt, h)[1:-1, :])) for i in range(len(Hx_tot))])
    # errHy = np.array([np.mean(abs(Hy_tot[i][:, 1:-1] - f_Hy(omega, kx, ky, x, y, i*dt, dt, h)[:, 1:-1])) for i in range(len(Hy_tot))])
    errE=np.array(errE)
    errHx=np.array(errHx)
    errHy=np.array(errHy)
    return ((np.mean(errE) + np.mean(errHx) + np.mean(errHy)) / 3, [errE, errHx, errHy])


def run_scheme(C):


    kx = C.kx
    ky = C.ky
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    

    fourth=C.fourth_order
 


    data = {'T':[],'cfl':[],'N':[],'err':[],'err(time)':[]}

    for T in C.T:
     for cfl in C.cfl:
      for i, n in enumerate(C.ns):
        print(n)
 

        x = np.linspace(0, 1, n + 1)
        y = np.linspace(0, 1, n + 1)
        h = x[1] - x[0]
        dt=h*cfl
        time_steps =  int(T/dt)
        

        DxE, DyE = create_Ds2(x, y)
        DxHx, DyHx = create_Ds2(x, y[1:])
        DxHy, DyHy = create_Ds2(x[1:], y)
        start = timeit.default_timer()
        AE = DxE + DyE + ((h ** 2) / 6) * DxE @ DyE

        AHx = DxHx + DyHx + ((h ** 2) / 6) * DxHx @ DyHx
        AHy = DxHy + DyHy + ((h ** 2) / 6) * DxHy @ DyHy
        data['T'].append(T)
        data['cfl'].append(cfl)
        data['N'].append(n)
        res=max_solver(fourth, omega,kx,ky,  dt, x, y, h, time_steps, DxE, DyE,
                            DxHx, DyHx, DxHy, DyHy,
                            AE, AHx, AHy)
        data['err'].append(res[0])
        data['err(time)'].append(res[1])                    
    return data    

    #
    # x = np.log(1 / np.array(ns))
    # y = np.log(np.array(err))
    # 
    # log_err=np.diff(y) / np.diff(x)
    # err=np.array(err)
    # err=['%.2E' % Decimal(err[i]) for i in range(len(err))]
    # log_err=['%.2f' % log_err[i] for i in range(len(log_err))]

    # path=C.path
    # figure=C.figure

    # with open(path+ figure+'err.txt', 'w') as fp:
    #     for i,item in enumerate(err):
    #         # write each item on a new line
    #         fp.write(str(ns[i])+' & '+ item+ ' \\\\ '+"\n")

    # with open(path+figure+'log_err.txt', 'w') as fp:
    #     for i,item in enumerate(log_err):
    #         # write each item on a new line
    #         fp.write(str(ns[i+1])+' & '+ item+ ' \\\\ '+"\n")

# plt.plot(ns, err)
# plt.show()
