import matplotlib.pyplot as plt
import numpy as np
import math

from auxilary import Hy_step, E_step, mod_helmholtz, Hx_step


def max_solver(dt, x, y, h, time_steps, E_a, Hx_a, Hy_a):
   E_tot = []
   Hx_tot = []
   Hy_tot = []
   E0 = E_a[0]
   Hx0 = Hx_a[0]
   Hy0 = Hy_a[0]
   for t in range(time_steps):
      E_tot.append(E0.copy())
      Hx_tot.append(Hx0.copy())
      Hy_tot.append(Hy0.copy())
      E0 = E_step(dt, x, y, h, E0.copy(), Hx0.copy(), Hy0.copy()).copy()
      Hx0 = Hx_step(dt, x, y, h, E0.copy(), Hx0.copy()).copy()
      Hy0 = Hy_step(dt, x, y, h, E0.copy(), Hy0.copy()).copy()
   errE = np.array([np.mean(abs(E_tot[i][1:-1,1:-1] - E_a[i][1:-1,1:-1])) for i in range(time_steps)])
   errHx =np.array([np.mean(abs(Hx_tot[i][1:-1,:] - Hx_a[i][1:-1,:])) for i in range(time_steps)])
   errHy = np.array([np.mean(abs(Hy_tot[i][:,1:-1]- Hy_a[i][:,1:-1])) for i in range(time_steps)])
   return ((np.mean(errE) + np.mean(errHx) + np.mean(errHy)) / 3)
kx=5
ky=5
omega=math.pi*np.sqrt(kx**2+ky**2)

err=[]
ns=[ 20,25,30,35]

# N=[21,41,81]
for i,n in enumerate(ns):
   print(n)
   x = np.linspace(0, 1, n+1)
   y = np.linspace(0, 1, n+1)
   h = x[1]-x[0]
   time_steps=ns[i]*2+1
   T=1
   dt =1/(time_steps-1)


   t, X, Y = np.meshgrid(np.linspace(0,T,time_steps),x, y, indexing='ij')
   E_a = np.cos(omega*t)*np.sin(math.pi*kx*X) * np.sin(math.pi*ky*Y)
   # BF=-(omega**2+2)*E_a[0]
   # F=BF[1:-1,1:-1].flatten()
   Hx_a =-(np.sin(omega*(t+dt/2))*np.sin(math.pi*kx*X) * np.cos(math.pi*ky*(Y+h/2))*math.pi*ky/omega)[:,:,:-1]
   Hy_a = (np.sin(omega*(t+dt/2))*np.cos(math.pi*kx*(X+h/2)) * np.sin(math.pi*ky*Y)*math.pi*kx/omega)[:,:-1, :]
   # err.append(np.mean(abs(Hy_step(dt, x, y, h, E_a[1], Hy_a[0])[:,1:-1]-Hy_a[1][:,1:-1])))
   # err.append(np.mean(abs(Hx_step(dt, x, y, h, E_a[1], Hx_a[0])[1:-1, :] - Hx_a[1][1:-1,:])))
   err.append(max_solver(dt, x, y, h, time_steps, E_a, Hx_a, Hy_a))
   # err.append(np.mean(abs(E_step(dt, x, y, h, E_a[0], Hx_a[0], Hy_a[0]) - E_a[1])))
   # err.append(np.mean(abs(mod_helmholtz(x, y,BF*0,F,BF ,2) - E_a[0])))





#
x=np.log(1/np.array(ns))
y=np.log(np.array(err))
plt.plot(ns,err)
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