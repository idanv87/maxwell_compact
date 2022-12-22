import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as mpatches
from auxilary import Hy_step, E_step, mod_helmholtz, Hx_step
# n=4
# h=1/n
# x = np.linspace(0, 1, n + 1)
# y = np.linspace(h/2, 1+h/2, n + 1)[:-1]
#
# x1 = np.linspace(0, 1, n + 1)
# y1 = np.linspace(0, 1, n + 1)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# for i in range(len(x)):
#    for j in range(len(y)):
#       ax1.scatter(x[i],y[j],color='red',marker='x',s=200)
#       # ax1.scatter(x[0], y[j], color='red')
#       # ax1.scatter(x[-1], y[j], color='red')
# # ax1.scatter(x[i], y[0], color='red')
# [ax1.scatter(x[-1], y[j], color='red',s=200,marker='s') for j in range(len(y))]
# [ax1.scatter(x[0], y[j], color='red',s=200, marker='s') for j in range(len(y))]
# [ax1.scatter(x[i], y[-1], color='red',s=700, marker='o') for i in range(len(x))[1:-1]]
# [ax1.scatter(x[i], y[0], color='red',s=700, marker='o') for i in range(len(x))[1:-1]]
# for i in range(len(x1)):
#    for j in range(len(y1)):
#       ax1.scatter(x1[i], y1[j], color='blue',s=10)
# ax1.title.set_text('The domain where we solve Helmholtz for H_x')
# plt.show()
# print(q)

n=5
h=1/n
x = np.linspace(0, 1, n + 1)
y = np.linspace(0, 1, n + 1)

x1 = np.linspace(0, 1, n + 1)
y1 = np.linspace(h/2, 1+h/2, n + 1)[:-1]

x2 = np.linspace(h/2, 1+h/2, n + 1)[:-1]
y2 = np.linspace(0, 1, n + 1)

fig, ax1 = plt.subplots(1, sharex=False, sharey=False)
for i in range(len(x)):
   for j in range(len(y)):
       ax1.scatter(x[i],y[j],color='black',s=5)
      # ax1.scatter(x[i], y[j], color='black', s=100, marker='x')
      # ax1.scatter(x[0], y[j], color='black', s=100, marker='s')
      # ax1.scatter(x[-1], y[j], color='black', s=100, marker='s')
      # ax1.scatter(x[i], y[0], color='black', s=100, marker='s')
      # ax1.scatter(x[i], y[-1], color='black', s=100, marker='s')

for i in range(len(x1)):
   for j in range(len(y1)):
      ax1.scatter(x1[i], y1[j], color='blue', s=100,marker='x')
      ax1.scatter(x1[0], y1[j], color='blue', s=100, marker='s')
      ax1.scatter(x1[-1], y1[j], color='blue', s=100, marker='s')
      ax1.scatter(x1[i], y1[0], color='blue', s=100, marker='s')
      ax1.scatter(x1[i], y1[-1], color='blue', s=100, marker='s')
      # ax1.scatter(x1[i], y1[j], color='blue', s=50, marker='x')

# for i in range(len(x2)):
#    for j in range(len(y2)):
#       ax1.scatter(x2[i], y2[j], color='red', s=5)


black_patch = mpatches.Patch(color='black', label='E_z')
blue_patch = mpatches.Patch(color='blue', label='H_x')
red_patch = mpatches.Patch(color='red', label='H_y')
ax1.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5,
           handles=[black_patch,blue_patch])

box = ax1.get_position()
ax1.title.set_text('The domain where we solve Helmholtz for H_x')
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
# plt.grid()
PATH = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'
plt.savefig(PATH + 'Hx.eps' , format='eps',
            bbox_inches='tight')
# plt.show()
# print(q)
#
#
# X1,Y1=np.meshgrid(x,y,indexing='ij')
# plt.scatter(X1,Y1)
# y = np.linspace(h, 1+h, n + 1)
# X,Y=np.meshgrid(x,y,indexing='ij')
# # plt.scatter(X,Y)
#
# plt.show()
# print(q)



def max_solver(dt, x, y, h, time_steps, E_a, Hx_a, Hy_a):
   E_tot = []
   Hx_tot = []
   Hy_tot = []
   E0 = E_a[0]
   Hx0 = Hx_a[0]
   Hy0 = Hy_a[0]
   for t in range(time_steps-1):
      E_tot.append(E0)
      Hx_tot.append(Hx0)
      Hy_tot.append(Hy0)
      E0 = E_step(dt, x, y, h, E0.copy(), Hx0.copy(), Hy0.copy())
      Hx0 = Hx_step(dt, x, y, h, E0.copy(), Hx0.copy(), (Hx_a[t+1]-Hx_a[t])/dt)
      Hy0 = Hy_step(dt, x, y, h, E0.copy(), Hy0.copy(), (Hy_a[t+1]-Hy_a[t])/dt)

   # plt.plot(E_tot[-4][:,-5])
   # plt.plot(E_a[-4][:,-5])
   # plt.show()
   # print(q)
   errE = np.array([np.mean(abs(E_tot[i][1:-1,1:-1] - E_a[i][1:-1,1:-1])) for i in range(len(E_tot))])
   errHx =np.array([np.mean(abs(Hx_tot[i][1:-1,:] - Hx_a[i][1:-1,:])) for i in range(len(Hx_tot))])
   errHy = np.array([np.mean(abs(Hy_tot[i][:,1:-1]- Hy_a[i][:,1:-1])) for i in range(len(Hy_tot))])
   return ((np.mean(errE) + np.mean(errHx) + np.mean(errHy)) / 3)
kx=4
ky=3
omega=math.pi*np.sqrt(kx**2+ky**2)

err=[]
ns=[ 20,25,30,35,40]

# N=[21,41,81]
for i,n in enumerate(ns):
   print(n)
   x = np.linspace(0, 1, n+1)
   y = np.linspace(0, 1, n+1)
   h = x[1]-x[0]
   time_steps=2*ns[i]+1
   T=1
   dt =T/(time_steps-1)


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