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


n=5
h=1/n
x = np.linspace(0, 1, n + 1)
y = np.linspace(0, 1, n + 1)

x1 = np.linspace(0, 1, n + 1)
y1 = np.linspace(h/2, 1+h/2, n + 1)[:-1]

x2 = np.linspace(h/2, 1+h/2, n + 1)[:-1]
y2 = np.linspace(0, 1, n + 1)
f=plt.figure()
fig, ax1 = plt.subplots(1, sharex=False, sharey=False)




for i in range(len(x)):
   for j in range(len(y)):
       # ax1.scatter(x[i],y[j],color='black',s=5)
       ax1.scatter(x[i], y[j], color='black', s=100, marker='x')
       ax1.scatter(x[0], y[j], color='black', s=100, marker='s')
       ax1.scatter(x[-1], y[j], color='black', s=100, marker='s')
       ax1.scatter(x[i], y[0], color='black', s=100, marker='s')
       ax1.scatter(x[i], y[-1], color='black', s=100, marker='s')

for i in range(len(x1)):
   for j in range(len(y1)):
      ax1.scatter(x1[i], y1[j], color='blue', s=5,marker='s')
      # ax1.scatter(x1[0], y1[j], color='blue', s=5, marker='s')

#       ax1.scatter(x1[i], y1[j], color='blue', s=100,marker='x')
#       ax1.scatter(x1[0], y1[j], color='blue', s=100, marker='s')
#       ax1.scatter(x1[-1], y1[j], color='blue', s=100, marker='s')
#       ax1.scatter(x1[i], y1[0], color='blue', s=100, marker='s')
#       ax1.scatter(x1[i], y1[-1], color='blue', s=100, marker='s')

for i in range(len(x2)):
   for j in range(len(y2)):
       ax1.scatter(x2[i], y2[j], color='red', s=5, marker='s')

#       ax1.scatter(x2[i], y2[j], color='red', s=100,marker='x')
#       ax1.scatter(x2[0], y2[j], color='red', s=100, marker='s')
#       ax1.scatter(x2[-1], y2[j], color='red', s=100, marker='s')
#       ax1.scatter(x2[i], y2[0], color='red', s=100, marker='s')
#       ax1.scatter(x2[i], y2[-1], color='red', s=100, marker='s')


black_patch = mpatches.Patch(color='black', label='E_z')
blue_patch = mpatches.Patch(color='blue', label='H_x')
red_patch = mpatches.Patch(color='red', label='H_y')
# ax1.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5,
#            handles=[black_patch,blue_patch])

# ax1.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5,
#            handles=[black_patch,red_patch])

ax1.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5,
           handles=[black_patch,blue_patch,red_patch])

box = ax1.get_position()
# ax1.title.set_text('')
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 1])

# Put a legend below current axis
# plt.grid()
PATH = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'
# plt.savefig(PATH + 'Ez.eps' , format='eps',
#            bbox_inches='tight')

plt.show()
print(q)
#
#
# X1,Y1=np.meshgrid(x,y,indexing='ij')
# plt.scatter(X1,Y1)
# y = np.linspace(h, 1+h, n + 1)
# X,Y=np.meshgrid(x,y,indexing='ij')
# # plt.scatter(X,Y)
#

