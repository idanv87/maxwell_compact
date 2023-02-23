from datetime import datetime
now = datetime.now() # current date and time
date_time=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
from tqdm import tqdm

import os
import pickle
import  matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import math
import multiprocessing
from multiprocessing import pool
import timeit
import datetime
import matplotlib.patches as mpatches


from auxilary import create_A_2, conv_rate
from  scheme import run_scheme, yee_solver
from DRP_multiple_networks.evaluate import error_print







AI_h={'1':[-0.01031967, -0.08788657,  0.00514533], '4':[-0.00699426, -0.04867933, -0.00697986],'5':[ 0.00725373,-0.05299953,0.00756048]}
legend_name={'C4':'C4','C2':'C2','NC':'NC','AI':r'$AI^h$','AILH':r'$AI^l$'}
fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

class constants:
    path = '/Users/idanversano/Documents/papers/compact_maxwell/data/'
    def __init__(self, ns,cfl,T,kx,ky,fourth_order,figure):
        self.ns=ns
        self.cfl=cfl
        self.T=T
        self.kx=kx
        self.ky=ky 
        self.fourth_order=fourth_order
        self.figure=figure





def solveC4(N_test,cfltest,T_test,kx,ky):
    return run_scheme(constants(N_test,cfltest,T_test,kx,ky,True,'44'))

def solveC2(N_test,cfltest,T_test,kx,ky):
    return run_scheme(constants(N_test,cfltest,T_test,kx,ky,False,'42'))

def solveYee4(n,cfl,T,kx,ky):
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    h = x[1] - x[0]
    dt=h*cfl
    time_steps =  int(T/dt)
    return yee_solver(0,-1/24,0, omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)

def solveAI_h(n,cfl,T,kx,ky):

    beta=AI_h['1'][0]
    delta=AI_h['1'][1]
    gamma=AI_h['1'][2]
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    h = x[1] - x[0]
    dt=h*cfl
    time_steps =  int(T/dt)
    # return yee_solver(0.022726564064250554, -0.05452655828983806,  -0.004294779176336773, omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)
    return yee_solver(beta, delta,  gamma, omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)

def solveAI_l(n,cfl,T,kx,ky):
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    h = x[1] - x[0]
    dt=h*cfl
    time_steps =  int(T/dt)
    # return yee_solver(0.018264537956920218, -0.05244206290558081, -0.0026566385573771087 , omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)
    return yee_solver(0.010770443840020762, -0.05404731781456237, -0.015260532660881681 , omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)

def solveDRP(n,cfl,T,kx,ky):
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    h = x[1] - x[0]
    dt=h*cfl
    time_steps =  int(T/dt)
    return yee_solver(0., -0.06772109,  0., omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)

  


def comparison(f,name, path = '/Users/idanversano/Documents/papers/compact_maxwell/data/table1down/'):

    cfltest=[1/6/(2**0.5)]
    T_test=[1]
    N_test=[64]
    kx_test=[71]
    ky_test=[71]

    data = {'T':[],'cfl':[],'N':[],'err':[],'err(time)':[],'conv_rates':[],'kx':[],'ky':[]}
    for kx, ky in zip(kx_test, ky_test):
          for T in T_test:
            for cfl in cfltest:
              for i, n in enumerate(tqdm(N_test)):
                if True:
                    res=f(n,cfl,T,kx,ky)
                    data['T'].append(T)
                    data['cfl'].append(cfl)
                    data['N'].append(n)
                    data['err'].append(res[0])
                    data['err(time)'].append(res[1])      
                    data['kx'].append(kx)
                    data['ky'].append(ky)

    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)                
    pickle.dump(data, open(path + name, "wb"))     

    return 1


# functions=[solveC2 ,solveC4, solveYee4
# , solveAI_h,solveAI_l
# ]
# names=['C2.pkl','C4.pkl', 'NC.pkl'
# , 'AI.pkl', 'AILH.pkl'
# ]

functions=[solveC4, solveC2
, solveAI_h
]
names=['C4.pkl', 'C2.pkl'
, 'AI.pkl'
]
# [comparison(functions[i],names[i]) for i in range(len(functions))]
if 1:

    path3 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table6up/'
    path4 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table6down/'

    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)
    color={'C4':'r','C2':'g','NC':'orange', 'AI':'purple', 'AILH':'blue'}

    for i,path in enumerate([path3,path4]):
     for r, d, f in os.walk(path):
       for file in f:
         if file.endswith(".pkl"):
           name=os.path.splitext(file)[0]
           with open(path + file, 'rb') as file:
               X=pickle.load(file)
           key, *val = name[0:4].split()
            #  print(X['err(time)'][0].shape)

           ax1[i].plot(X['err(time)'][0], label=legend_name[key],c=color[name])     
              
    ax1[0].legend(loc="upper left")
    ax1[1].legend(loc="upper left")
    plt.xlabel('Time steps')
    plt.ylabel('error')
    plt.savefig(fig_path + 'table6.eps', format='eps',bbox_inches='tight')

    plt.show()

if 0:

    path3 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table6up/'
    path4 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table6down/'

    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)
    color={'C4':'r','C2':'g','NC':'orange', 'AI':'purple', 'AILH':'blue'}

    for i,path in enumerate([path3,path4]):
     for r, d, f in os.walk(path):
       for file in f:
         if file.endswith(".pkl"):
           name=os.path.splitext(file)[0]
           with open(path + file, 'rb') as file:
               X=pickle.load(file)
           key, *val = name[0:4].split()
            #  print(X['err(time)'][0].shape)
           if name in ['C2', 'C4']: 
              print(X['cfl'])
              ax1[i].plot(X['err(time)'][0], label=legend_name[key],c=color[name])     
              
    ax1[0].legend(loc="upper left")
    ax1[1].legend(loc="upper left")
    plt.xlabel('Time steps')
    plt.ylabel('error')
    plt.savefig(fig_path + 'table8.eps', format='eps',bbox_inches='tight')

    plt.show() 



if 0:
    path1 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table1up/'
    path2 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table1down/'
    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

 

    for i,path in enumerate([path1,path2]):
     for r, d, f in os.walk(path):
       for file in f:
         if file.endswith(".pkl"):
           name=os.path.splitext(file)[0]
           with open(path + file, 'rb') as file:
               X=pickle.load(file)
               print(name)
               print(conv_rate(X['N'],X['err']))
               
if 0:
    path1 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table7up/'
    path2 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table7down/'

    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)
    color={'C4':'r','C2':'g','NC':'orange', 'AI':'purple', 'AILH':'blue'}

    for i,path in enumerate([path1,path2]):
     for r, d, f in os.walk(path):
       for file in f:
         if file.endswith(".pkl"):
           name=os.path.splitext(file)[0]
           with open(path + file, 'rb') as file:
               X=pickle.load(file)
           key, *val = name[0:4].split()
            #  print(X['err(time)'][0].shape)
   
           ax1[i].plot(X['err(time)'][0], label=legend_name[key],c=color[name])     
              

    ax1[0].legend(loc="upper left")
    ax1[1].legend(loc="upper left")
    plt.xlabel('Time steps')
    plt.ylabel('error')
    plt.savefig(fig_path + 'table7.eps', format='eps',bbox_inches='tight')

    plt.show()  

if 0:
    path1 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table7up/'
    path2 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table7down/'

    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)
    color={'C4':'r','C2':'g','NC':'orange', 'AI':'purple', 'AILH':'blue'}

    for i,path in enumerate([path1,path2]):
     for r, d, f in os.walk(path):
       for file in f:
         if file.endswith(".pkl"):
           name=os.path.splitext(file)[0]
           with open(path + file, 'rb') as file:
               X=pickle.load(file)
           key, *val = name[0:4].split()
            #  print(X['err(time)'][0].shape)
           if name in ['C2', 'C4', 'NC']: 
              ax1[i].plot(X['err(time)'][0], label=legend_name[key],c=color[name])     
              

    ax1[0].legend(loc="upper left")
    ax1[1].legend(loc="upper left")
    plt.xlabel('Time steps')
    plt.ylabel('error')
    plt.savefig(fig_path + 'table9.eps', format='eps',bbox_inches='tight')

    plt.show()  

if 0:

    path3 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table1up/'
    path4 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table1down/'

    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)
    color={'C4':'r','C2':'g','NC':'orange', 'AI':'purple', 'AILH':'blue'}

    for i,path in enumerate([path3,path4]):
     for r, d, f in os.walk(path):
       for file in f:
         if file.endswith(".pkl"):
           name=os.path.splitext(file)[0]
           with open(path + file, 'rb') as file:
               X=pickle.load(file)
           key, *val = name[0:4].split()
            #  print(X['err(time)'][0].shape)
           if name in ['C2', 'C4', 'AI']: 
              print(X['cfl'])
              ax1[i].plot(X['err(time)'][0], label=legend_name[key],c=color[name])     
              
    ax1[0].legend(loc="upper left")
    ax1[1].legend(loc="upper left")
    plt.xlabel('Time steps')
    plt.ylabel('error')
    plt.savefig(fig_path + 'table10.eps', format='eps',bbox_inches='tight')

    plt.show() 

 


if False:
    path1 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table4up/'
    path2 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table4down/'
    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)

    for r, d, f in os.walk(path1):
      for file in f:
        if file.endswith(".pkl"):
          name=os.path.splitext(file)[0]
          with open(path1 + file, 'rb') as file:
               X=pickle.load(file)
          if name[0:3]!='AI': 
             key, *val = name[0:3].split()
             ax1[0].plot(X['T'],X['err'], label=legend_name[key])  

   
    for r, d, f in os.walk(path2):
      for file in f:
        if file.endswith(".pkl"):
          name=os.path.splitext(file)[0]
          with open(path1 + file, 'rb') as file:
               X=pickle.load(file)
          if name[0:3]!='AI':      
             key, *val = name[0:3].split()
             ax1[1].plot(X['T'],X['err'], label=legend_name[key])  

    plt.savefig(fig_path + 'table4.eps', format='eps',bbox_inches='tight')
    ax1[0].legend()
    ax1[1].legend()
    plt.xlabel(r'$T$')
    plt.ylabel('error')
    plt.show()  

if False:
    path1 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table3up/'
    path2 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table3down/'
    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)

    for r, d, f in os.walk(path1):
      for file in f:
        if file.endswith(".pkl"):
          name=os.path.splitext(file)[0]
          with open(path1 + file, 'rb') as file:
               X=pickle.load(file)
          if name[0:3]!='AI':       
             key, *val = name[0:3].split()
             ax1[0].plot(X['cfl'][:-1],X['err'][:-1], label=legend_name[key])  

   
    for r, d, f in os.walk(path2):
      for file in f:
        if file.endswith(".pkl"):
          name=os.path.splitext(file)[0]
          with open(path1 + file, 'rb') as file:
               X=pickle.load(file)
          if name[0:3] !='AI':      
             key, *val = name[0:3].split()
             ax1[1].plot(X['cfl'][:-1],X['err'][:-1], label=legend_name[key])  

    plt.savefig(fig_path + 'table3.eps', format='eps',bbox_inches='tight')
    ax1[0].legend()
    ax1[1].legend()
    plt.xlabel(r'$r$')
    plt.ylabel('error')
    plt.show()  


if False:
    path = '/Users/idanversano/Documents/papers/compact_maxwell/data/table2/'
    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'
    name4='d442023_02_19-09_45_48_PM.pkl'
    name2='d422023_02_19-09_45_48_PM.pkl'
    namey4='dyee42023_02_19-10_38_45_PM.pkl'
    nameai='dai2023_02_19-10_38_45_PM.pkl'

    with open(path + name4, 'rb') as file:
        X4 = pickle.load(file)  
    with open(path + name2, 'rb') as file:
        X2 = pickle.load(file)    
    with open(path + namey4, 'rb') as file:
        Xy = pickle.load(file)    
    with open(path + nameai, 'rb') as file:
        Xai = pickle.load(file)    
    print(X4['err'])
    fig, ax1 = plt.subplots(1, sharex=False, sharey=False)
    ax1.plot(X4['kx'],X4['err'], label='C(4,4)')
    ax1.plot(X2['kx'],X2['err'],label='C(4,2)')
    ax1.plot(Xy['kx'],Xy['err'], label='NC')
    ax1.plot(Xai['kx'],Xai['err'],label='AI')

    print(' saved as:dispersion_figure'  + '.eps')
    plt.savefig(fig_path + 'table2.eps', format='eps',bbox_inches='tight')
    plt.legend(loc="upper left")
    plt.xlabel(r'$k$')
    plt.ylabel('error')
    plt.show()  





#
# dr_calculator(names, save=(True,'fig0001'))
# print(q)

    
    # return run_scheme(constants(N_test,cfltest,T_test,kx,ky,True,'44'))
# d44=run_scheme(constants(N_test,cfltest,T_test,kx,ky,True,'44'))
# pickle.dump(d44, open(path + name44, "wb"))
# d42=run_scheme(constants(N_test,cfltest,T_test,kx,ky,False,'42'))
# pickle.dump(d42, open(path + name42, "wb"))
# print('')
# print(d42['err'])    
# print('')
# print(d44['err'])  
if False:
    path = '/Users/idanversano/Documents/papers/compact_maxwell/data/'
    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'
    name44='d44_cfl_small2023_02_16-11_01_04_AM.pkl'
    name42='d42_cfl_small2023_02_16-11_01_04_AM.pkl'
    with open(path + name44, 'rb') as file:
        X44 = pickle.load(file)  
    with open(path + name42, 'rb') as file:
        X42 = pickle.load(file)    

    fig, ax1 = plt.subplots(1, sharex=False, sharey=False)
    ax1.plot(X44['kx'],X44['err'], label='C(4,4)')
    ax1.plot(X42['kx'],X42['err'],label='C(4,2)')

    print(' saved as:dispersion_figure'  + '.eps')
    plt.savefig(fig_path + 'err(k)_figure.eps', format='eps',bbox_inches='tight')
    plt.legend(loc="upper left")
    plt.xlabel(r'$k$')
    plt.ylabel('error')
    plt.show()    
if 0:
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
        # ax1.scatter(x[0], y[j], color='black', s=100, marker='s')
        # ax1.scatter(x[-1], y[j], color='black', s=100, marker='s')
        # ax1.scatter(x[i], y[0], color='black', s=100, marker='s')
        # ax1.scatter(x[i], y[-1], color='black', s=100, marker='s')

    for i in range(len(x1)):
      for j in range(len(y1)):
        ax1.scatter(x1[i], y1[j], color='blue', s=100,marker='8')
        # ax1.scatter(x1[0], y1[j], color='blue', s=5, marker='s')

        # ax1.scatter(x1[i], y1[j], color='blue', s=100,marker='x')
        # ax1.scatter(x1[0], y1[j], color='blue', s=100, marker='s')
        # ax1.scatter(x1[-1], y1[j], color='blue', s=100, marker='s')
        # ax1.scatter(x1[i], y1[0], color='blue', s=100, marker='s')
        # ax1.scatter(x1[i], y1[-1], color='blue', s=100, marker='s')

    for i in range(len(x2)):
      for j in range(len(y2)):
        ax1.scatter(x2[i], y2[j], color='red', s=100, marker='*')

    #       ax1.scatter(x2[i], y2[j], color='red', s=100,marker='x')
    #       ax1.scatter(x2[0], y2[j], color='red', s=100, marker='s')
    #       ax1.scatter(x2[-1], y2[j], color='red', s=100, marker='s')
    #       ax1.scatter(x2[i], y2[0], color='red', s=100, marker='s')
    #       ax1.scatter(x2[i], y2[-1], color='red', s=100, marker='s')


    black_patch = mpatches.Patch(color='black', label='E_3')
    blue_patch = mpatches.Patch(color='blue', label='H_1')
    red_patch = mpatches.Patch(color='red', label='H_2')


    ax1.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5,
            handles=[black_patch,blue_patch,red_patch])

    box = ax1.get_position()
    # ax1.title.set_text('')
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 1])
    ax1.grid()
    # Put a legend below current axis
    # plt.grid()
    PATH = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'
    # plt.savefig(PATH + 'omega.eps' , format='eps',
    #            bbox_inches='tight')

    plt.show()


# print(X['N'])
# print(X['err'])    
# print(X['cfl'])
# print(X['T'])

# run_scheme(constants([16,32,64,128,256,512],[0.5],[1],2,1,True,'44'))
# run_scheme(constants([16,32,64,128,256,512],[0.5],[1],2,1,False,'42'))

# run_scheme(constants([16,32,64,128,256,512],17,18,False,'figure_2_a'))
# run_scheme(constants([32]*5,2,1,False,'figure_3_a'))
# run_scheme(constants([32]*5,2,1,True,'figure_3_b'))
# run_scheme(constants([16]*1,2,1,True,'figure_try1'))


     