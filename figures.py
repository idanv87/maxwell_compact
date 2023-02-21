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




from auxilary import create_A_2
from  scheme import run_scheme, yee_solver
from DRP_multiple_networks.evaluate import error_print








legend_name={'C4':'C4','C2':'C2','NC':'NC','AI':'AI','DRP':'DRP'}
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

def solveAI(n,cfl,T,kx,ky):
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    h = x[1] - x[0]
    dt=h*cfl
    time_steps =  int(T/dt)
    return yee_solver(-0.00841542, -0.09234213,  0.01128501, omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)

def solveAI_lh(n,cfl,T,kx,ky):
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    h = x[1] - x[0]
    dt=h*cfl
    time_steps =  int(T/dt)
    return yee_solver(0.00349403, -0.08069799, -0.01218202 , omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)
    # [ 0.00349403 -0.08069799 -0.01218202]  low+high trained

def solveDRP(n,cfl,T,kx,ky):
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    h = x[1] - x[0]
    dt=h*cfl
    time_steps =  int(T/dt)
    return yee_solver(0., -0.06772109,  0., omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)

  


def comparison(f,name, path = '/Users/idanversano/Documents/papers/compact_maxwell/data/table2down/'):

    cfltest=[3/6/(2**0.5)]
    T_test=[1,10,100]
    N_test=[16,32,64,128,256]
    kx_test=[8, 16,32,64,128]
    ky_test=[1]

    data = {'T':[],'cfl':[],'N':[],'err':[],'err(time)':[],'conv_rates':[],'kx':[],'ky':[]}
    for kx in kx_test:
          for T in T_test:
            for cfl in cfltest:
              for i, n in enumerate(N_test):
                ky=kx
                if int(n)/int(kx)==2:
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


functions=[solveC2, solveC4, solveYee4, solveAI,solveAI_lh]
names=['C2.pkl','C4.pkl', 'NC.pkl', 'AI.pkl', 'AILH.pkl']

# namedrp='DRP.pkl'

if __name__ == '__main__':
    jobs = []
    for f, name in  zip(functions, names):
        p = multiprocessing.Process(target=comparison, args=(f,name))
        jobs.append(p)
        p.start()

    result = []
    for proc in jobs:
        proc.join()


# d4=comparison(solveC4)    
# pickle.dump(d4, open(path + name4, "wb"))     
# d2=comparison(solveC2)
# pickle.dump(d2, open(path + name2, "wb"))
# dyee=comparison(solveYee4)
# pickle.dump(dyee, open(path + namey4, "wb"))
# dai=comparison(solveAI) 
# pickle.dump(dai, open(path + nameai, "wb"))
# dailh=comparison(solveAI_lh) 
# pickle.dump(dailh, open(path + nameailh, "wb"))
# ddrp=comparison(solveDRP) 
# print(ddrp['kx'])
# pickle.dump(ddrp, open(path + namedrp, "wb"))



if False:
    path1 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table6/'
    path2 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table6_64/'
    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)

    for r, d, f in os.walk(path1):
      for file in f:
        if file.endswith(".pkl"):
          name=os.path.splitext(file)[0]
          with open(path1 + file, 'rb') as file:
               X=pickle.load(file)
               print(X['kx'])
          if True:
             key, *val = name[0:4].split()
             ax1[0].plot(X['kx'][0::2],X['err'][0::2], label=legend_name[key])  
             

   
    # for r, d, f in os.walk(path2):
    #   for file in f:
    #     if file.endswith(".pkl"):
    #       name=os.path.splitext(file)[0]
    #       with open(path1 + file, 'rb') as file:
    #            X=pickle.load(file)
    #            print(X['kx'])
    #       if True:      
    #          key, *val = name[0:4].split()
             
    #          ax1[1].plot(X['kx'][0::2],X['err'][0::2], label=legend_name[key])  

    plt.savefig(fig_path + 'table6.eps', format='eps',bbox_inches='tight')
    ax1[0].legend()
    ax1[1].legend()
    plt.xlabel(r'$k$')
    plt.ylabel('error')
    plt.show()  

if False:
    path1 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table5up/'
    path2 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table5down/'
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
          if name[0:3] not in ['NC','AI']:      
             key, *val = name[0:3].split()
             ax1[1].plot(X['T'],X['err'], label=legend_name[key])  

    plt.savefig(fig_path + 'table5.eps', format='eps',bbox_inches='tight')
    ax1[0].legend()
    ax1[1].legend()
    plt.xlabel(r'$T$')
    plt.ylabel('error')
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


     