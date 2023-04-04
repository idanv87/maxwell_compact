from datetime import datetime
now = datetime.now() # current date and time
date_time=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
from tqdm import tqdm
from decimal import Decimal

from latex_hacks import tex_table
import os
import pickle
import  matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import math

import timeit
import datetime
import matplotlib.patches as mpatches


from auxilary import create_A_2, conv_rate
from  scheme import run_scheme, yee_solver
from DRP_multiple_networks.evaluate import error_print

print(os.getcwd())
# x=np.linspace(0,1/2**0.5,20)
# plt.plot(np.cos(math.pi*2**0.5*2*x))
# plt.show()
AI_h={'1':[-0.01031967, -0.08788657,  0.00514533],
      '2':[0.030110106445778904,-0.19134659869216905,0.05624205921198148],
      '3':[0.016091811496056083, -0.14807714640318745, 0.045012759482413504],
       '4':[-0.00699426, -0.04867933, -0.00697986],'5':[ 0.00725373,-0.05299953,0.00756048]}
       
legend_name={'C4':'C4','NC':'NC','AI':r'$AI^h$'}
line_s={'C4':'solid','NC':'solid','AI':'solid'}
fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'
tex_path='/Users/idanversano/Documents/papers/compact_maxwell/'
table_path='/Users/idanversano/Documents/papers/compact_maxwell/tables/'
data_path='/Users/idanversano/Documents/papers/compact_maxwell/data/'

def plot_table(x_name,y_name,data_path1,data_path2, fig_save, title):
    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)
    color={'C4':'r','NC':'orange', 'AI':'purple', 'AILH':'blue'}

    for i,path in enumerate([data_path1,data_path2]):
     for r, d, f in os.walk(path):
       for file in f:
         if file.endswith(".pkl"):
           name=os.path.splitext(file)[0]
           with open(path + file, 'rb') as file:
               X=pickle.load(file)
           key, *val = name[0:4].split()
            #  print(X['err(time)'][0].shape)
           if name in list(legend_name):
             ax1[i].plot(X[x_name], X[y_name], label=legend_name[key],c=color[name], linestyle=line_s[name]) 
             ax1[i].set_xlabel(x_name)
             ax1[i].set_ylabel(y_name)
             ax1[i].set_title(title[i])    
              
    ax1[0].legend(loc="upper left")
    ax1[1].legend(loc="upper left")
    fig.tight_layout(pad=1.0)
    # plt.xlabel(x_name)
    # plt.ylabel(y_name)
    plt.savefig(fig_save, format='eps',bbox_inches='tight')
    # plt.savefig(fig_path + 'table7.eps', format='eps',bbox_inches='tight')
    # plt.show(block=False)
    # plt.pause(3)
    # plt.close()
    plt.show()

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


def solveN(N_test,cfltest,T_test,kx,ky):
    return run_scheme(constants(N_test,cfltest,T_test,kx,ky,'N','N'))

def solveYee4(n,cfl,T,kx,ky):
    omega = math.pi * np.sqrt(kx ** 2 + ky ** 2)
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    h = x[1] - x[0]
    dt=h*cfl
    time_steps =  int(T/dt)
    return yee_solver(0,-1/24,0, omega,kx,ky, dt, x, y, h, time_steps, cfl,n, T)

def solveAI_h(n,cfl,T,kx,ky):
    num=str(int(cfl*6*2**0.5))


    beta=AI_h[num][0]
    delta=AI_h[num][1]
    gamma=AI_h[num][2]
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





def comparison(f,name, path, cfltest, T_test, N_test, kx_test, ky_test ):
    # cfltest,T_test,N_test,kx_test,ky_test


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

functions=[solveN
             ,solveAI_h, solveYee4]

names=['C4.pkl'
         ,'AI.pkl', 'NC.pkl']


def exp_conv_rates( kwargs, table_name):
  path = '/Users/idanversano/Documents/papers/compact_maxwell/data/temp/'
  parameters={'h': kwargs['N_test'][1:]}
  if True:
    [comparison(functions[i],names[i], path, **kwargs) for i in range(len(functions))]
    data=[]
    headers=[]
    for i,path in enumerate([path]):
       for r, d, f in os.walk(path):
         for file in f:
           if file.endswith(".pkl"):
             name=os.path.splitext(file)[0]
             with open(path + file, 'rb') as file:
              if name in list(legend_name):
                 X=pickle.load(file)
                 headers.append(name)
                 data.append(conv_rate(X['N'],X['err']))
              #  print(name)
              #  print(conv_rate(X['N'],X['err']))
    par_data=[parameters[name] for name in list(parameters)]
    data=par_data+data
    data=np.array(data).T.tolist()
  
    headers=list(parameters)+headers
    tex_table(table_path, headers,data, table_name)
    
    


def exp_cfl( kwargs, table_name):
  path = '/Users/idanversano/Documents/papers/compact_maxwell/data/temp/'
  parameters={'cfl':kwargs['cfltest']}
  # parameters={'cfl':[r'$1$',r'$1$']}
  if True:
    [comparison(functions[i],names[i], path, **kwargs) for i in range(len(functions))]
    data=[]
    headers=[]
    for i,path in enumerate([path]):
       for r, d, f in os.walk(path):
         for file in f:
           if file.endswith(".pkl"):
             name=os.path.splitext(file)[0]
             with open(path + file, 'rb') as file:
              if name in list(legend_name):
                 X=pickle.load(file)
                 headers.append(name)
                 data.append([f"{Decimal(x):.2e}" for x in X['err']])
                 
    par_data=[parameters[name] for name in list(parameters)]
    data=par_data+data
    data=np.array(data).T.tolist()
    headers=list(parameters)+headers
    tex_table(table_path, headers,data, table_name)



def exp_erros( kwargs, table_name, path):
  parameters={'k':kwargs['kx_test']}
  # parameters={'cfl':[r'$1$',r'$1$']}
  if True:
    [comparison(functions[i],names[i], path, **kwargs) for i in range(len(functions))]
    data=[]
    headers=[]
    for i,path in enumerate([path]):
       for r, d, f in os.walk(path):
         for file in f:
           if file.endswith(".pkl"):
             name=os.path.splitext(file)[0]
             with open(path + file, 'rb') as file:
              if name in list(legend_name):
                 X=pickle.load(file)
                 headers.append(name)
                 data.append(X['err'])           
    par_data=[parameters[name] for name in list(parameters)]
    data=par_data+data
    print(data)
    data=np.array(data).T.tolist()
    headers=list(parameters)+headers
    tex_table(table_path, headers,data, table_name)

T_test=2*2**0.5
exp_conv_rates({'cfltest':[5/6/2**0.5], 'T_test':[T_test],'N_test':[16,32,64,128,256,512],'kx_test':[1],
            'ky_test':[1] }, table_name='conv_rates_low.tex')
exp_conv_rates({'cfltest':[5/6/2**0.5], 'T_test':[T_test],'N_test':[16,32,64,128,256,512],'kx_test':[21],
            'ky_test':[21] }, table_name='conv_rates_high.tex')

exp_cfl({'cfltest':[1/6/2**0.5,2/6/2**0.5,3/6/2**0.5,4/6/2**0.5,5/6/2**0.5], 'T_test':[T_test],'N_test':[64],'kx_test':[1],
            'ky_test':[1] }, table_name='cfl_high.tex')
exp_cfl({'cfltest':[1/6/2**0.5,2/6/2**0.5,3/6/2**0.5,4/6/2**0.5,5/6/2**0.5], 'T_test':[T_test],'N_test':[64],'kx_test':[21],
            'ky_test':[21] }, table_name='cfl_low.tex')

exp_erros({'cfltest':[1/6/2**0.5], 'T_test':[T_test],'N_test':[64],'kx_test':[i for i in range(1,50)],
            'ky_test':[i for i in range(1,50)] }, table_name='error(k)1.tex', path= data_path+'cfl1/')
exp_erros({'cfltest':[5/6/2**0.5], 'T_test':[T_test],'N_test':[64],'kx_test':[i for i in range(1,50)],
            'ky_test':[i for i in range(50)]}, table_name='error(k)5.tex', path=data_path+'cfl5/')

plot_table('kx','err',data_path+'cfl1/',data_path+'cfl5/', fig_path+'error(k).eps',
           title=['CFL='+r'$\frac{1}{6\sqrt{2}}$', 'CFL='+r'$\frac{5}{6\sqrt{2}}$']) 







# plot_table('kx','err',data_path+'low2/',data_path+'high2/', fig_path+'error(k)2.eps',
#             title='CFL='+r'$\frac{5}{6\sqrt{2}}$') 

# exp_erros({'cfltest':[1/6/2**0.5], 'T_test':[T_test],'N_test':[64],'kx_test':[i for i in range(50)],
#             'ky_test':[i for i in range(50)] }, table_name='error(k)_high1.tex', path=
#               data_path+'high1/')

# exp_erros({'cfltest':[5/6/2**0.5], 'T_test':[T_test],'N_test':[16],'kx_test':[1,2,3],
#             'ky_test':[1,2,3] }, table_name='error(k)_low2.tex', path= data_path+'low2/')
                                 
if 0:

    # path3 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table6up/'
    # path4 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table6down/'
    path3 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table7up/'
    path4 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table7down/'

    fig_path = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    fig, ax1 = plt.subplots(2, sharex=False, sharey=False)
    color={'C4':'r','NC':'orange', 'AI':'purple', 'AILH':'blue'}

    for i,path in enumerate([path3,path4]):
     for r, d, f in os.walk(path):
       for file in f:
         if file.endswith(".pkl"):
           name=os.path.splitext(file)[0]
           with open(path + file, 'rb') as file:
               X=pickle.load(file)
           key, *val = name[0:4].split()
            #  print(X['err(time)'][0].shape)
           if name in list(legend_name):
             ax1[i].plot(X['err(time)'][0], label=legend_name[key],c=color[name], linestyle=line_s[name]) 
             ax1[i].set_title('kx=ky='+str(X['kx'][0]))    
              
    ax1[0].legend(loc="upper left")
    ax1[1].legend(loc="upper left")
    fig.tight_layout(pad=1.0)
    plt.xlabel('Time steps')
    plt.ylabel('error')
    # plt.savefig(fig_path + 'table6.eps', format='eps',bbox_inches='tight')
    # plt.savefig(fig_path + 'table7.eps', format='eps',bbox_inches='tight')
    # plt.show(block=False)
    # plt.pause(3)
    # plt.close()
    plt.show()
    

if 0:

    # path3 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table8up/'
    # path4 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table8down/'
    path3 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table9up/'
    path4 = '/Users/idanversano/Documents/papers/compact_maxwell/data/table9down/'

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
           if name in list(legend_name): 
          #  if (i==1 and name in list(legend_name)) or ((i==0) and (name in list(legend_name) and name not in ['AI'])): 
              print(X['cfl'])
              ax1[i].plot(X['err(time)'][0], label=legend_name[key],c=color[name], linestyle=line_s[name])     
              ax1[i].set_title('kx=ky='+str(X['kx'][0])) 

    ax1[0].legend(loc="upper left")
    ax1[1].legend(loc="upper left")
    fig.tight_layout(pad=1.0)
    plt.xlabel('Time steps')
    plt.ylabel('error')
    # plt.savefig(fig_path + 'table9.eps', format='eps',bbox_inches='tight')
    # plt.savefig(fig_path + 'table8.eps', format='eps',bbox_inches='tight')

    # plt.show(block=False)
    # plt.pause(3)
    # plt.close()
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
    from matplotlib.legend_handler import HandlerBase

    list_color  = ["black", "blue", "red"]
    list_mak    = ["x","8","*"]
    list_lab    = ['E_3','H_1','H_2']
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
    s=70
    for i in range(len(x2)):
      for j in range(len(y2)):
        ax1.scatter(x2[i], y2[j], color='red', s=s, marker='*')

    #       ax1.scatter(x2[i], y2[j], color='red', s=100,marker='x')
    #       ax1.scatter(x2[0], y2[j], color='red', s=100, marker='s')
    #       ax1.scatter(x2[-1], y2[j], color='red', s=100, marker='s')
    #       ax1.scatter(x2[i], y2[0], color='red', s=100, marker='s')
    #       ax1.scatter(x2[i], y2[-1], color='red', s=100, marker='s')


    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                          marker=tup[1],color=tup[0], transform=trans)]



   
    ax1.legend(list(zip(list_color,list_mak)), list_lab, 
              handler_map={tuple:MarkerHandler()}, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5) 
    ax1.grid()
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #       fancybox=True, shadow=True, ncol=5)
    # Put a legend below current axis
    # plt.grid()
    PATH = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'
    # plt.savefig(PATH + 'omega.eps' , format='eps',
    #            bbox_inches='tight')

    plt.show()

if 0:
    n = 5
    h = 1/n
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)

    x1 = np.linspace(0, 1, n + 1)
    y1 = np.linspace(h/2, 1+h/2, n + 1)[:-1]

    x2 = np.linspace(h/2, 1+h/2, n + 1)[:-1]
    y2 = np.linspace(0, 1, n + 1)
    f = plt.figure()
    fig, (ax1,ax2) = plt.subplots(1,2, sharex=True, sharey=True)
  
    style='dashed'
    ax1.plot(x,x*0,color='black', linestyle=style)
    ax1.plot(x,x*0+1,color='black', linestyle=style)
    ax1.plot(x*0,x,color='black', linestyle=style)
    ax1.plot(x*0+1,x,color='black', linestyle=style)
    s1=100
    s2=100
    for i in range(len(x1)):
      for j in range(len(y1)):
        ax1.scatter(x1[i], y1[j], color='blue', s=s1, marker='o')

    ax1.scatter(x1[1:-1], x1[1:-1]*0-h/2, color='blue', s=s1, marker='o', facecolors='none')
    ax1.scatter(x1[1:-1], x1[1:-1]*0+1+h/2, color='blue', s=s1, marker='o', facecolors='none')
    ax1.axis('off')
    ax1.set_aspect(1)


    x1 = np.linspace(h/2, 1+h/2, n + 1)[:-1]
    y1 = np.linspace(0, 1, n + 1)
    
    style='dashed'
    ax2.plot(x,x*0,color='black', linestyle=style)
    ax2.plot(x,x*0+1,color='black', linestyle=style)
    ax2.plot(x*0,x,color='black', linestyle=style)
    ax2.plot(x*0+1,x,color='black', linestyle=style)
    for i in range(len(x1)):
      for j in range(len(y1)):
         ax2.scatter(x1[i], y1[j], color='red', s=s2, marker='*')

    ax2.scatter( y1[1:-1]*0-h/2,y1[1:-1], color='red', s=s2, marker='*',facecolors='none')
    ax2.scatter(y1[1:-1]*0+1+h/2,y1[1:-1] , color='red', s=s2, marker='*', facecolors='none')
    ax2.axis('off')

    # ax1.set_ylim(-h/2,1+h/2)
    # ax1.set_xlim(0,1)  
    ax2.set_aspect(1)
    PATH = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'
    size=10
    ax1.text(0.5, 1, 'Neumann', dict(size=size))
    ax1.text(0.5, -0.05, 'Neumann', dict(size=size))
    ax1.text(-0.15, 0.6, 'Dir.', dict(size=size))
    ax1.text(1.05, 0.6, 'Dir.', dict(size=size))
    ax2.text(0.5, 1.05, 'Dir', dict(size=size))
    ax2.text(0.5, -0.1, 'Dir', dict(size=size))
    ax2.text(-0.1, 0.5, 'Neumann.', dict(size=size))
    ax2.text(0.9, 0.5, 'Neumann', dict(size=size))
    plt.savefig(PATH + 'omega4.eps' , format='eps',
               bbox_inches='tight')
    plt.show()


   

if 0:
    n = 5
    h = 1/n
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)


    x1 = np.linspace(h/2, 1+h/2, n + 1)[:-1]
    y1 = np.linspace(0, 1, n + 1)
    f = plt.figure()
    fig, ax1 = plt.subplots(1, sharex=False, sharey=False)
    style='dashed'
    ax1.plot(x,x*0,color='black', linestyle=style)
    ax1.plot(x,x*0+1,color='black', linestyle=style)
    ax1.plot(x*0,x,color='black', linestyle=style)
    ax1.plot(x*0+1,x,color='black', linestyle=style)
    s=70
    for i in range(len(x1)):
      for j in range(len(y1)):

        # ax1.scatter(x[i],y[j],color='black',s=5)
        ax1.scatter(x1[i], y1[j], color='red', s=100, marker='*')

    ax1.scatter( y1[1:-1]*0-h/2,y1[1:-1], color='red', s=100, marker='*',facecolors='none')
    ax1.scatter(y1[1:-1]*0+1+h/2,y1[1:-1] , color='red', s=100, marker='*', facecolors='none')
    ax1.axis('off')

    # ax1.set_ylim(-h/2,1+h/2)
    # ax1.set_xlim(0,1)  
    ax1.set_aspect('equal', adjustable='box')
    PATH = '/Users/idanversano/Documents/papers/compact_maxwell/figures/'

    size=10
    plt.text(0.5, 1.05, 'Dir', dict(size=size))
    plt.text(0.5, -0.1, 'Dir', dict(size=size))
    plt.text(-0.1, 0.5, 'Neumann.', dict(size=size))
    plt.text(0.9, 0.5, 'Neumann', dict(size=size))
    plt.savefig(PATH + 'omega_3.eps' , format='eps',
               bbox_inches='tight')
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


     