from datetime import datetime
now = datetime.now() # current date and time
date_time=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

import pickle
import  matplotlib.pyplot as plt
from datetime import datetime




from auxilary import create_A_2
from  scheme import run_scheme
from DRP_multiple_networks.evaluate import error_print




name44='d44_cfl_small'+date_time+'.pkl'
name42='d42_cfl_small'+date_time+'.pkl'


path = '/Users/idanversano/Documents/papers/compact_maxwell/data/'
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

def solveYee4(N_test,cfltest,T_test,kx,ky):
    n = [N_test]
    t = [1]
    cfl = [0.1]
    k1_test = [[kx]]
    k2_test = [[ky]]
    names=['Yee(4,0)']
    return error_print('time',names, n, [1], t, cfl, k1_test, k2_test, solve=True, save=(False,'fig36'))


def comparison(f):
    cfltest=[3/6/(2**0.5)]
    T_test=[1]
    N_test=[32]
    kx_test=[i for i in range(1,3)]
    ky_test=[1]

    data = {'T':[],'cfl':[],'N':[],'err':[],'err(time)':[],'conv_rates':[],'kx':[],'ky':[]}
    for kx in kx_test:
          for T in T_test:
            for cfl in cfltest:
              for i, n in enumerate(N_test):
                ky=kx
                res=f(n,cfl,T,kx,ky)
                data['T'].append(T)
                data['cfl'].append(cfl)
                data['N'].append(n)
                print(type(res))
                print(len(res))
                data['err'].append(res[0])
                data['err(time)'].append(res[1])      
                data['kx'].append(kx)
                data['ky'].append(ky)
    return data    
d4=comparison(solveC4)         
d2=comparison(solveC2)
dy=comparison(solveYee4)

    
          












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


     