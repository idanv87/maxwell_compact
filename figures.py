import pickle


from auxilary import create_A_2
from  scheme import run_scheme

path = '/Users/idanversano/Documents/papers/compact_maxwell/data/'
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

# d44=run_scheme(constants([16,32],[0.5],[1.,2.],2,1,True,'44'))
# pickle.dump(d44, open(path + 'd44.pkl', "wb"))
# d42=run_scheme(constants([16,32,64,128,256,512],[0.5],[1.,2.],2,1,False,'42'))
# pickle.dump(d42, open(path + 'd42.pkl', "wb"))
with open(path + 'd44.pkl', 'rb') as file:
    X = pickle.load(file)
print(X['err(time)'][0][0])    

# run_scheme(constants([16,32,64,128,256,512],[0.5],[1],2,1,True,'44'))
# run_scheme(constants([16,32,64,128,256,512],[0.5],[1],2,1,False,'42'))

# run_scheme(constants([16,32,64,128,256,512],17,18,False,'figure_2_a'))
# run_scheme(constants([32]*5,2,1,False,'figure_3_a'))
# run_scheme(constants([32]*5,2,1,True,'figure_3_b'))
# run_scheme(constants([16]*1,2,1,True,'figure_try1'))


     