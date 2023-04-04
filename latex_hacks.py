from tabulate import tabulate
from texttable import Texttable
import latextable
import sys
import  matplotlib.pyplot as plt
import os
import pickle
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
# sys.path.append('/foo/bar/mock-0.3.1')

# from testcase import TestCase
# from testutils import RunTests
# from mock import Mock, sentinel, patch



def tex_table(tex_path, headers,data, name,dt=1):
    rows=[]
    rows.append(headers)
    for dat in data:
        rows.append(dat)
    table = Texttable(0)
    if dt:
        table.set_precision(2)
        cols_type=['e' for i in range(len(rows[0]))]
        cols_type[0]='i'
    else:
        table.set_precision(2)
        cols_type=['f' for i in range(len(rows[0]))]
        cols_type[0]='t'

    table.set_cols_dtype(cols_type)
    # table.set_rows_dtype(['e' for i in range(len(rows[0]))])
    table.set_cols_align(["c"] * len(headers))
    table.set_deco(Texttable.HEADER | Texttable.VLINES )
    table.add_rows(rows)  
    with open(tex_path+name, 'w') as f:
      f.write(latextable.draw_latex(table, caption=""))
    print(table.draw())
    
    

    # print(latextable.draw_latex(table, caption=""))
   




   
   
   