from tabulate import tabulate
from texttable import Texttable
import latextable


def tex_table(tex_path, headers,data, name):
    rows=[]
    rows.append(headers)
    for dat in data:
        rows.append(dat)
    table = Texttable(0)
    table.set_precision(1)
    table.set_cols_dtype(['e' for i in range(len(rows[0]))])
    table.set_cols_align(["c"] * len(headers))
    table.set_deco(Texttable.HEADER | Texttable.VLINES )
    table.add_rows(rows)  
    # print(table.draw())
    
    

    # print(latextable.draw_latex(table, caption=""))
   
    with open(tex_path+name, 'w') as f:
      f.write(latextable.draw_latex(table, caption="", alias={'.':'.', 'e':'$e$', '-':'-'}))



   
   
   