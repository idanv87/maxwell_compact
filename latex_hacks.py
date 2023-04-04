from tabulate import tabulate
from texttable import Texttable
import latextable

def tex_table(tex_path, headers,data, name):
    rows=[]
    rows.append(headers)
    for dat in data:
        rows.append(dat)
    table = Texttable()
    table.set_cols_align(["c"] * len(headers))
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)    
    with open(tex_path+name, 'w') as f:
      f.write(latextable.draw_latex(table, caption=""))


