""" SXTPY_NombreCuaderno.ipynb - PRUEBAS """

# LIBRERIAS ================================================
# Básicas
import numpy as np      
import pandas as pd
from IPython.display import display, HTML
from tqdm.notebook import tqdm


# Soluciones
from soluciones import fun_solucion

# Otras


PLINE = '__________________________________'


def pretty_print(df):
    """ Muestra tabla con IPython.display """
    if isinstance(df, dict): df = pd.DataFrame(df)
    return display(HTML(df.to_html()))


# 1) ================================================
def test_fun(fun):
    """
    Prueba función fun.
    Interrumpe ejecición en caso de error    
    """
    pass
