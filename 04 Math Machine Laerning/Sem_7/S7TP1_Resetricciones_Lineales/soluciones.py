""" SXTPY_NombreCuaderno.ipynb - SOLUCIONES """

# LIBRERIAS ================================================
# Básicas
import numpy as np
from tqdm.notebook import tqdm
from copy import copy


import plotly.colors as pcolors
import plotly.graph_objects as go

def fun_solucion(args):
    """ 
    Descripcion_función
    ___________________________________
    Entrada:
    args: [type] Descripción_parámetro
    ___________________________________
    Salida:
    out : [type] Descripción_salida
    """
    # Desarrollo_ previo
    # =====================================================
    # Solución
    out = 0
    # =====================================================
    return out



# 1) ================================================
def evaluar_trayectoria_solucion(x1, x2, f, n=100):
    """ 
    Evalua una función multivariada en el segmento de recta 
    ___________________________________
    Entrada:
    x1: [1D-array] Primer punto del segmento de recta
    x1: [1D-array] Segundo punto del segmento de recta
    f: [function] Función a evaluar
    n: [int] número de evaluaciones
    ___________________________________
    Salida:
    Z : [1D-array] evaluaciones de la función en el segmento de recta
    delta: [float] distancia entre cada par de puntos utilizados
    """
    x1, x2 = np.array(x1) , np.array(x2)
    x = np.linspace(x1, x2, n)
    x = np.array([ [xi[i] for xi in x] for i in range(len(x[0]))] )
    Z = f(*x)

    return Z


