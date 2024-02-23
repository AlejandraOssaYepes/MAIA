"""S5TP1 - Derivadas direccionales.ipynb - PRUEBAS

# **Ejercicio N°1:** Derivadas direccionales

***Matemáticas para Machine Learning.***

Semana 5 - Actividad 1
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np      
import pandas as pd
import random
from itertools import product
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# Soluciones
#from soluciones import obtener_trayectoria_solucion
from soluciones import gradfABC_solucion
from soluciones import minfd_solucion
from soluciones import DG_solucion

# Otras


PLINE = '__________________________________'


def pretty_print(df):
    """ Muestra tabla con IPython.display """
    if isinstance(df, dict): df = pd.DataFrame(df)
    return display(HTML(df.to_html()))


# 1) ================================================
def test_obtener_trayectoria(obtener_trayectoria):
    """
    Prueba función obtener_trayectoria
    ___________________________________
    Entrada:
    obtener_trayectoria: [function] función a probar
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """
    pass

def test_calculo_gradiente(gradfABC):
    """
    Prueba función obtener_trayectoria
    ___________________________________
    Entrada:
    obtener_trayectoria: [function] función a probar
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """
    X = random.randint(0,10)
    Y = random.randint(0,10)
    A = [[random.randint(-5,5),0],[0,random.randint(-5,5)]]
    B = [random.randint(-5,5),random.randint(-5,5)]
    
    dx_try, dy_try = gradfABC(X,Y,A,B)
    dx_real, dy_real = gradfABC_solucion(X,Y,A,B)
    
    if dx_try!=dx_real:
        print('Error en el calculo de la derivada parcial respecto a X. \n')
        df = {'Punto':[X,Y],'A':A,'B':B,'dx Real':dx_real,'dx Hallada':dx_try}
        df = pd.DataFrame(df)
        pretty_print(df)
        assert False
    elif dy_try!=dy_real:
        print('Error en el calculo de la derivada parcial respecto a Y. \n')
        df = {'Punto':[X,Y],'A':A,'B':B,'dy Real':dy_real,'dy Hallada':dy_try}
        df = pd.DataFrame(df)
        pretty_print(df)
        assert False
        
def test_calculo_tasa(minfd):
    
    x0 = [random.randint(-5,5),random.randint(-5,5)]
    d = [random.randint(-5,5),random.randint(-5,5)]
    A = [[random.randint(-5,5),0],[0,random.randint(-5,5)]]
    B = [random.randint(-5,5),random.randint(-5,5)]
    
    real_t = minfd_solucion(x0,d,A,B)
    trial_t = minfd(x0,d,A,B)
    
    if trial_t!= real_t:
        print('Error en el calculo de la tasa de aprendizaje. \n')
        df = {'Punto':x0,'Dirección':d,'A':A,'B':B,'Tasa Real':real_t,'Tasa Hallada':trial_t}
        df = pd.DataFrame(df)
        pretty_print(df)
        assert False
        
def test_descenso_gradiente(DG):
    x0 = [random.randint(-5,5),random.randint(-5,5)]
    A = [[random.randint(-5,5),0],[0,random.randint(-5,5)]]
    B = [random.randint(-5,5),random.randint(-5,5)]
    C = random.randint(-5,5)
    
    real_sol = DG(x0,A,B,C)
    real_coords = real_sol['coords'][-1]
    real_coords = np.round(real_coords,4)
    
    trial_sol = DG_solucion(x0,A,B,C)
    trial_coords = trial_sol['coords'][-1]
    trial_coords = np.round(trial_coords,4)
    
    if (trial_coords!= real_coords).any():
        print('Error en el algoritmo de Descenso de Gradiente. \n')
        df = {'Punto':[x0],'A':[A],'B':[B],'C':C,'Coordenadas Reales':[real_coords],'Coordenadas Halladas':[trial_coords]}
        df = pd.DataFrame(df)
        pretty_print(df)
        assert False