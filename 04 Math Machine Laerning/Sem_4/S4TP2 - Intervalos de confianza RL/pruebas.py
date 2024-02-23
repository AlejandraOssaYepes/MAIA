"""S4TP2 - Intervalos de confianza.ipynb - PRUEBAS

# **Ejercicio N°2:** 

***Matemáticas para Machine Learning.***

Semana 4 - Actividad 2
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
from soluciones import regresion_lineal_sol
from soluciones import intervalo_confianza_betas_sol

from datos import generar_datos


PLINE = '__________________________________'

def pretty_print(df):
    """ Muestra tabla con IPython.display """
    if isinstance(df, dict): df = pd.DataFrame(df)
    return display(HTML(df.to_html()))


# 1) ================================================
def test_regresion_lineal(regresion_lineal):
    """
    Prueba función norm
    ___________________________________
    Entrada:
    norm: [function] función a probar
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """
    
    data = generar_datos(a=random.randint(-2,2), b=random.randint(1, 10), Xmu=0, Xstd=random.randint(5, 10), std_error=1, n=100, seed=None)
    
    a, b = regresion_lineal(data)
    a0, b0 = regresion_lineal_sol(data)
    
    trial = [a,b]
    real = [a0,b0]
    
    if trial!=real:
        print('Comparación de Regresión Lineal \n')
        df = {'Real':real,'Hallado':trial}
        df = pd.DataFrame(df, index=['Intercepto','Pendiente'])
        pretty_print(df)
        assert False  

# 3) ================================================ 
def test_intervalo_confianza_betas(intervalo_confianza_betas):
    """
    Prueba función norm
    ___________________________________
    Entrada:
    norm: [function] función a probar
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """
    
    data = generar_datos(a=random.randint(-2,2), b=random.randint(1, 10), Xmu=0, Xstd=random.randint(5, 10), std_error=1, n=100, seed=None)
    
    X = data['X']
    Y = data['Y']
    
    ic_real = intervalo_confianza_betas_sol(X,Y)
    ic_trial = intervalo_confianza_betas(X,Y)
    
    if ic_trial!=ic_real:
        print('Comparación de Regresión Lineal \n')
        df = {'Real':ic_real,'Hallado':ic_trial}
        df = pd.DataFrame(df, index=['Intercepto','Pendiente'])
        pretty_print(df)
        assert False  

def test_intervalos_confianza(obtener_intervalos_confianza):
    pass
