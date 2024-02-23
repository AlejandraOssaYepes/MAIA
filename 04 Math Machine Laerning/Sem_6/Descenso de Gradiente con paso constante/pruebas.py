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
from soluciones import mse_solucion
from soluciones import dmse_solucion
from soluciones import DG_paso_fijo_solucion

# Otras
PLINE = '__________________________________'


def pretty_print(df):
    """ Muestra tabla con IPython.display """
    if isinstance(df, dict): df = pd.DataFrame(df)
    return display(HTML(df.to_html()))


# 1) ================================================

def test_calcular_mse(mse):
    a = random.randint(-5,5)
    b = random.randint(0,5)
    
    data = [(random.randint(-20,20), random.randint(0,40)) for _ in range(10)]

    mse_real = mse_solucion(a,b,data)
    mse_trial = mse(a,b,data)
    
    if mse_trial!= mse_real:
        print('Error en el calculo del Error Cuadrático Medio. \n')
        df = {'a':a,'b':b,'MSE Real':mse_real,'MSE Hallado':mse_trial}
        df = pd.DataFrame(df,index=['Prueba'])
        pretty_print(df)
        assert False
        
def test_calcular_derivada_mse(dmse):
    a = random.randint(-5,5)
    b = random.randint(0,5)
    
    data = [(random.randint(-20,20), random.randint(0,40)) for _ in range(10)]

    dmse_real = dmse_solucion(a,b,data)
    dmse_trial = dmse(a,b,data)
    
    if dmse_trial!= dmse_real:
        print('Error en el calculo del Error Cuadrático Medio. \n')
        df = {'a':a,'b':b,'Gradiente MSE Real':[dmse_real],'Gradiente MSE Hallado':[dmse_trial]}
        df = pd.DataFrame(df,index=['Prueba'])
        pretty_print(df)
        assert False

def test_descenso_gradiente(DG_paso_fijo):
    x0 = [random.randint(-5,5),random.randint(-5,5)]
    data = [(random.randint(-20,20), random.randint(0,40)) for _ in range(5)]
    
    real_sol = DG_paso_fijo_solucion(x0,data,max_iter=10)
    real_coords = [real_sol['a'][-1],real_sol['b'][-1]]
    real_coords = np.round(real_coords,4)
    
    trial_sol = DG_paso_fijo(x0,data,max_iter=10)
    trial_coords = [trial_sol['a'][-1],trial_sol['b'][-1]]
    trial_coords = np.round(trial_coords,4)
    
    if (trial_coords!= real_coords).any():
        print('Error en el algoritmo de Descenso de Gradiente. \n')
        df = {'Solución Inicial':[x0],'Solución Real':[real_coords],'Solución Hallad':[trial_coords]}
        df = pd.DataFrame(df)
        pretty_print(df)
        assert False
