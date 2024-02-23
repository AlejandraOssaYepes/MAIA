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
from soluciones import obtener_trayectoria_solucion
from soluciones import alpha_solucion
from soluciones import gamma_solucion
from soluciones import obtener_trayectoria_solucion
from soluciones import dgamma_solucion
# Otras


PLINE = '__________________________________'


def pretty_print(df):
    """ Muestra tabla con IPython.display """
    if isinstance(df, dict): df = pd.DataFrame(df)
    return display(HTML(df.to_html()))


# 4.1 ================================================
def test_alpha(alpha):
    X = [random.randint(-3,3) for _ in range(10)]
    Y = [random.randint(-3,3) for _ in range(10)]
    
    real_list= []
    trial_list = []
    
    for x,y in zip(X,Y):
        real_list.append(alpha_solucion(x, y))
        trial_list.append(alpha(x, y))
    
    if trial_list!=real_list:
        print('Evaluación Función Alpha \n')
        df = {'X':X,'Y':Y,'Salida Real':real_list,'Salida Hallada':trial_list}
        df = pd.DataFrame(df, index=[f'Prueba {i+1}' for i in range(10)])
        pretty_print(df)
        assert False  

def test_gamma(gamma):
    X = [random.randint(-3,3) for _ in range(10)]
    Y = [random.randint(-3,3) for _ in range(10)]
    Z = [random.randint(-3,3) for _ in range(10)]
    
    real_list= []
    trial_list = []
    
    for x,y,z in zip(X,Y,Z):
        real_list.append(gamma_solucion(x, y,z))
        trial_list.append(gamma(x, y,z))
    
    if trial_list!=real_list:
        print('Evaluación Función Gamma \n')
        df = {'X':X,'Y':Y,'Z':Z,'Salida Real':real_list,'Salida Hallada':trial_list}
        df = pd.DataFrame(df, index=[f'Prueba {i+1}' for i in range(10)])
        pretty_print(df)
        assert False 


# 4.2 ================================================
def test_obtener_trayectoria(obtener_trayectoria):
    x1 = np.array([random.randint(-3,3),random.randint(-3,3)])
    x2 = np.array([random.randint(-3,3),random.randint(-3,3)])

    real_list = obtener_trayectoria_solucion(x1,x2)
    trial_list = obtener_trayectoria(x1,x2)
    
    if (trial_list[0]!=real_list[0]).all() or (trial_list[1]!=real_list[1]).all():
        print('Evaluación Obterner Trayectoria \n')
        df = {'Primer Punto':x1,'Segundo Punto':x2,
            'Trayectoria Real en X':real_list[0],'Trayectoria Hallada en X':trial_list[0],
            'Trayectoria Real en Y':real_list[1],'Trayectoria Hallada en Y':trial_list[1]}
        
        df = pd.DataFrame(df, index=[f'Prueba'])
        pretty_print(df)
        assert False 

# 4.3 ================================================
def test_evaluar_trayectoria(evaluar_trayectoria):
    pass

# 4.4 ================================================
def test_dgamma(dgamma):
    x = [random.randint(-3,3),random.randint(-3,3),random.randint(-3,3)]
    d = [random.randint(-3,3),random.randint(-3,3),random.randint(-3,3)]
    
    real_dd = dgamma_solucion(x,d)
    trial_dd = dgamma(x,d)
    
    if trial_dd!=real_dd:
        print('Evaluación Derivada Direccional de Gamma \n')
        df = {'Punto':[x],
            'Dirección':[d],
            'Derivada Direccional Real':real_dd,
            'Derivada Direccional Hallada': trial_dd}
        
        df = pd.DataFrame(df, index=[f'Prueba'])
        pretty_print(df)
        assert False