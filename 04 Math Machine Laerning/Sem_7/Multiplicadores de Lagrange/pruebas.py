"""S7TP2 - Multiplicadores de Lagrange.ipynb - UTILS

***Matemáticas para Machine Learning.***

Semana 7 - Tarea 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np      
import pandas as pd
from itertools import product
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import random

# Soluciones
from soluciones import *

# Parámetros de la forma cuadrática a usar para la calificación
A = [[1,0],[0,1]]
B = [1,0]
a = [[1,1],[0,0]]
c = [2,0]

def actualizar_parametros():
    """ 
    Actualiza los valores de A,B,a,c
    """
    ruido = random.uniform(0, 10)
    
    A = [[10,ruido],[ruido,10]]
    B = [random.uniform(0.5,5),random.uniform(0.5,5)]
    a = [[random.randint(-2,2),random.randint(-2,2)],[random.randint(-2,2),random.randint(-2,2)]]
    c = [random.randint(-2,2),random.randint(-2,2)]

def test_lagrange(fun_tentativa):
    """ 
    Compara la solución del estudiante con la solución correcta.
    ___________________________________
    F: [function] función a evaluar
    """
    i = 0
    sol_tentativa_correcta = True
    while i < 100 and sol_tentativa_correcta == True:
        
        sol = metodo_lagrange(A,B,a,c)
        sol = [j for j in sol.values()]
        
        sol_tentativa = fun_tentativa(A,B,a,c)
        sol_tentativa = [j for j in sol_tentativa.values()]
        
        if sol_tentativa!=sol:
            sol_tentativa_correcta = False
        assert sol_tentativa_correcta, f"Error en la prueba, el punto {sol[0],sol[1]} no es el mínimo local de f(x,y) cuando A={A}, B={B}, a={a} y c={c}."
        
        actualizar_parametros()
        
        i+=1
        