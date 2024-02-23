"""S5TP2 - Minimos Localesw.ipynb - UTILS

***Matemáticas para Machine Learning.***

Semana 5 - Tarea 2
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
a = 1
b = 2
c = 3
d = 4

def actualizar_parametros():
    """ 
    Actualiza los valores de a,b,c,d
    """
    a = random.uniform(-10, 0.5)
    b = random.uniform(0.5,5)
    c = random.randint(-2,2)
    d = random.randint(-4,4)+1

def funcion_cuadratica(X,Y):
    return a*X**2 +b*Y**2 + c*X + d*Y

def test_minimo_local(fun_tentativa):
    """ 
    Compara la solución del estudiante con la solución correcta.
    ___________________________________
    F: [function] función a evaluar
    """
    i = 0
    sol_tentativa_correcta = True
    while i < 100 and sol_tentativa_correcta == True:
        actualizar_parametros()
        x_op = -1*c/(2*a)
        y_op = -1*d/(2*b)
        sol = minimo_local([x_op,y_op],funcion_cuadratica,0.1)
        sol_tentativa = fun_tentativa([x_op,y_op],funcion_cuadratica,0.1)
        if sol_tentativa!=sol:
            sol_tentativa_correcta = False
        assert sol_tentativa_correcta, f"Error en la prueba, el punto {x_op,y_op} es mínimo local de f(x,y)= {a}x^{2} + {b}y^{2} + {c}x + {d}y."
        i+=1
        
def test_definida_positiva(fun_tentativa):
    """ 
    Compara la solución del estudiante con la solución correcta.
    ___________________________________
    F: [function] función a evaluar
    """
    i = 0
    sol_tentativa_correcta = True
    while i < 100 and sol_tentativa_correcta == True:
        actualizar_parametros()
        A = np.array([[a,b],[c,d]])
        sol = definida_positiva(A)
        sol_tentativa = fun_tentativa(A)
        if sol_tentativa!=sol:
            sol_tentativa_correcta = False
        assert sol_tentativa_correcta, f"Error en la prueba, la matriz {A}\n Corrobore manualmente si {A} es positiva definida."
        i+=1