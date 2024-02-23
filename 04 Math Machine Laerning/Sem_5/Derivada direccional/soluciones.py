"""S5TP1 - Derivadas direccionales.ipynb - SOLUCIONES

# **Ejercicio N°1:** Derivadas direccionales

***Matemáticas para Machine Learning.***

Semana 5 - Actividad 1
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
from tqdm.notebook import tqdm
from copy import copy
import sympy

import plotly.colors as pcolors
import plotly.graph_objects as go

# 1) ================================================
# Funciones de laboratorio
def alpha_solucion(X,Y):
    # =====================================================
    Z =  X**2 + Y**2 + 3*Y + X
    # =====================================================
    return Z

# Función del laboratorio
def gamma_solucion(X,Y,Z):
    # =====================================================
    W =  X**2 + Y**2 + 3*Y + X + Z + Z**2
    # =====================================================
    return W

# 1) ================================================
def obtener_trayectoria_solucion(x1, x2, n=50):
    """ 
    Obtiene las coordenadas para una trayectoria entre los puntos x1 y x2
    ___________________________________
    Entrada:
    x1: [1D-array] Primer punto de la trayectoria
    x2: [1D-array] Ultimo punto de la trayectoria
    n: [int] número de puntos en la trayectoria
    ___________________________________
    Salida:
    line : [2D-array] Contiene una lista por cada coordenada del espacio de trabajo. 
                    La n-ésima list contiene las coordenadas de la n-ésima dimensión de los elementos de la trayectoria
    """
    # =====================================================
    return np.linspace(x1, x2, num=n).T
    # =====================================================


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

def dgamma_solucion(x,d):
    """ 
    Obtiene la derivada direccional de beta en el punto x en la dirección d
    ___________________________________
    Entrada:
    x: [1D-array] Punto de evaluación
    d: [1D-array] Vector dirección
    ___________________________________
    Salida:
    dval : [float] Valor de la derivada en el punto en la dirección
    """

    # Variables simbólicas
    X_sym, Y_sym, Z_sym = sympy.symbols('X Y Z')

    # Calcular derivadas parciales
    df_dX = sympy.diff(gamma_solucion(X_sym, Y_sym, Z_sym), X_sym)
    df_dY = sympy.diff(gamma_solucion(X_sym, Y_sym, Z_sym), Y_sym)
    df_dZ = sympy.diff(gamma_solucion(X_sym, Y_sym, Z_sym), Z_sym)
    
    # Evaluar las derivadas parciales
    grad_X = df_dX.subs({X_sym: x[0], Y_sym: x[1], Z_sym: x[2]}).evalf()
    grad_Y = df_dY.subs({X_sym: x[0], Y_sym: x[1], Z_sym: x[2]}).evalf()
    grad_Z = df_dZ.subs({X_sym: x[0], Y_sym: x[1], Z_sym: x[2]}).evalf()
    
    # Calcular la norma del vector direccional d
    norma = np.linalg.norm(d)

    # Calcular la derivada direccional
    dd_gamma = grad_X*d[0]/norma + grad_Y*d[1]/norma + grad_Z*d[2]/norma

    return dd_gamma
