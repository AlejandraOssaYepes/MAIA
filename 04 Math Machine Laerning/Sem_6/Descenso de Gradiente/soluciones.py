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


import plotly.colors as pcolors
import plotly.graph_objects as go

# Parámetros función polinomial
A = np.array([[1,3],[3,4]])
B = np.array([2,3])
C = 5


def fABC_solucion(X, Y, A=[[1,0],[0,1]] , B=[1,1], C=0):
    """ 
    Evalua la función polinomial x^TAx + Bx + C en los puntos (xi,yi) con xi en X y yi en Y
    ___________________________________
    Entrada:
    X: [2D-array] Coordenadas eje x (Tipo meshgrid)
    Y: [2D-array] Coordenadas eje y (Tipo meshgrid)
    A: [2D-array] Matriz de forma cuadrática
    B: [2D-array] Vector de forma lineal
    C: [float] Constante
    ___________________________________
    Salida:
    Z: [2D-array] Resultado eje z (Tipo meshgrid)
    """  
    A, B = np.array(A), np.array(B)
    # =====================================================
    Z = A[0,0]*(X**2) + (A[1,0]+A[0,1])*X*Y + A[1,1]*(Y**2) + B[0]*X + B[1]*Y + C
    # =====================================================
    return Z

def gradfABC_solucion(X, Y, A=A, B=B):
    """ 
    Evalua la función polinomial en los puntos (xi,yi) con xi en X y yi en Y
    ___________________________________
    Entrada:
    X: [2D-array] Coordenadas eje x (Tipo meshgrid)
    Y: [2D-array] Coordenadas eje y (Tipo meshgrid)
    A: [2D-array] Matriz de forma cuadrática
    B: [1D-array] Vector forma lineal
    ___________________________________
    Salida:
    dX, dY: [2D-array] Derivada sobre cada coordenada (Tipo meshgrid)
    """    
    A, B = np.array(A), np.array(B)
    # =====================================================
    dX = 2*A[0,0]*X + (A[1,0]+A[0,1])*Y + B[0]
    dY = (A[1,0]+A[0,1])*X + 2*A[1,1]*Y + B[1] 
    # =====================================================
    return dX, dY

def minfd_solucion(x0, d, A=A, B=B):
    """ 
    Encuentra el mínimo de f desde el punto x0 en la dirección d.
    ___________________________________
    Entrada:
    x0:[1D-array] vector inicial
    A: [2D-array] Matriz de forma cuadrática
    B: [1D-array] Vector de forma lineal
    C: [float] Constante
    ___________________________________
    Salida:
    t: [float] retorna t de tal manera  que f(x0 + t*d) es mínimo en t.
    """    
    x0, d, A, B = np.array(x0), np.array(d), np.array(A), np.array(B)       # Formato
    # =====================================================
    t = -(d@A@x0 + x0@A@d + B@d)/(2*d@A@d)
    # =====================================================
    return t

def DG_solucion(x0, A=A, B=B, C=C, max_iter=10,tol=1e-6):
    """ 
    Encuentra el mínimo de f = x^TAx + Bx + C, Empieza iterciones en x0
    ___________________________________
    Entrada:
    x0:[1D-array] vector inicial
    A: [2D-array] Matriz de forma cuadrática
    B: [1D-array] Vector de forma lineal
    C: [float] Constante
    max_iter: [int] Máximo número de iteraciones
    ___________________________________
    Salida:
    xf: [dict] Información sobre el trayecto recorrido hasta el mínimo {coords, val}
    """  
    
    x0,A,B = np.array(x0),np.array(A), np.array(B)  
    x = x0
    dic= {'coords':[x0], 'val':[fABC_solucion(x0[0],x0[1],A,B,C)]}
    
    dif = 1e3
    i = 0
    while (i<max_iter and dif>tol):
    
        grad = 2*A@x + B
        a = minfd_solucion(x,-grad,A,B)
        x = x-a*grad
        val = fABC_solucion(x[0],x[1],A,B,C)
        dif = np.linalg.norm(a*grad)
        
        dic['coords'].append(x)
        dic['val'].append(val)
        i=i+1
    return dic