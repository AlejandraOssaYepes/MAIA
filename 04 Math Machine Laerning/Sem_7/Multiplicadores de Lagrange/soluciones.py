"""S7TP2 - Multiplicadores de Lagrange.ipynb - UTILS

***Matemáticas para Machine Learning.***

Semana 7 - Tarea 2
"""

# LIBRERIAS ================================================
from sympy import *
import numpy as np
import matplotlib.pyplot as plt


def grafica_nivel(A:list,B:list,a:list,c:list,xlim:list,ylim:list,n=1000):
    """ 
    Grafica las curvas de nivel de f y las rectas correspondientes a las restricciones lineales.
    ___________________________________
    Entrada:
    A: [list] Lista cuyos elementos son las filas de la matriz A.
    B: [list] Lista que contiene los coeficientes del vector B.
    a: [list] Lista cuyos elementos son los vector de las restricciones lineales.
    c: [list] Lista cuyos elementos son las constantes asociadas a cada restricción.
    xlim: [list] Limites del eje x.
    ylim: [list] Limites del eje y.
    n: [int] Número de muestras para cada variable independiente.
    ___________________________________
    Salida:
    Se graficaron las curvas de nivel de la función y las rectas de las restricciones.
    """
    x = np.linspace(xlim[0],xlim[1],n)
    y = np.linspace(ylim[0],ylim[1],n)
    X,Y = np.meshgrid(x,y)
    
    Z = A[0][0]*X**2 + A[1][1]*Y**2 + (A[0][1]+A[1][0])*X*Y + B[0]*X + B[1]*Y
    
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize = (15, 5))

    CS = ax_l.contourf(X, Y, Z, 8,cmap='winter')
    for i in range(len(a)):
        ax_l.plot(x,(c[i]-a[i][0]*x)/a[i][1])
    cbar = fig.colorbar(CS,ax=ax_l)

    CS = ax_r.contour(X, Y, Z, 8,cmap='winter')
    for i in range(len(a)):
        ax_r.plot(x,(c[i]-a[i][0]*x)/a[i][1])
    cbar = fig.colorbar(CS,ax=ax_r)

def metodo_lagrange(A:list,B:list,a:list,c:list):
    """ 
    Implementa el método de Multiplicadores de Lagrange
    ___________________________________
    Entrada:
    A: [list] Lista cuyos elementos son las filas de la matriz A.
    B: [list] Lista que contiene los coeficientes del vector B.
    a: [list] Lista cuyos elementos son los vector de las restricciones lineales.
    c: [list] Lista cuyos elementos son las constantes asociadas a cada restricción.
    ___________________________________
    Salida:
    out : [dic] Argumento que minimiza f(x,y) y valores de los multiplicadores de Lagrange. 
                Si el problema es infactible retorna [0,0].
    """
    x, y, l0, l1 = symbols('x, y, l0, l1')
    xs = Matrix([[x], [y]])
    
    A = Matrix(A)
    B = Matrix(B)
    
    fo = xs.dot(A*xs) + B.dot(xs)
    
    r = []
    for i in range(len(a)):
        r.append(a[i][0]*x+a[i][1]*y-c[i])
    
    L = fo - l0*r[0]
    if len(a)>1:
        L= L-l1*r[1]
        
    dl = [diff(L,x),diff(L,y),diff(L,l0)]
    
    if len(a)>1:
        dl.append(diff(L,l1))
        
    try:
        sol = solve(dl,[x,y,l0,l1],dic=True)
    except:
        sol = [0,0]
    return sol