"""S1TP1_KNN.ipynb - SOLUCIONES
# **Ejercicio N°1:** Vecinos mas Cercanos

***Matemáticas para Machine Learning.***

Semana 1 - Lección 1
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


def fun_cuadratica_solucion(X, Y, A, b=0):
    """ 
    Evalua una función x^TAx sobre un conjunto de coordenadas
    ___________________________________
    Entrada:
    X: [2D-array] Valores de x
    Y: [2D-array] Valores de y
    A: [2D-array] Matriz de forma cuadrática
    ___________________________________
    Salida:
    Z : [1D-array] Derivada en cada punto.
    """
    A = np.array(A)
    Z = X*(A[0,0]*X + A[0,1]*Y) + Y*(A[1,0]*X + A[1,1]*Y) + b

    return Z

def curvas_nivel_Solucion(x,y,A):
    X,Y = np.meshgrid(np.linspace(x[0],x[1],100),np.linspace(y[0],y[1],100))
    Z = fun_cuadratica_solucion(X,Y,A)
    
    # Contour
    fig, ax = plt.subplots()
    cnt = ax.contour(X, Y, Z, colors = "k", linewidths = 0.5)
    ax.clabel(cnt, cnt.levels, inline = True, fontsize = 10)
    ax.contourf(X, Y, Z, cmap = "plasma")
    
    w,v = np.linalg.eig(A)
    plt.quiver([0,0],[0,0],v[0,:],v[1,:],color='r')
    
    ax.set_aspect('equal', adjustable='box')
    plt.title('Eigen Vectors & Contour lines')
    plt.show()