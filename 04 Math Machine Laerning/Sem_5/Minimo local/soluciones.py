"""S5TP2 - Minimos Localesw.ipynb - UTILS

***Matemáticas para Machine Learning.***

Semana 5 - Tarea 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np


def minimo_local(P,F,epsilon,n=10):
    """ 
    Evalua los puntos de la región [x*-epsilon,x*+epsilon]X[y*-epsilon,y*+epsilon],
    luego identifica si hay un punto (x,y) cuya función objetivo es menor a f(x*,y*).
    ___________________________________
    Entrada:
    P: [1D-array] Valores del punto (x*,*)
    F: [function] función a evaluar
    epsilon: [float] longitud de la mitad del lado del cuadrado.
    n: [int] número de puntos a evaluar
    ___________________________________
    Salida:
    otro_punto : [Boolean] Indica si hay un mejor candidato a mínimo local en la región.
    """
    # Candidato a mínimo
    F_P = F(P[0],P[1])
    
    # Limites de la grilla
    x_lims = [P[0]-epsilon,P[0]+epsilon]
    y_lims = [P[1]-epsilon,P[1]+epsilon]
    
    # Grilla
    X, Y = np.linspace(*x_lims, num=n), np.linspace(*y_lims, num=n)
    X, Y = np.meshgrid(X, Y)
    
    # Evalua
    Z = F(X,Y)
    
    # Parámetros del ciclo
    i=0
    otro_punto = False
    
    # Ciclo que itera en el eje Y.
    while (i<n and otro_punto==False):
        
        # Ciclo que itera en el eje X.
        j = 0
        while (j<n and otro_punto==False):
            
            # Condición de mínimo local.
            if Z[i][j]< F_P:
                otro_punto = True
            j += 1
        i +=1
    return otro_punto

def definida_positiva(A):
    """ 
    Identifica si la matriz A es definida positiva 
    ___________________________________
    Entrada:
    A: [2D-array] Matriz Hessiana 2x2.
    ___________________________________
    Salida:
    [Boolean] True si A es definida positiva, False en caso contrario.
    """
    return np.all(np.linalg.eigvals(A) > 0)