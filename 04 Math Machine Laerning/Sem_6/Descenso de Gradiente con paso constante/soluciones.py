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

def mse_solucion(a, b, data):
    """ 
    Obtiene el valor del error cuadrático medio (MSE) a partir de un modelo lineal con parámetros a,b
    y un conjunto de datos.
    ___________________________________
    Entrada:
    data: [2D-array] Datos
    a, b: [float] Valores de los parámetros a y b del modelo
    n: [int] número de puntos a evaluar
    ___________________________________
    Salida:
    error : [1D-array] Valor del error cuadrático para cada valor de a considerado
    """
    return sum([(b*xi[0] + a - xi[1])**2 for xi in data])

def dmse_solucion(a, b, data):
    """ 
    Obtiene el gradiente del mse en el punto (a,b)
    ___________________________________
    Entrada:
    data: [2D-array] Datos
    a: [float] Valor de a
    b: [int] número de puntos a evaluar
    ___________________________________
    Salida:
    error : [1D-array] Valor del error cuadrático para cada valor de a considerado
    """
    dda = sum([2*(b*xi[0] + a - xi[1]) + 1 for xi in data])
    ddb = sum([2*(b*xi[0] + a - xi[1])*xi[0] for xi in data])
    return dda, ddb

def DG_paso_fijo_solucion(x0, data, lrate=0.01, max_iter=1000, tol=1e-5):
    """
    Encuentra el punto (a,b) que me brinda el mínimo MSE para los datos en data
    ___________________________________
    Entrada:
    x0:    [1D-array] punto inicial (a0, b0)
    data:  [2D-array] datos
    lrate: [float] Velocidad de aprendizaje
    max_iter: [int] Máximo número de iteraciones
    ___________________________________
    Salida:
    xf: [dict] Información sobre el trayecto recorrido hasta el mínimo {coords, val}
    """
    w = x0
    dif = 1e3
    i = 0
    dic = {'a':[w[0]],'b':[w[1]]}
    
    while (i<max_iter and dif>tol):
        
        g = dmse_solucion(w[0],w[1],data)
        d = list(-1*j*lrate for j in g)
        w = list(map(sum, zip(w, d)))
        
        dif = np.linalg.norm(d)
        
        dic['a'].append(w[0])
        dic['b'].append(w[1])
        
        i = i+1
    return dic

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
    line = np.linspace(x1, x2, num=n)
    line = [[it[i] for it in line] for i in range(len(x1))]
    # =====================================================
    
    return line



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


