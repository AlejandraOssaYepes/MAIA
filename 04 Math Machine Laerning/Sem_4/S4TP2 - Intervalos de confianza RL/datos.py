"""S4TP2 - Intervalos de confianza.ipynb - DATOS
# **Ejercicio N°2:** 

***Matemáticas para Machine Learning.***

Semana 4 - Lección 2
"""

# Librerias Básicas
import numpy as np
import pandas as pd
import random


# GENERAR DATOS BASICOS ==================================================
def generar_datos(a, b, Xmu, Xstd, std_error, n=100, seed=None):
    """ 
    Genera n datos Y = a + bX + error
    ___________________________________
    Entrada:
    a: [float] intercepto
    b: [float] Pendiente
    Xmu:  [float] Media de los datos predictores
    Xstd: [float] Desviación estándar de los datos predictores
    std_error: [float] Desviación estándar del error
    n:    [int] número de datos a generar
    seed: [int] semilla de aleatoriedad
    ___________________________________
    Salida:
    df : [pd.DataFrame] Datos
    
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    X = np.random.normal(Xmu, Xstd, n)
    ui = np.random.normal(0, std_error, n)

    Y = b*X + a + ui
    
    df = pd.DataFrame(data=[X,Y]).T
    df.columns = ['X','Y']
  
    return df

