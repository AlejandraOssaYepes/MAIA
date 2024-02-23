"""S4TP2 - Intervalos de confianza.ipynb - SOLUCIONES
# **Ejercicio N°2:** 

***Matemáticas para Machine Learning.***

Semana 4 - Lección 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
import pandas as pd
from scipy.stats import t

def regresion_lineal_sol(data):
    """ 
    Realiza regresión lineal ordinaria sobre los datos
    ___________________________________
    Entrada:
    data: [pd.dataFrame] Columnas ['X','Y'] con datos
    ___________________________________
    Salida:
    a: [float] intersección
    b: [float] pendiente
    """  
    # =====================================================
    # COMPLETAR ===========================================
    # -
    # AYUDA:
    dic = {'uno':[1]*len(data),'X':data['X']}
    X = np.array([values for values in dic.values()])
    X = X.T
    
    betas = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,data['Y']))
    
    a = betas[0]
    b = betas[1]
    
    return a, b

def intervalo_confianza_betas_sol(X, Y, alpha=0.05):
    """ 
    Retorna los prámetros e media y ancho del intervalo de confianza:
    ___________________________________
    Entrada:
    X: [1D-array] Muestra de las variables explicativas
    Y: [1D-array] Muestra de la variable de respuesta
    alpha: [float] número entre 0 y 1 que determina la significancia   
    ___________________________________
    Salida:
    ic: [2D-array] Intervalo de confianza para cada parámetro
    """    

    n = len(X)
    p = 2
    
    dic = {'uno':[1]*len(X),'X':X}
    X = np.array([values for values in dic.values()])
    X = X.T
    
    XX = np.dot(X.T,X)
    H = np.linalg.inv(XX)
    XY = np.dot(X.T,Y)

    betas = np.dot(H,XY)
    
    error = Y - np.dot(X,betas)
    sigma = np.linalg.norm(error)/(n-p)

    deviation = sigma*np.sqrt(np.diag(H))
    ec = t.ppf(1-alpha/2, n-p)
    bandwidth = ec*deviation
    
    ic = [[betas[i]-bandwidth[i],betas[i]+bandwidth[i]] for i in range(p)]

    return ic