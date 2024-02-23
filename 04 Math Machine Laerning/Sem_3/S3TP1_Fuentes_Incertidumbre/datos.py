"""S3TP2 - Fuentes de incertidumbre.ipynb - DATOS
# **Ejercicio N°2:** Fuentes de incertidumbre

***Matemáticas para Machine Learning.***

Semana 3 - Lección 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
import pandas as pd
from sklearn import datasets
from itertools import product
import random

# Soluciones
def generar_datos_estocasticos(mus, sigmas, n=10, dist=None, seed=None):
    """ 
    Genera n datos de alguna de las distribuciones introducidas
    ___________________________________
    Entrada:
    mus:    [1D-array] lista de medias 
    sigmas: [1D-array] lista de varianzas 
    n: [1D-array]lista de varianzas 
    ___________________________________
    Salida:
    data: [array] Datos producidos 
    """
    m = len(mus)  # Número de distribuciones
    # Selecciona distribución aleatoria
    if dist is None:
        dist = round(m*np.random.rand()-0.5)
    if seed is not None:
        np.random.seed(seed)
        
    mu = mus[dist]
    sigma = sigmas[dist]

    # Genera datos 1 dimensión
    if isinstance(mu, (int, float, complex)):
        data = np.random.normal(mu, sigma, n)
    # Genera datos multidimensionales
    else:
        data = np.random.multivariate_normal(mu, sigma, n)
    return data


# Generar datos
def obtener_datos():
    """ OBtiene 3000 datos de 3 categorias distintas """
    # Parámetros
    n = 1000
    mus=[20,27]
    sigmas = [3,3]

    # Generación de datos
    X, Y = [], []
    for i, (mu, sigma) in enumerate(zip(mus,sigmas)):
        x = generar_datos_estocasticos(mus=[mu], sigmas=[sigma], n=n, dist=0, seed=i)
        # Guardar
        Y.extend([i]*len(x))
        X.extend(x)
    
    # Caso única distribución
    if isinstance(X[0], float) or isinstance(X[0], int):
        df = pd.DataFrame(data=[X,Y]).T
        cols = ['X']
    # Caso varias distribuciones
    else:
        df = pd.DataFrame(data= X)
        cols = [f"X{i+1}" for i in range(X[0])]
        df['Y'] = Y
        
    cols += ['Y']
    df.columns = cols

    return df 


def datos_dividir_en_clases(data):
    """ Cambia la estructura de datos """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
     
    nclases = len(set(data[:,1]))
    Xclase = {idx:[data[i][0] for i in range(len(data)) if data[i][1] == idx] for idx in range(nclases)}
    
    return Xclase
