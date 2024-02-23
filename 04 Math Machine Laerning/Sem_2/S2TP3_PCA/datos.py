"""S2TP3_PCA.ipynb - DATOS

# **Ejercicio N°3:**  PCA

***Matemáticas para Machine Learning.***

Semana 2 - Tarea 3
"""

from sklearn import datasets
from itertools import product
import numpy as np

# GENERAR DATOS BASICOS ==================================================
def generar_datos(n=10, dim=2, nlog=3, seed=None):
    """
    Genera datos aleatorios con coordenadas enteras
    ___________________________________
    Entrada:
    n:    [int] número de datos
    dim:  [int] dimensión de datos
    nlog: [int] vectores en [0, ,10^nlog]]
    seed: [int] semilla de alleatoriedad
    ___________________________________
    Salida:
    V: [2D-array] n datos de dimensión dim
    x: [1D-array] 1 dato de dimensión dim
    """
    # Inicializa aleatoriedad
    if seed is not None: np.random.seed(seed)

    # Familia de vectores
    V = np.around(np.random.rand(n, dim), nlog)*10**(nlog)

    return V.astype(int)


def generar_datos_distribuidos(n=10, dim=2, nlog=3, seed=None):
    """
    Genera datos aleatorios con coordenadas enteras
    ___________________________________
    Entrada:
    n:    [int] número de datos
    dim:  [int] dimensión de datos
    nlog: [int] vectores en [0, ,10^nlog]]
    seed: [int] semilla de alleatoriedad
    ___________________________________
    Salida:
    V: [2D-array] n datos de dimensión dim
    x: [1D-array] 1 dato de dimensión dim
    """
    # Inicializa aleatoriedad
    if seed is not None: np.random.seed(seed)

    # Familia de vectores
    V = np.around(np.random.rand(n, dim), nlog)*10**(nlog)

    return V.astype(int)

# GENERAR MATRICES PD ==================================================
def generar_matrices_pd(dim=[2, 3], n=2, rnd=1, as_int=True):
    """
    Genera matrices positivas definidas y aleatorias
    ___________________________________
    Entrada:
    dim: [1D-array] lista de dimensiones
    n: [int] matrices pd y aleatorias por dimensión
    rnd: [int] semilla de aleatoriedad
    as_int: [bool] llevar los números a enteros
    ___________________________________
    Salida:
    A_list: [list]  Lista de matrices
    """
    # Matrices positivas definidas
    Apd = []
    # n matrices por dimensión
    for d, _ in product(dim, range(n)):
        Apd.append(datasets.make_spd_matrix(d, random_state=rnd))
        rnd += 1

    # redondear
    if as_int:
        Apd = [(np.array(A).round(3) * 1000).astype(int) for A in Apd]

    return Apd
