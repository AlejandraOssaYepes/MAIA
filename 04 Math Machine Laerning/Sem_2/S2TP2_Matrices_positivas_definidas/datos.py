"""S1TP1_KNN.ipynb - DATOS
# **Ejercicio N°1:** Vecinos mas Cercanos

***Matemáticas para Machine Learning.***

Semana 1 - Lección 1
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
from sklearn import datasets
from itertools import product
import random

# Soluciones
from soluciones import obtener_forma_simetrica_sol


# GENERAR MATRICES PD ==================================================
def generar_matrices_balanceadas(dim=[2, 3], n=2, rnd=1, as_int=True):
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

    # matrices Aleatorias
    Arnd = []
    np.random.seed(rnd)
    # n matrices por dimensión
    for d in dim:
        Atmp = np.random.rand(n, d, d)
        Atmp = [obtener_forma_simetrica_sol((A - 0.5) * 2) for A in Atmp]
        Arnd.extend(list(Atmp))

    # Salida
    Aout = Apd + Arnd
    random.shuffle(Aout)

    # redondear
    if as_int:
        Aout = [(np.array(A).round(3) * 1000).astype(int) for A in Aout]

    return Aout
