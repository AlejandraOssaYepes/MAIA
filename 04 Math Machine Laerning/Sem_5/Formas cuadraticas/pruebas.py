"""S1TP1 - Distancias y algoritmo KNN.ipynb - PRUEBAS

# **Ejercicio N°1:** Distancias y algoritmo KNN

***Matemáticas para Machine Learning.***

Semana 1 - Tarea 1
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np      
import pandas as pd
from itertools import product
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# Soluciones
from soluciones import fun_cuadratica_solucion
# Otras


PLINE = '__________________________________'


def pretty_print(df):
    """ Muestra tabla con IPython.display """
    if isinstance(df, dict): df = pd.DataFrame(df)
    return display(HTML(df.to_html()))


def test_fun_cuadratica(fun_cuadratica):
    """ prueba la función comparando con los resultados en solución """
    # TODO
    pass
