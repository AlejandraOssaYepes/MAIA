"""S5TP1 - Derivadas direccionales.ipynb - UTILS

# **Ejercicio N°1:** Derivadas direccionales

***Matemáticas para Machine Learning.***

Semana 1 - Actividad 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np      
from itertools import product


# Visualización
from IPython.display import display, HTML
from mtr import manipulate_ipython


class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'