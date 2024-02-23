"""S7TP2 - Multiplicadores de Lagrange.ipynb - UTILS

***Matemáticas para Machine Learning.***

Semana 7 - Tarea 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np      
import pandas as pd
from itertools import product

# Dibujar
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

# Visualización
from IPython.display import display, HTML

# Pruebas
from pruebas import test_lagrange

# Soluciones
from soluciones import metodo_lagrange

# Otras
from mtr import manipulate_ipython

class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
        
    # PRUEBAS ========================================================
    def calificar_minimo_local(self, fun_tentativa):
        """ Califica la función *test_lagrange*
        fun_tentativa: [function] Función realizada por el estudiante
        """
        test_lagrange(fun_tentativa)
