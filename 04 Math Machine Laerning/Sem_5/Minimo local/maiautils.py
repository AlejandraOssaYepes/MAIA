"""S5TP2 - Minimos Localesw.ipynb - UTILS

***Matemáticas para Machine Learning.***

Semana 5 - Tarea 2
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
from pruebas import test_minimo_local, test_definida_positiva

# Otras
from mtr import manipulate_ipython

class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
        
    # PRUEBAS ========================================================
    def calificar_minimo_local(self, fun_tentativa):
        """ Califica la función *minimo_local*
        fun_tentativa: [function] Función realizada por el estudiante
        """
        print(self.PLINE, 'Verificando errores...\n\n', sep='\n')
        test_minimo_local(fun_tentativa)
        print('\n\nSin Errores', self.PLINE, sep='\n')
        
    def calificar_definida_positiva(self, fun_tentativa):
        """ Califica la función *minimo_local*
        fun_tentativa: [function] Función realizada por el estudiante
        """
        print(self.PLINE, 'Verificando errores...\n\n', sep='\n')
        test_definida_positiva(fun_tentativa)
        print('\n\nSin Errores', self.PLINE, sep='\n')