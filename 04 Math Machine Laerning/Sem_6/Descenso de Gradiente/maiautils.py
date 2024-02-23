"""S5TP1 - Derivadas direccionales.ipynb - UTILS

# **Ejercicio N°1:** Derivadas direccionales

***Matemáticas para Machine Learning.***

Semana 5 - Actividad 1
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
from pruebas import test_calculo_gradiente
from pruebas import test_calculo_tasa
from pruebas import test_descenso_gradiente

# Otras
from visualizacion import dibujar3D
from visualizacion import dibujarCN
from visualizacion import add_manual_grid
from visualizacion import hfigures
from visualizacion import plot_data
from visualizacion import plot_trayectoria


from mtr import manipulate_ipython


class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
        
        # VISUALIZAR
        self.dibujar3D = dibujar3D
        self.dibujarCN = dibujarCN
        self.add_grid = add_manual_grid
        self.hfigures = hfigures
        self.plot_data = plot_data
        self.plot_trayectoria = plot_trayectoria
        # DATOS
        
        
    # BASICO ===========================================================
    def correr_prueba(self, test_fun, fun):
        """
        Corre prueba para una función. Muestra progreso en mensajes.
        ___________________________________
        Entrada:
        test_fun: [function] Modulo de prueba especifico a la función
        fun:      [function] Función a probar
        """
        print(self.PLINE, 'Verificando errores...\n\n', sep='\n')
        test_fun(fun)
        print('\n\nSin Errores', self.PLINE, sep='\n')

        
    def var_name(obj):
        """ 
        Obtiene el nombre del objeto en el entorno global
        """
        variables_found = [name for name, val in globals().items() if val is obj]
        return variables_found[0]

    # PRUEBAS ========================================================
    def correr_prueba_calculo_gradiente(self,gradfABC):
        self.correr_prueba(test_calculo_gradiente,gradfABC)
        
    def correr_prueba_calcular_tasa(self,minfd):
        self.correr_prueba(test_calculo_tasa,minfd)
    
    def correr_prueba_descenso_gradiente(self,DG):
        self.correr_prueba(test_descenso_gradiente,DG)