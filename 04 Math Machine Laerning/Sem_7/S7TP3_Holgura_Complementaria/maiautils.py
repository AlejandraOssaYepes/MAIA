""" SXTPY_NombreCuaderno.ipynb - UTILS """

# LIBRERIAS ================================================
# Básicas
import numpy as np      
import pandas as pd

# Dibujar
import matplotlib.pyplot as plt 

# Visualización
from IPython.display import display, HTML

# Pruebas
from pruebas import test_fun

# Otras
from visualizacion import dibujar3D
from visualizacion import dibujarCN
#from visualizacion import add_grid
from visualizacion import hfigures

from mtr import manipulate_ipython


class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
        
        # VISUALIZAR
        self.dibujar3D = dibujar3D
        self.dibujarCN = dibujarCN
        #self.add_grid = add_grid
        self.hfigures = hfigures
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
        """ Obtiene el nombre del objeto en el entorno global """
        variables_found = [name for name, val in globals().items() if val is obj]
        return variables_found[0]
    
    def fun_cuadratica(self,X, Y, A, B):
        """ 
        Evalua una función x^TAx sobre un conjunto de coordenadas
        ___________________________________
        Entrada:
        X: [2D-array] Valores de x
        Y: [2D-array] Valores de y
        A: [2D-array] Matriz de forma cuadrática
        ___________________________________
        Salida:
        Z : [1D-array] Derivada en cada punto.
        """
        A = np.array(A)
        Z = X*(A[0,0]*X + A[0,1]*Y) + Y*(A[1,0]*X + A[1,1]*Y) + B[0]*X + B[1]*Y

        return Z

    # PRUEBAS ========================================================
    def correr_prueba_fun(self, fun):
        """ Modulo de pruebas función fun """
        self.correr_prueba(test_fun, fun)