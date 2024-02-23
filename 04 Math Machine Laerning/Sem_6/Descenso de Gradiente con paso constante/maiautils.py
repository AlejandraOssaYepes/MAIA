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
from pruebas import test_calcular_mse
from pruebas import test_calcular_derivada_mse
from pruebas import test_descenso_gradiente

# Otras
from visualizacion import dibujar3D
from visualizacion import dibujarCN
from visualizacion import add_grid
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
        self.add_grid = add_grid
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
        """ 
        Obtiene el nombre del objeto en el entorno global
        """
        variables_found = [name for name, val in globals().items() if val is obj]
        return variables_found[0]
    
    def modelo(self,a,b,x):
        return a+b*x
    
    def plot_data(self,data):
        x = [punto[0] for punto in data]
        y = [punto[1] for punto in data]
        plt.scatter(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        
    def plot_model(self,a,b,data):
        x = [punto[0] for punto in data]
        y = [punto[1] for punto in data]
        y_pred = [self.modelo(a,b,i) for i in x]
        plt.scatter(x,y)
        plt.plot(x,y_pred,'k--')
        plt.title('Modelo de Regresión Lineal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    # PRUEBAS ========================================================
    def correr_prueba_obtener_trayectoria(self, obtener_trayectoria):
        """ Modulo de pruebas función obtener_trayectoria """
        self.correr_prueba(test_obtener_trayectoria, obtener_trayectoria)
        
    def correr_prueba_calcular_mse(self,mse):
        self.correr_prueba(test_calcular_mse, mse)
        
    def correr_prueba_calcular_derivada_mse(self,dmse):
        self.correr_prueba(test_calcular_derivada_mse,dmse)
        
    def correr_prueba_descenso_gradiente(self,DG_paso_fijo):
        self.correr_prueba(test_descenso_gradiente,DG_paso_fijo)