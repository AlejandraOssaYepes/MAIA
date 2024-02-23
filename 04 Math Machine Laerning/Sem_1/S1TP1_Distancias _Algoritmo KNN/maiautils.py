"""S1TP1 - Distancias y algoritmo KNN.ipynb - UTILS

# **Ejercicio N°1:** Distancias y algoritmo KNN

***Matemáticas para Machine Learning.***

Semana 1 - Tarea 1
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
from pruebas import pretty_print
from pruebas import test_norm
from pruebas import test_k_mas_cercanos
from pruebas import test_k_mas_cercanos_indice
from pruebas import test_encontrar_etiqueta
from pruebas import test_completar_tabla

# Otras
from datos import datos
from datos import GEN, EDA
from datos import generar_datos_basicos
from visualizar import visualizar_cercanos
from mtr import manipulate_ipython


class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
        
        # VISUALIZAR
        self.visualizar_cercanos = visualizar_cercanos
        
        # DATOS
        datos_conocidos, datos_desconocidos, datos_comparacion = datos()
        self.DATOS_KNN = datos_conocidos, datos_desconocidos
        
        
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

        
    # PRUEBAS ========================================================
    def correr_prueba_norm(self, norm):
        """ Modulo de pruebas función norm """
        self.correr_prueba(test_norm, norm)
        
    def correr_prueba_k_mas_cercanos(self, k_mas_cercanos):
        """ Modulo de pruebas función k_mas_cercanos """
        self.correr_prueba(test_k_mas_cercanos, k_mas_cercanos)
        
    def correr_prueba_k_mas_cercanos_indice(self, k_mas_cercanos_indice):
        """ Modulo de pruebas función k_mas_cercanos_indice """
        self.correr_prueba(test_k_mas_cercanos_indice, k_mas_cercanos_indice)
        
    def correr_prueba_encontrar_etiqueta(self, encontrar_etiqueta):
        """ Modulo de pruebas función encontrar_etiqueta """
        self.correr_prueba(test_encontrar_etiqueta, encontrar_etiqueta)
        
    def correr_prueba_completar_tabla(self, completar_tabla):
        """ Modulo de pruebas función completar_tabla """
        self.correr_prueba(test_completar_tabla, completar_tabla)
     
        
        
        

        
        
        
        