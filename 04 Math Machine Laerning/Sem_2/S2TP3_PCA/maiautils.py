"""S2TP2 - PCA.ipynb - UTILS

# **Ejercicio N°3:**  PCA

***Matemáticas para Machine Learning.***

Semana 2 - Tarea 3
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
from sympy import Matrix

# Soluciones
from pruebas import test_proyeccion
from pruebas import test_PCA
from pruebas import test_implementar_PCA

# Otras
from datos import generar_datos
from datos import generar_datos_distribuidos
from datos import generar_matrices_pd

from mtr import manipulate_ipython


class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
               
        # DATOS
        self.generar_datos = generar_datos
        self.generar_datos_distribuidos = generar_datos_distribuidos
        self.generar_matrices_pd = generar_matrices_pd
        
    # BASICO ===========================================================
    def correr_prueba(self, test_fun, fun, args=None):
        """
        Corre prueba para una función. Muestra progreso en mensajes.
        ___________________________________
        Entrada:
        test_fun: [function] Modulo de prueba especifico a la función
        fun:      [function] Función a probar
        args:     [list] Lista de argumentos adicionales de la función
        """
        print(self.PLINE, 'Verificando errores...\n\n', sep='\n')
        if args is None:
            test_fun(fun)
        else:
            test_fun(fun, *args)
        print('\n\nSin Errores', self.PLINE, sep='\n')
        

    def correr_prueba_proyeccion(self, proyeccion):
        """ Modulo de pruebas función proyeccion """
        self.correr_prueba(test_proyeccion, proyeccion)
        
    def correr_prueba_PCA(self, PCA):
        """ Modulo de pruebas función PCA """
        self.correr_prueba(test_PCA, PCA)
        
    def correr_prueba_implementar_PCA(self, implementar_PCA, args):
        """ Modulo de pruebas función implementar_PCA """
        self.correr_prueba(test_implementar_PCA, implementar_PCA, args)
     
    
