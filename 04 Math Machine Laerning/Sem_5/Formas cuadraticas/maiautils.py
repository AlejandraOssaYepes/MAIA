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
from scipy.spatial import ConvexHull

# Dibujar
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
#from shapely.geometry.polygon import Polygon

# Visualización
from IPython.display import display, HTML

# Pruebas
from pruebas import test_fun_cuadratica

# Otras
from mtr import manipulate_ipython

class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
        
        # VISUALIZAR
        
        # DATOS
        
        
    # BASICO ===========================================================
    def correr_prueba(self, test_fun, fun, args=None, kwargs=None):
        """
        Corre prueba para una función. Muestra progreso en mensajes.
        ___________________________________
        Entrada:
        test_fun: [function] Modulo de prueba especifico a la función
        fun:      [function] Función a probar
        """
        print(self.PLINE, 'Verificando errores...\n\n', sep='\n')
        
        if args is not None: 
            test_fun(fun, args=args)
        elif kwargs is not None: 
            test_fun(fun, kwargs=kwargs)
        else:
            test_fun(fun)
        print('\n\nSin Errores', self.PLINE, sep='\n')
        
    
    # EXPLICACIÓN ======================================================
    def plot_example_convex_set(self):
        """
        Genera una secuencia aleatoria de 15 puntos. Luego genera una gráfica
        de su casco convexo y de un ejemplo de conjunto no convexo.
        ___________________________________
        Salida:
        fig: [Graphics] Gráfica que compara los puntos iniciales,
                        su casco convexo y un conjunto no convexo.
        """
        points = np.random.randint(0, 10, size=(15, 2))  # Random points in 2-D

        hull = ConvexHull(points)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 3))

        for ax in (ax1, ax2, ax3):
            ax.plot(points[:, 0], points[:, 1], '.', color='k')
            if ax == ax1:
                ax.set_title('Given points')
            elif ax==ax2:
                ax.set_title('Convex hull')
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 'b')
                ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
            else:
                ax.set_title('Non Convex hull')
                idx = np.random.randint(10, size=6)
                polygon = Polygon(points[idx,:])
                x,y = polygon.exterior.xy
                ax.plot(x, y, color='b')
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
        plt.show()

        
    # PRUEBAS ========================================================
    def correr_prueba_fun_cuadratica(self, fun_cuadratica):
        """ Modulo de pruebas función fun_cuadratica """
        self.correr_prueba(test_fun_cuadratica, fun_cuadratica)


        
        
        
        