"""S3TP2 - Fuentes de incertidumbre.ipynb - UTILS
# **Ejercicio N°2:** Fuentes de incertidumbre

***Matemáticas para Machine Learning.***

Semana 3 - Lección 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
from sympy import Matrix

# Pruebas
from pruebas import test_crear_histograma
from pruebas import test_parametros_distribucion
from pruebas import test_obtener_todos_los_parametros
from pruebas import test_fdp
from pruebas import test_clasificador_distribucion_muestra
from pruebas import test_verosimilitud
from pruebas import test_clasificador_distribucion_muestras

# Auxiliar
from datos import generar_datos_estocasticos
from datos import obtener_datos
from datos import datos_dividir_en_clases

from visualizacion import create_figure

from mtr import manipulate_ipython



# GENERAR MATRICES PD ==================================================
class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'

        # Datos
        self.generar_datos_estocasticos = generar_datos_estocasticos
        self.DATOS = obtener_datos()
        self.datos_dividir_en_clases = datos_dividir_en_clases
        
        # Visualización
        self.create_figure = create_figure
    
    # BASICO ========================================================
    def correr_prueba(self, test_fun, fun, args=None, kwargs=None):
        """
        Corre prueba para una función. Muestra progreso en mensajes.
        ___________________________________
        Entrada:
        test_fun: [function] Modulo de prueba especifico a la función
        fun:      [function] Función a probar
        args:   [1D-array] Argumentos de fun ordenados
        Kwargs: [dict] Argumentos de fun por nombre
        """
        print(self.PLINE, 'Verificando errores...\n\n', sep='\n')
        if args is not None:
            test_fun(fun, *args)
        elif kwargs is not None:
            test_fun(fun, **kwargs)
        else:
            test_fun(fun)        
        print('\n\nSin Errores', self.PLINE, sep='\n')
        
      
    
    # PRUEBAS ========================================================
    def correr_prueba_crear_histograma(self, crear_histograma):
        """ Modulo de pruebas función crear_histograma """
        self.correr_prueba(test_crear_histograma, crear_histograma)

   
    def correr_prueba_(self, crear_histograma):
        """ Modulo de pruebas función cos_sim """
        self.correr_prueba(test_crear_histograma, crear_histograma)
        
   
    def correr_prueba_parametros_distribucion(self, parametros_distribucion):
        """ Modulo de pruebas función cos_sim """
        self.correr_prueba(test_parametros_distribucion, parametros_distribucion)
    
   
    def correr_prueba_obtener_todos_los_parametros(self, obtener_todos_los_parametros):
        """ Modulo de pruebas función obtener_todos_los_parametros """
        self.correr_prueba(test_obtener_todos_los_parametros, obtener_todos_los_parametros)
        
   
    def correr_prueba_fdp(self, fdp):
        """ Modulo de pruebas función fdp """
        self.correr_prueba(test_fdp, fdp)
        
        
    def correr_prueba_clasificador_distribucion_muestra(self, clasificador_distribucion_muestra):
        """ Modulo de pruebas función cos_sim """
        self.correr_prueba(test_clasificador_distribucion_muestra, clasificador_distribucion_muestra)
        
   
    def correr_prueba_verosimilitud(self, verosimilitud):
        """ Modulo de pruebas función verosimilitud """
        self.correr_prueba(test_verosimilitud, verosimilitud)

        
    def correr_prueba_clasificador_distribucion_muestras(self, clasificador_distribucion_muestras):
        """ Modulo de pruebas función clasificador_distribucion_muestras """
        self.correr_prueba(test_clasificador_distribucion_muestras, clasificador_distribucion_muestras)

