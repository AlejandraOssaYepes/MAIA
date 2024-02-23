"""S1TCP_Word2Vec.ipynb - UTILS
# **Taller N°1:** Word2Vec

***Matemáticas para Machine Learning.***

Semana 1 - Taller Práctico
"""

# LIBRERIAS ================================================

# Visualización
import pandas as pd
from IPython.display import display, HTML
from tqdm.notebook import tqdm


# Adicionales
from mtr import manipulate_ipython
from visualizacion import reduccion_dimensionalidad

# Pruebas
from pruebas import test_cos_sim
from pruebas import test_palabras_cercanas
from pruebas import test_palabras_lejanas
from pruebas import test_mas_cercanas_en_vocabulario


class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
        
        # VISUALIZAR
        self.reduccion_dimensionalidad = reduccion_dimensionalidad
               
        
    # BASICO ===========================================================
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

    @staticmethod
    def mostrar_tabla(df, transpose=True):
        """ Mostrar los resultados de una tabla
        ___________________________________
        Entrada:
        df : [pd.DataFrane/dict] tabla a mostrar
        transpose: [bool] Transponer visualización
        """
        if isinstance(df, dict): df = pd.DataFrame(df)
        if transpose: df = df.T
        return display(HTML(df.to_html()))
    
    # PRUEBAS ========================================================
    def correr_prueba_cos_sim(self, cos_sim, args):
        """ Modulo de pruebas función cos_sim """
        self.correr_prueba(test_cos_sim, cos_sim, args)

    def correr_prueba_palabras_cercanas(self, palabras_cercanas, args):
        """ Modulo de pruebas función palabras_cercanas """
        self.correr_prueba(test_palabras_cercanas, palabras_cercanas, args)

    def correr_prueba_palabras_lejanas(self, palabras_lejanas, args):
        """ Modulo de pruebas función palabras_lejanas """
        self.correr_prueba(test_palabras_lejanas, palabras_lejanas, args)

    def correr_prueba_mas_cercanas_en_vocabulario(self, mas_cercanas_en_vocabulario, args):
        """ Modulo de pruebas función mas_cercanas_en_vocabulario """
        self.correr_prueba(test_mas_cercanas_en_vocabulario, mas_cercanas_en_vocabulario, args)



