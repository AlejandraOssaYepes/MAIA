"""S4TP3 - DesigualdadesProbabilisticas.ipynb - UTILS

# **Ejercicio N°3:** 

***Matemáticas para Machine Learning.***

Semana 4 - Actividad 3
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
import sys
import random

from mtr import manipulate_ipython
from soluciones import *

class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'

        # Datos
        
        # Visualización

    def generar_datos_markov(self,n=100):
        x  = []
        for i in range(100):
            p = random.uniform(0.05, 0.99)
            p = np.round(p,2)
            q = random.uniform(0.05, 0.99)
            q = np.round(q,2)
            x.append([p,q])
        return x
    
    def generar_datos_chernoff(self,n=100):
        x  = []
        for i in range(100):
            delta = random.uniform(0.001, 0.5)
            delta = np.round(delta,4)
            ep = random.uniform(0.001, 0.2)
            ep = np.round(ep,4)
            x.append([delta,ep])
        return x

    def calificar_cota(self,c,func_tentativa):
        """
        Función que verifica que las soluciones del estudiante.
        """
        sol_tentativa_correcta = False

        x = self.generar_datos_markov() if c==1 else self.generar_datos_chernoff()
        for i in x:
            p = i[0]
            q = i[1]

            sol_real = cota_markov(p,q) if c==1 else cota_chernof(p,q)
            sol_tentativa = func_tentativa(p,q)

            if (sol_real == sol_tentativa).all():
                sol_tentativa_correcta = True
            
            if c == 1:
                assert sol_tentativa_correcta, f"Error en la prueba. La cota de Markov no es correcta para una moneda con probabilidad de exito {p} y un porcentaje deseado de caras de {q}."
            else:
                assert sol_tentativa_correcta, f"Error en la prueba. La cantidad de datos es incorrecta para una precisión de {q} y confianza de {1-p}."

    def calificar_cota_markov(self,func_tentativa):
        """
        Verifica la solución de la función *cota_markov*
        """
        self.calificar_cota(1,func_tentativa)

    def calificar_cota_chernof(self,func_tentativa):
        """
        Verifica la solución de la función *cota_chernof*
        """
        self.calificar_cota(0,func_tentativa)

    @staticmethod
    def manipulate_ipython(ipython):
        """ 
        Cambia el traceback para mostrar únicamente los errores obtenidos por Asserts
        En caso de que se encuentre en estra configuración cambia a la configuración inicial 
        ___________________________________
        Entrada:
        ipython:    [ipykernel] Kernel utilizado en la sesión actual
        ___________________________________
        Salida:
        toggle_traceback: [function] Función para manipular el traceback

        """
   
        if 'TRACEDEFAULT' not in globals():
            TRACEDEFAULT = ipython.showtraceback

        def hide_traceback(exc_tuple=None, filename=None, tb_offset=None,
                        exception_only=False, running_compiled_code=False):
            """ Oculta el reporte predeterminado de python respecto al error encontrado. """
            etype, value, tb = sys.exc_info()
            value._cause_ = None  # suppress chained exceptions
            return ipython._showtraceback(etype, value, ipython.InteractiveTB.get_exception_only(etype, value))


        def toggle_traceback():
            """ Cambia el reporte de traceback """
            if ipython.showtraceback == TRACEDEFAULT:
                ipython.showtraceback = hide_traceback
            else:
                ipython.showtraceback = TRACEDEFAULT

        return toggle_traceback