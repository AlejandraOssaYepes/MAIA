"""S2TP2 - Matrices Positivas Definidas.ipynb - UTILS

# **Ejercicio N°2:**  Matrices Positivas Definidas

***Matemáticas para Machine Learning.***

Semana 2 - Tarea 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
from sympy import Matrix

# Soluciones
from pruebas import test_positiva_definida_vprop
from pruebas import test_positiva_definida_det
from pruebas import test_positiva_definida_pivots

# Otras
from datos import generar_matrices_balanceadas
from mtr import manipulate_ipython


class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
               
        # DATOS
        self.generar_matrices_balanceadas = generar_matrices_balanceadas
        
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


    def mostrar_resultados(self, Alist, test_fun, matrices=True, valores=True):
        """
        Prueba la función test_fun para verificar cada matriz A en Alist por la
        propiedad de ser positiva definida
        ___________________________________
        Entrada:
        Alist:    [1D-array] Lista de matrices
        test_fun: [function] función que identifica si una matriz es p.d.
        matrices: [bool] Muestra matrices usadas
        calores:  [bool] Muestra valores de prueba encontrados
        ___________________________________
        Salida:
        Muestra resultados en consola
        """
        for A in Alist:
            ### USO FUNCIÓN CREADA
            out, vals = test_fun(A)
            ### ==================

            # Matriz
            if matrices:
                print('\n','\n'.join(['\t'.join([str(cell) for cell in row]) for row in A]), '\n', sep='')

            # Resultado
            msj = 'ES' if out else 'NO ES'
            print(f'La matriz {msj} positiva definida')

            # Valores propios
            if valores:
                print(f'Los valores obtenidos para esta prueba son:')
                vals = np.array(vals)
                if len(vals.shape) == 1:
                    [print(round(v, 3)) for v in vals]
                else:
                    print(vals)
            print(self.PLINE)

        
    # PRUEBAS ========================================================
    def correr_prueba_positiva_definida_vprop(self, positiva_definida_vprop):
        """ Modulo de pruebas función obtener_forma_simetrica """
        self.correr_prueba(test_positiva_definida_vprop, positiva_definida_vprop)
        
        
    def correr_prueba_positiva_definida_det(self, positiva_definida_det):
        """ Modulo de pruebas función obtener_forma_simetrica """
        self.correr_prueba(test_positiva_definida_det, positiva_definida_det)
        
       
    def correr_prueba_positiva_definida_pivots(self, positiva_definida_pivots):
        """ Modulo de pruebas función obtener_forma_simetrica """
        self.correr_prueba(test_positiva_definida_pivots, positiva_definida_pivots)
        
   
