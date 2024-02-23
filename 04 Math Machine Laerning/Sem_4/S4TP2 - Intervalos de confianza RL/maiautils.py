"""S4TP2 - Intervalos de confianza.ipynb - UTILS

# **Ejercicio N°2:** 

***Matemáticas para Machine Learning.***

Semana 4 - Actividad 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# PRUEBAS
from pruebas import test_regresion_lineal 
from pruebas import test_intervalo_confianza_betas
from pruebas import test_intervalos_confianza

# AUXILIAR
from mtr import manipulate_ipython

# GENERAR MATRICES PD ==================================================
class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'

        # Datos
        
        # Visualización

    def plot_confidence_intervals(self,dic_ic):
        list_intercept = dic_ic['Intercepto']
        list_slope = dic_ic['Pendiente']
        
        x = range(1,len(list_intercept)+1)
        
        down_intercept = [ic[0] for ic in list_intercept]
        up_intercept = [ic[1] for ic in list_intercept]
        
        center_intercept = [(i+j)/2 for i,j in zip(up_intercept,down_intercept)]
        
        down_slope = [ic[0] for ic in list_slope]
        up_slope = [ic[1] for ic in list_slope]
        
        center_slope = [(i+j)/2 for i,j in zip(up_slope,down_slope)]
        
        fig = plt.figure(1, figsize=(12,5))
        plt.subplot(1, 2, 1)
        plt.fill_between(x, up_intercept, down_intercept,
                        color='b', alpha=.5)
        plt.plot(x, center_intercept, 'o', color='b')
        plt.tight_layout()
        plt.grid()
        plt.title('Confidence Intervals for Intercept')
        plt.xlabel("Sample")
        
        plt.subplot(1, 2, 2)
        plt.fill_between(x, up_slope, down_slope,
                        color='g', alpha=.5)
        plt.plot(x, center_slope, 'o', color='g')
        plt.tight_layout()
        plt.grid()
        plt.title('Confidence Intervals for Slope')
        plt.xlabel("Sample")
        
        plt.show()
        
    def plot_data(self,data):
        fig = px.scatter(data, x='X', y='Y')

        fig.update_layout(
        font_family="Courier New",
        font_color="darkblue",
        title_font_family="Times New Roman",
        title_font_color="black",
        legend_title_font_color="black") 

        fig.update_xaxes(title_text='Predictor')
        fig.update_yaxes(title_text='Objetivo')
        return fig
    
    def generar_datos(self, a, b, Xmu, Xstd, std_error, n=100, seed=None):
        """ 
        Genera n datos Y = a + bX + error
        ___________________________________
        Entrada:
        a: [float] intercepto
        b: [float] Pendiente
        Xmu:  [float] Media de los datos predictores
        Xstd: [float] Desviación estándar de los datos predictores
        std_error: [float] Desviación estándar del error
        n:    [int] número de datos a generar
        seed: [int] semilla de aleatoriedad
        ___________________________________
        Salida:
        df : [pd.DataFrame] Datos
        
        """
        
        if seed is not None:
            np.random.seed(seed)
            
        X = np.random.normal(Xmu, Xstd, n)
        ui = np.random.normal(0, std_error, n)

        Y = b*X + a + ui
        
        df = pd.DataFrame(data=[X,Y]).T
        df.columns = ['X','Y']
    
        return df
        
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
    # 1) =============================================================
    def correr_prueba_regresion_lineal(self, regresion_lineal):
        """ Modulo de pruebas función crear_histograma """
        self.correr_prueba(test_regresion_lineal, regresion_lineal)
    
    # 2) =============================================================
    def correr_prueba_intervalo_confianza_betas(self, intervalo_confianza_betas):
        """ Modulo de pruebas función crear_histograma """
        self.correr_prueba(test_intervalo_confianza_betas, intervalo_confianza_betas)
        
    # 3) =============================================================
    def correr_prueba_obtener_intervalos_confianza(self, obtener_intervalos_confianza):
        """ Modulo de pruebas función crear_histograma """
        self.correr_prueba(test_intervalos_confianza, obtener_intervalos_confianza)