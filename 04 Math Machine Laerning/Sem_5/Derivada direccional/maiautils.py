"""S5TP1 - Derivadas direccionales.ipynb - UTILS

# **Ejercicio N°1:** Derivadas direccionales

***Matemáticas para Machine Learning.***

Semana 5 - Actividad 1
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np      
import pandas as pd
from copy import copy
from itertools import product

# Dibujar
import matplotlib.pyplot as plt 
#%matplotlib inline
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots
import ipywidgets as widgets
import mpl_toolkits.mplot3d as plt3d

# Visualización
from IPython.display import display, HTML

# Pruebas
from pruebas import test_obtener_trayectoria
from pruebas import test_alpha
from pruebas import test_gamma
from pruebas import test_evaluar_trayectoria
from pruebas import test_dgamma

# Otras
from visualizacion import dibujar3D
from visualizacion import dibujarCN
from visualizacion import hfigures
from visualizacion import dibujar_evaluacion

from mtr import manipulate_ipython

# Estilo básico
LAYOUT = go.Layout(
    title_text=None, title_x=0.5,
    margin=dict(l=20, r=20, t=30, b=20),
    autosize=False, width=500, height=500,
    xaxis_title="$X$", 
    yaxis_title="$Y$",
    paper_bgcolor='rgba(255,255,255, 1)',
    plot_bgcolor='rgba(255,255,255, 1)'
)

class MaiaUtils():
    
    def __init__(self, ipython):
        # Informe de errores
        self.toggle_traceback = manipulate_ipython(ipython)
        self.PLINE = '__________________________________'
        
        # VISUALIZAR
        self.dibujar3D = copy(dibujar3D)
        self.dibujarCN = copy(dibujarCN)
        self.dibujar_evaluacion = dibujar_evaluacion
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
    
    # FUNCIONES AUXILIARES
    def dibujar_trayectoria2D(self,x1,x2,line,fig=None,color=0,layout=LAYOUT):
        """ 
        Dibuja una trayectoria sobre un espacio 2D
        ___________________________________
        Entrada:
        line: [2D-array] Coordenadas de la trayectoria
        fig : [plotly.Figure] Figura de trabajo
        color: [int/str] Referencia númerica del color/ color exacto
        layout: [go.Layout] Configuración de vista
        ___________________________________
        Salida:
        fig : [plotly.Figure] Figura con la trayectoria agregada
        """
        
        # Selecciona el color
        if isinstance(color, int):
            colors = pcolors.DEFAULT_PLOTLY_COLORS
            color = colors[color]
            
        # Crea la figura si no existe
        fig = go.Figure() if fig is None else copy(fig)
            
        # Dibuja
        fig.add_trace(go.Scatter(x=line[0], y=line[1],
                                mode='lines',
                                name=f'{x1}-{x2}',
                                line=dict(color=color, width=4)
                                ))
        
        fig.update_layout(layout)

        return fig

    # PRUEBAS ========================================================
    def correr_prueba_alpha(self, alpha):
        self.correr_prueba(test_alpha, alpha)

    def correr_prueba_gamma(self, gamma):
        self.correr_prueba(test_gamma, gamma)
        
    def correr_prueba_obtener_trayectoria(self, obtener_trayectoria):
        self.correr_prueba(test_obtener_trayectoria, obtener_trayectoria)

    def correr_prueba_evaluar_trayectoria(self, evaluar_trayectoria):
        self.correr_prueba(test_evaluar_trayectoria, evaluar_trayectoria)
            
    def correr_prueba_dalpha(self, dalpha):
        self.correr_prueba(test_dalpha, dalpha)

    def correr_prueba_dgamma(self, dgamma):
        self.correr_prueba(test_dgamma, dgamma)
    
    def correr_prueba_dapprox(self, dapprox):
        self.correr_prueba(test_dapprox, dapprox)
    
    
        

        
        
        
        