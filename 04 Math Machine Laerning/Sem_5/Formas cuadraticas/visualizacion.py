"""S5TP3 - formas cuadráticas.ipynb - VISUALIZACION

# **Ejercicio N°1:** Derivadas direccionales

***Matemáticas para Machine Learning.***

Semana 5 - Actividad 1
"""

# Librerias
import numpy as np
import pandas as pd
from copy import copy


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# LIBRERIAS ================================================

# Estilo básico
LAYOUT = go.Layout(
    title_text=None, title_x=0.5,
    margin=dict(l=20, r=20, t=20, b=20),
    autosize=False, width=500, height=500,
    xaxis_title="$X$", 
    yaxis_title="$Y$",
    paper_bgcolor='rgba(255,255,255, 1)',
    plot_bgcolor='rgba(255,255,255, 1)'
)


def dibujarCN(xlims, ylims, f, n=50, title='', fig=None, showscale=False, layout=LAYOUT):
    """ 
    Dibuja las curvas de nivel de la función f: R x R -> R en el recuadro [xlims] x [ylims]
    ___________________________________
    Entrada:
    xlims: [1D-array] Limites en el eje x
    ylims: [1D-array] Limites en el eje y
    f: [function] función a evaluar
    n: [int] número de puntos a evaluar
    ___________________________________
    Salida:
    fig : [plotly.Figure] Figura 3D
    """
    # Grilla
    x, y = np.linspace(*xlims, num=n), np.linspace(*xlims, num=n)
    X, Y = np.meshgrid(x, y)

    # Evalua
    z = f(X,Y)

    # Dibuja
    if fig is None:
        fig = go.Figure(data=go.Contour(z=z, x=x, y=y, showscale=showscale,
                                       contours_coloring='heatmap'))
        
    layout = copy(layout)
    layout.title = f'Función : {title}'
    fig.update_layout(layout)

    return fig
    

def dibujar3D(f, xlims, ylims, n=50, title=None, contours=False, showscale=False, layout=LAYOUT,
             args=None, kwargs=None):
    """ 
    Dibuja función f: R x R -> R en el recuadro [xlims] x [ylims]
    ___________________________________
    Entrada:
    xlims: [1D-array] Limites en el eje x
    ylims: [1D-array] Limites en el eje y
    f: [function] función a evaluar
    n: [int] número de puntos a evaluar
    ___________________________________
    Salida:
    fig : [plotly.Figure] Figura 3D
    """
    # Grilla
    x, y = np.linspace(*xlims, num=n), np.linspace(*xlims, num=n)
    X, Y = np.meshgrid(x, y)
    
    # Evalua
    if kwargs is not None:
        z = f(X,Y, **kwargs)
    elif args is not None:
        z = f(X,Y, *args)
    else:
        z = f(X,Y)
        
    # Dibuja
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y,showscale=showscale)])
    
    layout = copy(layout)
    if title is not None: layout.title = f'{title}'
    fig.update_layout(layout)
    
    # Contornos
    if contours:
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))

    return fig

