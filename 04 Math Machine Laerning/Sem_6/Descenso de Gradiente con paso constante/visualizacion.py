"""S5TP1 - Derivadas direccionales.ipynb - VISUALIZACION

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
    

def dibujar3D(xlims, ylims, f, n=50, title='', contours=False, showscale=False, layout=LAYOUT):
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
    z = f(X,Y)
    
    # Dibuja
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y,showscale=showscale)])
    
    layout = copy(layout)
    layout.title = f'Función : {title}'
    fig.update_layout(layout)
    
    # Contornos
    if contours:
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))

    return fig


def dibujar_evaluacion2D(z, x1, x2, dim=2, fig=None, layout=LAYOUT):
    """ 
    Dibuja trayectoria sobre segmento de recta
    ___________________________________
    Entrada:
    z: [1D-array] Vlores evaluados
    x1: [1D-array] Primer Valor del segmento de recta evaluado
    x1: [1D-array] Segundo Valor del segmento de recta evaluado
    n: [int] número de puntos evaluados
    dim: [int] dimensión a dibujar (2/3)
    layout: [go.Layout] Configuración de vista
    ___________________________________
    Salida:
    fig : [plotly.Figure] evaluaciones de la función en el segmento de recta
    """
    x1, x2, z = np.array(x1), np.array(x2), np.array(z)
    n = len(z)
    title = 'Función evaluada'
    
    if dim == 2:
        # Genera eje x
        dist = np.linalg.norm(x1-x2)
        x = np.linspace(0, dist, n)

        # Organiza en tabla
        df = pd.DataFrame([x,z]).T
        df.columns = ['x','z']
        fig = px.line(df, x='x', y='z', title=title)

    elif dim == 3:
        # Organiza tabla
        df = pd.DataFrame(np.linspace(x1, x2, n))
        df.columns = ['x','y']
        df['z'] = z

        if fig is None:
            fig = px.line_3d(df, x="x", y="y", z="z", title=title)
        else :
            line_marker = dict(color='#101010', width=8)
            fig.add_scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='lines', line=line_marker, name='')
            
    return fig


# AGREGAR GRILLA ================================================
def add_box(lfig, x, y, line_width=1, marker_color='white'):
    lfig.add_trace(go.Scatter(
        x=x,
        y=y,
        opacity=0.5,
        marker_color=marker_color,
        line_width=line_width,
        showlegend=False
    ))
    
def compute_horizontal_lines(x_min, x_max, y_data):
    x = np.tile([x_min, x_max, None], len(y_data))
    y = np.ndarray.flatten(np.array([[a, a, None] for a in y_data]))
    return x, y

def compute_vertical_lines(y_min, y_max, x_data):
    y = np.tile([y_min, y_max, None], len(x_data))
    x = np.ndarray.flatten(np.array([[a, a, None] for a in x_data]))
    return x, y

def n_lines(lims):
    lims = [np.ceil(xlims[0]), np.floor(xlims)]
    dist = 0
    # TERMINAR
    pass
                                     

def add_grid(fig, xlims, ylims, n=10, line_width=1, marker_color='white' ):
    fig = copy(fig)
    hx, hy = compute_horizontal_lines(*xlims, np.linspace(*ylims, n))
    vx, vy = compute_vertical_lines(*ylims, np.linspace(*xlims, n))

    add_box(fig, hx, hy, line_width, marker_color)
    add_box(fig, vx, vy, line_width, marker_color)
    
    return fig



# MOSTRAR FIGURAS ================================================
def hfigures(figs, layouts=LAYOUT):
    """ 
    Widget para visualizar figuras horizontalmente
    ___________________________________
    Entrada:
    figs : [1D-array] Lista de figuras plotly
    height : [int] Altura de las imágenes
    ___________________________________
    Salida:
    fig : [ipywidgets.Hbox] Widget con imagenes
    """
    if isinstance(layouts, go.Layout):
        layouts = [layouts]*len(figs)
    
    fig_list = [go.FigureWidget(fig.data, layout=layout) for fig, layout in zip(figs, layouts)]
    
    return widgets.HBox(fig_list)
