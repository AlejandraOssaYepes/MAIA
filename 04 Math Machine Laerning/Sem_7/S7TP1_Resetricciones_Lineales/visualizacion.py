""" SXTPY_NombreCuaderno.ipynb - VISUALIZACION """

# LIBRERIAS ================================================
# Basicas
import numpy as np
import pandas as pd
from copy import copy

# Dibujar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

COLORSCALES = ['Plasma', 'viridis', 'Blues'] 
tmp = list(px.colors.named_colorscales())
[tmp.remove(it) for it in COLORSCALES if it in tmp]
np.random.shuffle(tmp)
COLORSCALES += tmp 

def grilla_evaluacion(xlims, ylims, f, n):
    x, y = np.linspace(*xlims, num=n), np.linspace(*ylims, num=n)
    X, Y = np.meshgrid(x, y)
    z = f(X,Y)
    pass

def dibujarCN(xlims, ylims, f, fig=None, n=50, title='', 
              ngrid=None, showscale=False, 
              layout=LAYOUT, colorscale=COLORSCALES[0]):
    """ 
    Dibuja las curvas de nivel de la función f: R x R -> R en el recuadro [xlims] x [ylims]
    ___________________________________
    Entrada:
    xlims: [1D-array] Limites en el eje x
    ylims: [1D-array] Limites en el eje y
    f:     [function] función a evaluar
    fig:   [figure] Figura sobre la cual dibujar
    n:     [int] número de puntos a evaluar
    title: [str] Titulo de imágen
    ngrid: [int] número de lineas en grilla
    showscale:  [bool] Mostrar barra de escala?
    layout:     [dict] estilo de imagen
    colorscale: [str] escala de color
    ___________________________________
    Salida:
    fig : [plotly.Figure] Figura 3D
    """
    # Grilla & Evaluación
    x, y = np.linspace(*xlims, num=n), np.linspace(*ylims, num=n)
    X, Y = np.meshgrid(x, y)
    z = f(X,Y)

    # Dibuja
    if fig is None: fig = go.Figure(data=go.Contour(z=z, x=x, y=y, showscale=showscale,
                                                    contours_coloring='heatmap', colorscale=colorscale))
        
    layout = copy(layout) ; layout.title = f'{title}'
    fig.update_layout(layout)
    
    if ngrid is not None: fig = add_manual_grid(fig, xlims, ylims, n=ngrid, line_width=2)

    return fig
    

def dibujar3D(xlims, ylims, f, fig=None, n=50, title='', 
              showscale=False, cn=False,
              layout=LAYOUT, colorscale=COLORSCALES[0]):
    """ 
    Dibuja función f: R x R -> R en el recuadro [xlims] x [ylims]
    ___________________________________
    Entrada:
    xlims: [1D-array] Limites en el eje x
    ylims: [1D-array] Limites en el eje y
    f:     [function] función a evaluar
    fig:   [figure] Figura sobre la cual dibujar
    n:     [int] número de puntos a evaluar
    cn:    [bool] Dibujar curvas de nivel?
    showscale:  [bool] Mostrar barra de escala?
    layout:     [dict] estilo de imagen
    colorscale: [str] escala de color
    ___________________________________
    Salida:
    fig : [plotly.Figure] Figura 3D
    """
    
    # Grilla & Evaluación
    x, y = np.linspace(*xlims, num=n), np.linspace(*xlims, num=n)
    X, Y = np.meshgrid(x, y)
    z = f(X,Y)
    
    # Dibuja
    if fig is None: fig = go.Figure()
    fig = fig.add_surface(z=z, x=x, y=y, 
                          showscale=showscale, colorscale=colorscale)
    
    layout = copy(layout) ; layout.title = f'{title}'
    fig.update_layout(layout)
    
    # Curvas de nivel
    if cn: fig.update_traces(contours_z=dict(show=True, usecolormap=True,
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
                                     

def add_manual_grid(fig, xlims, ylims, n=10, line_width=1, marker_color='white' ):
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
    layouts : [list] Lista de opciones de visualización 
    ___________________________________
    Salida:
    widg : [ipywidgets.Hbox] Widget con imagenes
    """
    if isinstance(layouts, go.Layout):
        layouts = [layouts]*len(figs)
    
    fig_list = [go.FigureWidget(fig.data, layout=layout) for fig, layout in zip(figs, layouts)]
    
    return widgets.HBox(fig_list)


def vfigures(figs, layouts=LAYOUT):
    """ 
    Widget para visualizar figuras verticalmente
    ___________________________________
    Entrada:
    figs : [1D-array] Lista de figuras plotly
    layouts : [list] Lista de opciones de visualización 
    ___________________________________
    Salida:
    widg : [ipywidgets.Hbox] Widget con imagenes
    """
    if isinstance(layouts, go.Layout):
        layouts = [layouts]*len(figs)

    fig_list = [go.FigureWidget(fig.data, layout=layout) for fig, layout in zip(figs, layouts)]

    return widgets.VBox(fig_list)

# MOSTRAR FIGURAS ================================================

def ajustar_limites(X, lims=None, k=0.05, T=False):
    
    if lims is None: lims=[[0,1],[0,1]]
        
    X = np.array(X).T if T else np.array(X)

    new_lims = []
    for i in range(len(lims)):
        # Ajusta ventana para contener datos
        xlims = lims[i]                 # Limites atuales
        xl = (min(X[i]), max(X[i]))     # Limites datos
        xlims = [min([xl[0], xlims[0]]), max([xl[1], xlims[1]])]
        
        # Ajusta proporción extra
        d = np.abs(xlims[0]- xlims[1])
        xlims = [xlims[0]- d*k, xlims[1] + d*k]
        
        new_lims.append(xlims)      # Guarda
    return new_lims


def set_lims(fig, X, lims=None):
    lims = ajustar_limites(X)
    range_layout = go.Layout(
                        xaxis_range=lims[0], xaxis_autorange=False,
                        yaxis_range=lims[1], yaxis_autorange=False,
                            )
    if len(lims) == 2:  fig.update_yaxes( scaleanchor = "x", scaleratio = 1 ) 

    fig.update_layout(range_layout)
    set_ejes(fig, lims)
    
    
def set_grid(fig, showgrid=True, gridcolor='LightPink'):    
    # Grilla
    fig.update_xaxes(showgrid=showgrid, gridwidth=1, gridcolor=gridcolor)
    fig.update_yaxes(showgrid=showgrid, gridwidth=1, gridcolor=gridcolor)

def set_ejes(fig, lims=None, axisratio=True, color='LightPink'):
    n = 50 # default de np.linalg.linspace
    if lims is not None:
        if len(lims) == 2:      # Caso 2D
            # Extiende ejes
            xlims, ylims = lims
            xlims = np.array(xlims)*100
            ylims = np.array(ylims)*100
            # Dibuja
            fig.add_trace(go.Scatter(x=[0]*n, y=np.linspace(ylims[0],ylims[1]), line=dict(color=color, width=1),
                                mode='lines', name=None, showlegend=False))    
            fig.add_trace(go.Scatter(x=np.linspace(xlims[0],xlims[1]), y=[0]*n, line=dict(color=color, width=1),
                                mode='lines', name=None, showlegend=False)) 
            # Formato 2D  
            if axisratio:
                fig.update_yaxes( scaleanchor = "x", scaleratio = 1 ) 

        elif len(lims) == 3:    # Caso 3D
            set_white3d(fig) # Formato blanco

            n0 = [0]*n
            xlims, ylims, zlims = lims
            x, y, z = np.linspace(*xlims), np.linspace(*ylims), np.linspace(*zlims)

            # EJes
            axlines = {'x':[x, n0, n0],
                       'y':[n0, y, n0],
                       'z':[n0, n0, z]}

            for _, line in axlines.items():
                fig.add_trace(go.Scatter3d(x=line[0], y=line[1], z=line[2], 
                                           line=dict(color='#f8a197', width=5),
                                    mode='lines'))   
            # BLoque exterior
            # abcd
            # efgh
            xi, xf = [xlims[0]]*n, [xlims[1]]*n
            yi, yf = [ylims[0]]*n, [ylims[1]]*n
            zi, zf = [zlims[0]]*n, [zlims[1]]*n

            clines = {
                'ab': [xi, y, zf],
                'ae': [xi, yi, z],
                'bc': [x, yf, zf],
                'bf': [xi, yf, z],
                'cd': [xf, y, zf],
                'cg': [xf, yf, z],
                'da': [x, yi, zf],
                'dh': [xf, yi, z],
                'ef': [xi, y , zi],
                'fg': [x, yf, zi],
                'gh': [xf, y, zi],
                'he': [x, yi, zi]
                }   

            for _, line in clines.items():
                fig.add_trace(go.Scatter3d(x=line[0], y=line[1], z=line[2], 
                                           line=dict(color='#FFEED2', width=5),
                                    mode='lines'))       

def set_white3d(fig):
    fig.update_layout(#plot_bgcolor='rgb(12,163,135)',
          #paper_bgcolor='rgb(12,163,135)'
          #coloraxis={"colorbar": {"x": -0.2, "len": 0.5, "y": 0.8}}, #I think this is for contours
          scene = dict(
                xaxis = dict(
                      backgroundcolor="rgba(0, 0, 0,0)",
                      gridcolor="white",
                      showbackground=True,
                      zerolinecolor="white",),
                yaxis = dict(
                    backgroundcolor="rgba(0, 0, 0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"),
                zaxis = dict(
                    backgroundcolor="rgba(0, 0, 0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",),),
                  )