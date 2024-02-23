
# Básico
from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np

# Dibujar
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.colors as mcolors

from datos import datos_dividir_en_clases

COLORS = [c for c in mcolors.TABLEAU_COLORS.values()]


def create_figure_simple(x, y, title='', name='', color=COLORS[0], fig=None):
    layout = go.Layout(
        title = title,
        xaxis = go.layout.XAxis(
            title = 'Valor',
            showticklabels=True,
            showgrid=False),
        yaxis = go.layout.YAxis(
            title = '',
            showticklabels=False,
            showgrid=False
        ),
    )

    if fig is None:  
      fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=x,
            y=y,
            marker=dict(
                color=color,
                size=20,
                line=dict(
                    color='darkgray',
                    width=2
                )
            ),
            name=name,
            showlegend=True
        )
    )

    fig.update_layout(layout)

    return fig


def create_figure(datos, legend=None):
    nc = len(list(set(datos['Y'])))

    if legend is None:
        legend = [f'Muestra {i}' for i in range(nc)]
    
    cl = datos_dividir_en_clases(datos)
    
    fig = None
    for y, x in cl.items():
        
        fig = create_figure_simple(x, [y]*len(x), title='', name=legend[y], color=COLORS[y], fig=fig)

    return fig



