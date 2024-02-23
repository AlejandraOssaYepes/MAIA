"""S1TP1_KNN.ipynb - VISUALIZACION
# **Ejercicio N°1:** Vecinos mas Cercanos

***Matemáticas para Machine Learning.***

Semana 1 - Lección 1
"""

# Librerias
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def crear_plt_figure(dim :int, figsize=(8 ,8)):
    """
    Crea figura
    ___________________________________
    Entrada:
    dim: [int:2,3] Dimensión de trabajo
    figsize: [2-tuple] Tamaño de figura
    ___________________________________
    Salida:
    ax: imagen
    """
    if dim == 2:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    elif dim == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
    else:
        return 'Dimensión incorrecta'
    return ax


def set_ax_labels(ax):
    """
    Adiciona etiquetas de ejes
    ___________________________________
    Entrada:
    ax: imagen

    """
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    try:
        ax.set_zlabel('$Z$')
    except AttributeError:
        pass


def visualizar_cercanos(x, V, C=None,
                        ax=None, figsize=(8 ,8), s=10,
                        norm=None, k=2, k_mas_cercanos=None):
    """
    Crea una figura señalando el vector cental x y los elementos cercanos
    ___________________________________
    Entrada:
    x: [1D-array] Vector central
    V: [2D-array] Lista de vectores a buscar
    C: [2D-array] Lista de vectores de V cercanos a x
    figsize: [2-tuple] Tamaño de figura
    s: [int] tamaño de puntos
    k: [int] número de elementos cercanos (C=None)
    norm: [function] Función de norma a usar (C=None)
    k_mas_cercanos: [function] Función de encontrar cercanos a usar (C=None)
    ___________________________________
    Salida:
    ax: imagen
    """
    # Inicialización Básica
    ctab = list(mcolors.TABLEAU_COLORS.keys())
    dim = len(x)
    x = np.array(x)
    V = np.array(V)

    # Crea cercanos si no hay
    if C is None:
        C = k_mas_cercanos(x=x, V=V, k=k,
                           norm=norm)
    else:
        C = np.array(C)

    # Elimina elementos de V en C
    C_str = [str(it) for it in C]
    V_new = np.array([v for v in V if str(v) not in C_str])

    # Inicializa imagen
    if dim > 3:
        return 'No se puede gráficar en dimensiones mayores a 3'
    else:
        if ax is None: ax = crear_plt_figure(dim, figsize)

    V_new = [V_new[: ,i] for i in range(V_new.shape[1])]
    C_new = [C[: ,i] for i in range(C.shape[1])]
    # Dibuja
    ax.scatter(*V_new, c=ctab[0], s=s)
    ax.scatter(*C_new, c=ctab[1], s= 2 *s)
    ax.scatter(*x, c=ctab[3], s= 2 *s)

    ax.legend(['$V$', '$kNN$', '$x$'])
    try:
        ax.set_aspect('equal')
    except AttributeError:
        pass
    return ax
