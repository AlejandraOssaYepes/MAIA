"""S1TP1_KNN.ipynb - SOLUCIONES
# **Ejercicio N°1:** Vecinos mas Cercanos

***Matemáticas para Machine Learning.***

Semana 1 - Lección 1
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
from tqdm.notebook import tqdm
from datos import GEN, EDA, ALT, PES, IND


# 1) ================================================
def norm_solucion(v, typ=0) -> float:
    """
    Encuentra norma euclidiana de un vector
    ___________________________________
    Entrada:
    v: [1D-array] Vector
    ___________________________________
    Salida:
    nrm: [float]  Norma del vector
    """
    if typ:
        return np.linalg.norm(v)
    else:
        return sum([vi**2 for vi in v])**(1/2)


# 2) ================================================
def k_mas_cercanos_solucion(x, V, k=1, norm=norm_solucion):
    """
    Encuentra vecinos cercanos
    ___________________________________
    Entrada:
    x: [1D-array] Vector central
    V: [2D-array] Familia de vectores
    k: [int] # de vectores cercanos
    norm: [function] función de norma para utilizar
    ___________________________________
    Salida:
    cercanos: [2D-array] Lista de k vectores.
    """
    cercanos = sorted(V, key=lambda v: norm(np.array(v)-x))
    return cercanos[:k]


# 3a) ================================================
def k_mas_cercanos_indice_solucion(x, V, k=1, norm=norm_solucion):
    """
    Encuentra indice de vecinos cercanos
    ___________________________________
    Entrada:
    x: [1D-array] Vector central
    V: [2D-array] Familia de vectores
    k: [int] # de vectores cercanos
    norm: [function] función de norma a utilizar
    ___________________________________
    Salida:
    idx: [list] indice sobre V de los k vectores cercanos.
    """
    cercanos = sorted(enumerate(V), key=lambda iv: norm(np.array(iv[1])-x))
    # Obtiene indices
    idx = [c[0] for c in cercanos]
    return idx[:k]


# 3b) ================================================
def encontrar_etiqueta_solucion(d, datos, k=5, etiqueta=GEN):
    """
    Encuentra etiqueta de un dato usando KNN a partir de una base de datos
    ___________________________________
    Entrada:
    d:     [pandas.Series] dato desconocido (Fila de DataFrame)
    datos: [pandas.DataFrame] base de datos conocida (DataFrame)
    k:     [int] número de vecinos a tener en cuenta
    etiqueta: [GEN/EDA] etiqueta a predecir (Genero/Edad)
    ___________________________________
    Salida:
    label:  [str] etiqueta del dato d
    kinfo: [pd.DataFRame] Información de k-vecinos cecanos
    """
    global GEN, EDA, ALT, PES, IND
    # Identifica vectores de datos y etiquetas
    labels = datos[etiqueta]
    vecs = datos[[ALT, PES, IND]]

    # Identifica vector del dato
    x = d[[ALT, PES, IND]]

    # Encuentra etiquetas mas cercanas
    idx = k_mas_cercanos_indice_solucion(x=x, V=np.array(vecs), k=k)
    klabels = labels[idx]
    kvecs = vecs.iloc[idx]

    # Guarda información de mas cercanos
    kinfo = kvecs.copy()
    kinfo['Etiquetas'] = np.array(klabels)

    # Etiqueta mas frecuente
    label = max(set(klabels), key=list(klabels).count)

    return label, kinfo


# 3c) ================================================
def completar_tabla_solucion(datos_conocidos, datos_desconocidos, k=5):
    """
    Realiza interpolación de datos de una base de datos con etiquetas desconocidas
    a partir de una base de datos con etiquetas conocida
    utilizando el método KNN con parámetro k
    ___________________________________
    Entrada:
    datos_conocidos:    [pandas.DataFrame] base de datos conocida
    datos_desconocidos: [pandas.DataFrame] base de datos desconocida
    k:                  [int] número de vecinos a tener en cuenta
    ___________________________________
    Salida:
    datos_interpolados:  [pandas.DataFrame] base de datos desconocida interpolada
    """
    global GEN, EDA, ALT, PES, IND

    datos_interpolados = datos_desconocidos.copy(deep=True)

    for i in tqdm(range(len(datos_desconocidos))):
        d = datos_interpolados.iloc[i]
        d[EDA] = encontrar_etiqueta_solucion(d, datos=datos_conocidos, k=k, etiqueta=EDA)[0]
        datos_interpolados.iloc[i] = d

    return datos_interpolados
