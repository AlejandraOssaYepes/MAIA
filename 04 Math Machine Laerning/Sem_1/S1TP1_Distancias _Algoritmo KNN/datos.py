"""S1TP1_KNN.ipynb - DATOS
# **Ejercicio N°1:** Vecinos mas Cercanos

***Matemáticas para Machine Learning.***

Semana 1 - Lección 1
"""

# Librerias Básicas
import numpy as np
import pandas as pd
import random


# GENERAR DATOS BASICOS ==================================================
def generar_datos_basicos(n=10, dim=2, nlog=3, seed=None):
    """
    Genera datos aleatorios con coordenadas enteras
    ___________________________________
    Entrada:
    n:    [int] número de datos
    dim:  [int] dimensión de datos
    nlog: [int] vectores en [0, ,10^nlog]]
    seed: [int] semilla de alleatoriedad
    ___________________________________
    Salida:
    V: [2D-array] n datos de dimensión dim
    x: [1D-array] 1 dato de dimensión dim
    """
    # Inicializa aleatoriedad
    if seed is not None: np.random.seed(seed)

    # Familia de vectores
    V = np.around(np.random.rand(n, dim), nlog)*10**(nlog)
    # Vector aparte
    x = np.around(np.random.rand(1, dim), nlog)[0]*10**(nlog)

    # Formato
    V, x = V.astype(int), x.astype(int)
    return V, x


# GENERAR DATOS STANFORD ==================================================
# Constantes
ALT = 'Altura'
PES = 'Peso'
HOM = 'Hombre'
MUJ = 'Mujer'
ED1 = '12'
ED2 = '14'
GEN = 'Genero'
EDA = 'Edad'
IND = 'Indicador'


def llevar_a_ventana(x, ventana, gamma=10):
    """  """
    """ 
    Acerca 'x' a la ventana si no se encuentra en esta
    ___________________________________
    Entrada:
    x:       [float] número inicial
    ventana: [2-tuple] limites de ventana
    gamma:   [int] que tanto se acerca
    ___________________________________
    Salida:
    xn : [float] número acercado a ventana.
    """
    low = min(ventana)
    upp = max(ventana)
    if x < low:   return low - (low - x)/gamma
    elif x > upp: return upp + (x - upp)/gamma
    else:         return x


def generar_datos_crecimiento(n=1000, seed=1):
    """
    Genera datos a partir de la infromación de Stanford
    https://www.stanfordchildrens.org/es/topic/default?id=normalgrowth-90-P04728
    ___________________________________
    Entrada:
    n:    [int] Número de datos a generar
    seed: [int] Semilla de aletoriedad
    ___________________________________
    Salida:
    df : [pd.DataFrame] Tabla con información de los datos.
    """
    info = {
        MUJ: {
          ED1: {
              ALT: {'avg': 151, 'var': 8.5, 'max': 163, 'min': 140},
              PES: {'avg': 43, 'var': 9.5, 'max': 62, 'min': 30}
          },
          ED2: {
              ALT: {'avg': 158, 'var': 10, 'max': 172, 'min': 150},
              PES: {'avg': 60, 'var': 16, 'max': 79, 'min': 38}
          }
        },
        HOM: {
          ED1: {
              ALT: {'avg': 150, 'var': 10, 'max': 162, 'min': 138},
              PES: {'avg': 40, 'var': 8, 'max': 59, 'min': 30}
          },
          ED2: {
              ALT: {'avg': 162, 'var': 12, 'max': 177, 'min': 149},
              PES: {'avg': 58, 'var': 15, 'max': 78, 'min': 39}
          }
        }
    }

    # Generar datos ========================================
    np.random.seed(seed)
    data = []
    # Itera sobre cada genero (Hombre/Mujer)
    for sex, info_sex in info.items():
        # Itera sobre cada edad (12/14)
        for year, info_it in info_sex.items():
            # Itera sobre cada característica (Altura /Peso)
            dat = {}
            for key, val in info_it.items():
                # Generar Datos
                dat[key] = np.around(np.random.normal(val['avg'], val['var'], n), 1)

            # Guardar datos en formato
            for i in range(n):
                new = [sex, year] + [dat[key][i] for key in info_it.keys()]
                data.append(new)

    # Mezclar lista
    random.seed(seed)
    random.shuffle(data)

    df = pd.DataFrame(data, columns=[GEN, EDA, ALT, PES])

    # Construir Indicador
    df[IND] = (7*df[PES] + 3*df[ALT])/50
    return df


def datos():
    """ Genera los datos para el problema del laboratorio """
    seed = 1
    datos_conocidos = generar_datos_crecimiento(n=350, seed=seed)
    datos_desconocidos = generar_datos_crecimiento(n=50, seed=seed+1)
    datos_comparacion = datos_desconocidos.copy()
    # datos_desconocidos[GEN] = None
    datos_desconocidos[EDA] = None
    return datos_conocidos, datos_desconocidos, datos_comparacion




