"""S3TP2 - Fuentes de incertidumbre.ipynb - UTILS
# **Ejercicio N°2:** Fuentes de incertidumbre

***Matemáticas para Machine Learning.***

Semana 3 - Lección 2
"""


# LIBRERIAS ================================================
# Básicas
import numpy as np
import scipy
from datos import datos_dividir_en_clases

# 3.2) ================================================
def crear_histograma_sol(x, ngrupos=5):
    """ 
    Crea un histograma  
    ___________________________________
    Entrada:
    x:   [1D-array] Datos unidimensionales
    ngrupos [int] número de grupos a dividir el histograma
    ___________________________________
    Salida:
    valores : [1D-array] lista de valores representativos del histograma 
    conteos : [1D-array] Lista de conteos sobre cada valor representativo
    """
    
    valores, conteos = [], []
    ancho = [min(x), max(x)]
    gap = (ancho[1] - ancho[0])/ngrupos
    
    for i in range(ngrupos):
        
        sub = [ancho[0]+i*gap, ancho[0]+(i+1)*gap]
        x_sub = [xi for xi in x if xi > sub[0] and xi <= sub[1] ]
        valores.append( (sub[0]+sub[1])/2 )
        conteos.append(int(len(x_sub)))
        
    return valores, conteos


def parametros_distribucion_sol(x):
    """ 
    Obtiene parámetros de media y varianza sobre una distribución
    ___________________________________
    Entrada:
    x:   [1D-array] Datos unidimensionales
    ___________________________________
    Salida:
    mu : [float] Valor de la media 
    var: [float] Valor de la varianza
    """
    return np.mean(x), np.var(x)


def obtener_todos_los_parametros_sol(datos):
    """ 
    Obtiene todos los parámetros de las distribuciones en datos
    ___________________________________
    Entrada:
    datos:   [2D-array] Datos unidimensionales, la segunda coordenada es la etiqueta
    ___________________________________
    Salida:
    mus : [1D-array] Lista con las distintas medias
    sigmas: [1D-array] Lista con las distintas varianzas
    """
    #X, Y = datos_dividir_en_clases(datos)
    mus, sigmas = [], []
    Xclase = datos_dividir_en_clases(datos)
    for i, x in Xclase.items():
        mu, sigma = parametros_distribucion_sol(x)
        mus.append(mu)
        sigmas.append(sigma)

    return mus, sigmas


def fdp_sol(x, datos, dist=0):
    """ 
    Obtiene la probabilidad de obtener el valor x bajo la distribución dist
    ___________________________________
    Entrada:
    x:   [float] valor del dato 
    dist: [int] Distribución a utilizar (0/1/2)
    ___________________________________
    Salida:
    p: [float] probabilidad de que el dato pertenezca a la distribución
    """

    MUS, SIGMAS = obtener_todos_los_parametros_sol(datos)

    p = scipy.stats.norm(MUS[dist], SIGMAS[dist]).pdf(x)

    return p


def clasificador_distribucion_muestra_sol(x,datos):
    """ 
    Identifica la distribución de la que proviene una muestra
    ___________________________________
    Entrada:
    X:   [Float] Muestra
    ___________________________________
    Salida:
    dist: [int] Distribución a de la muestra (0/1/2)
    """
    plist = [fdp_sol(x, datos, dist=i) for i in range(2)]
    dist = np.argmax(np.array(plist))
    return dist


def verosimilitud_sol(X, datos,dist=0):
    """ 
    Obtiene la verosimilitud de los datos en X con la distribución dist
    ___________________________________
    Entrada:
    X:   [1D-array] lista de datos de una distribución 
    dist: [int] Distribución a utilizar (0/1/2)
    ___________________________________
    Salida:
    p: [float] verosimilitud de los datos para la distribución
    """
    out = 1
    for x in X:
        out = out*fdp_sol(x, datos, dist=dist)
    return out



def clasificador_distribucion_muestras_sol(X,datos):
    """ 
    Identifica la distribución de la que proviene un conjunto de muestras
    ___________________________________
    Entrada:
    X:   [Float] Muestra
    ___________________________________
    Salida:
    dist: [int] Distribución a de la muestra (0/1/2)
    """
    vsimil = [verosimilitud_sol(X, datos, dist=i) for i in range(2)]
    dist = np.argmax(np.array(vsimil))
    return dist
