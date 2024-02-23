"""S4TP3 - DesigualdadesProbabilisticas.ipynb - UTILS

# **Ejercicio N°3:** 

***Matemáticas para Machine Learning.***

Semana 4 - Actividad 3
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
import pandas as pd


def cota_markov(q,p):
    """ 
    Retorna la cota de Markov para la probabilidad de obtener
    cara más de q por ciento para una moneda con probabilidad de exito p,
    para n experimentos.
    ___________________________________
    Entrada:
    q: [float] Porcentaje de los intentos que se desea, sean cara.
    p: [float] Probabilidad de exito de la moneda .
    ___________________________________
    Salida:
    cota: [float] Cota de Markov para la probabilidad descrita.
    """  
    cota = p/q
    return cota

def cota_chernof(delta,epsilon):
    """ 
    Retorna la cantidad mínima de datos que garantiza una diferencia como máximo de 
    epsilon entre p muestral y p poblacional, con confianza de 1-delta.
    ___________________________________
    Entrada:
    epsilon: [float] Precisión. Máxima diferencia entre p muestral y p poblacional.
    delta: [float] Significancia.
    ___________________________________
    Salida:
    n: [float] Cota de Markov para la probabilidad descrita.
    """  
    n = np.log(2/delta)/(2*epsilon*epsilon)
    n = np.ceil(n)
    return n