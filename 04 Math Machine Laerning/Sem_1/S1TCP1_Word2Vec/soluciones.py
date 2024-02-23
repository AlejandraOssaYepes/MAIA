"""S1TP1_KNN.ipynb - SOLUCIONES
# **Ejercicio N°1:** Vecinos mas Cercanos

***Matemáticas para Machine Learning.***

Semana 1 - Lección 1
"""

# LIBRERIAS ================================================
# Librerias Básicas
import numpy as np                # Matemáticas
import itertools                  # Manipulación iterables
import pandas as pd
import random
import sys

# Procesamiento de datos
from sklearn.decomposition import TruncatedSVD, PCA

# Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

# ===================================================
# 6 PROBLEMAS =======================================
# ===================================================


# 1) ================================================
def cos_sim_sol(p1, p2, model, typ=0) -> float:
    """
    Obtiene la similitud de coseno entre 2 vectores
    ___________________________________
    Entrada:
    p1, p2: [str] palabras a comparar
    model:  [gensim.Word2Vec] Modelo bajo el cual se compararan las palabras
    typ:    [int:0-2] Tipo de calculo
    ___________________________________
    Salida:
    csim: [float] Similaridad entre los vectores
    """
    v, u = model.wv[p1], model.wv[p2]

    if typ == 0:
        num = sum([vi*ui for vi,ui in zip(v,u)])
        den = (sum([vi**2 for vi in v])**(1/2))*(sum([ui**2 for ui in u]))**(1/2)
    elif typ == 1:
        num = np.dot(v, u)
        den = (sum([vi**2 for vi in v])**(1/2))*(sum([ui**2 for ui in u]))**(1/2)
    elif typ == 2:
        num = sum([vi*ui for vi,ui in zip(v,u)])
        den = np.linalg.norm(v)*np.linalg.norm(u)
    elif typ == 3:
        num = np.dot(v, u)
        den = np.linalg.norm(v)*np.linalg.norm(u)
    csim = num/den
    return csim


def palabras_cercanas_sol(p: str, model, n=1, distances=False, lejanas=False) -> list:
    """
    Obtiene las n palabras más cercanas a la palabra p segun el modelo model
    ___________________________________
    Entrada:
    p:        [str] palabra a comparar
    model:    [gensim.Word2Vec] Modelo bajo el cual se compararan las palabras
    n:        [int] número de palabras más cercanas a retornar
    distances: [bool] Para mostrar distancias respecto a 'p' seleccionar True
    ___________________________________
    Salida:
    out:      [list(str)] Lista de n palabras más cercanas a p.
    """

    vocab = model.wv.index_to_key
    if p in vocab:
        vocab.remove(p)

    # Genera una lista de cercanas por cada forma de medir distancia
    cercanas = []
    cercanas_tmp = []
    for typ in range(4):
        vocab_dist = [cos_sim_sol(p, pi, model=model, typ=typ) for pi in vocab]
        # Organiza de mayor a menor
        if not lejanas:
            tmp = sorted(zip(vocab_dist, vocab), reverse=True)
        # Organiza de menor a mayor en caso de buscar lejanas
        else:
            tmp = sorted(zip(vocab_dist, vocab), reverse=False)

        tmp = tmp[:n]
        tmp_vocab = [it[1] for it in tmp]
        cercanas_tmp.append(tmp)
        cercanas.append(tmp_vocab)

    # Revisa que sean todas iguales
    chk = [str(l1) == str(l2) for l1, l2 in itertools.product(cercanas, cercanas)]

    if any([not it for it in chk]):
        print('ERROR: Se encontraron diferentes cercanos')
        for it in chk:
            print(it)
        return chk

    # Retorna alguna
    out = cercanas_tmp[0]
    if not distances:
        out = [it[1] for it in out[:n]]

    return out


def mas_cercanas_en_vocabulario_sol(model) -> list:
    """
    Obtiene las 2 palabras más cercanas entre sí en todo el vocabulario.
    ___________________________________
    Entrada:
    model:    [gensim.Word2Vec] Modelo bajo el cual se compararan las palabras
    ___________________________________
    Salida:
    cercanas, dist: [2-Tuple], [float] Par de palabras mas cercanas y su distancia correspondiente
    """
        
    vocab = model.wv.index_to_key 
    cercanas = []
    dist = -1
    
    for p in vocab:
        pc = palabras_cercanas_sol(p, model, n=1)[0]
        if cos_sim_sol(p, pc, model) > dist:
            cercanas = [p, pc]
            dist = cos_sim_sol(p, pc, model)
            
    return cercanas, dist




