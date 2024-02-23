"""S1TCP_Word2Vec.ipynb - PRUEBAS
# **Taller N°1:** Word2Vec

***Matemáticas para Machine Learning.***

Semana 1 - Taller Práctico
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np                # Matemáticas
import itertools                  # Manipulación iterables
import pandas as pd
import random

# Procesamiento de datos
from sklearn.decomposition import TruncatedSVD, PCA

# Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

# Visualización
import sys
from IPython.display import display, HTML
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")

# Soluciones
from soluciones import cos_sim_sol
from soluciones import palabras_cercanas_sol
from soluciones import mas_cercanas_en_vocabulario_sol

# Adicionales
from mtr import manipulate_ipython
from visualizacion import reduccion_dimensionalidad


# 1) ================================================
def test_cos_sim(cos_sim, modelos):
    """
    Prueba el correcto funcionamiento de cos_sim
    """
    n_rnd = 5  # Digitos a redondear en vectores
    k = 100  # Número de palabras a usar

    # Itera sobre cada modelo
    for name, model in modelos.items():
        vocab = model.wv.index_to_key

        # Submuestra para encontrar similaridades
        if k is not None:
            words = random.choices(vocab, k=k)
        else:
            words = vocab

        # Iteración sobre pares de vectores
        combinations = list(itertools.product(words, words))

        print(f'\nModelo: {name}\n')
        for i in tqdm(range(len(combinations))):
            p1, p2 = combinations[i]
            sim_self = round(float(cos_sim(p1, p2, model)), n_rnd)
            sim_sol0 = round(float(cos_sim_sol(p1, p2, model, typ=0)), n_rnd)
            sim_sol1 = round(float(cos_sim_sol(p1, p2, model, typ=1)), n_rnd)
            sim_sol2 = round(float(cos_sim_sol(p1, p2, model, typ=2)), n_rnd)
            sim_sol3 = round(float(cos_sim_sol(p1, p2, model, typ=3)), n_rnd)
            sim_model = round(float(model.wv.similarity(p1, p2)), n_rnd)

            solutions = [sim_sol0, sim_sol1, sim_sol2, sim_sol3]
            # Revisa si el error es mayor al 0.1%
            if sim_self == 0: sim_self = 10**(-15)
            check2 = [np.abs((sim_self - sim_sol) / sim_self) > 0.001 for sim_sol in solutions]

            # Se muestra error
            if all(check2):
                print(f'Error entre: {p1}/{p2}')
                print(f'cos_sim    = {sim_self}')
                print('\nValores esperados (posibles errores de aproximación):\n')

                print(f'cos_sim0 = {sim_sol0}')
                print(f'cos_sim1 = {sim_sol1}')
                print(f'cos_sim2 = {sim_sol2}')
                print(f'cos_sim3 = {sim_sol2}')

                print(f'W2V sim = {sim_model}')

                assert False


# 2) ================================================
def test_palabras_cercanas(palabras_cercanas, modelos, lejanas=False):
    """
    Prueba función palabras_cercanas
    ___________________________________
    Entrada:
    palabras_cercanas: [function] función a probar
    modelos: [dict] Directorio de modelos tipo gensim.Word2Vec con sus respectivos nombres
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """
    nlist = [1, 3, 5]
    n_rnd = 5  # Digitos a redondear en vectores
    k = 100  # Número de palabras a usar

    # Itera sobre cada modelo
    for name, model in modelos.items():
        vocab = model.wv.index_to_key

        # Submuestra para encontrar similaridades
        if k is not None:
            words = random.choices(vocab, k=k)
        else:
            words = vocab

        print(f'\nModelo: {name}\n')

        for i in tqdm(range(len(words))):
            p = words[i]
            for n in nlist:
                cercanas = palabras_cercanas(p, model, n=n)
                cercanas_sol = palabras_cercanas_sol(p, model, n=n, lejanas=lejanas)
                    
                if str(sorted(cercanas)) != str(sorted(cercanas_sol)):
                    
                    cercanas_sol = palabras_cercanas_sol(p, model, n=n, distances=True, lejanas=lejanas)
                    
                    msg = 'lejanas' if lejanas else 'cercanas'
                    
                    print(f'Utilizando la palabra "{p}" y {n} palabras {msg}\n')
                    print(f'Se obtuvieron las siguientes palabras {msg}: \n{cercanas}')
                    print(f'Se esperaban las siguientes palabras {msg}: \n{cercanas_sol}')


                    assert False




def test_palabras_lejanas(palabras_cercanas, modelos):
    """
    Prueba función palabras_cercanas
    ___________________________________
    Entrada:
    palabras_cercanas: [function] función a probar
    modelos: [1D-array] lista de modelos tipo gensim.Word2Vec
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """

    test_palabras_cercanas(palabras_cercanas, modelos, lejanas=True)
    
    
def test_mas_cercanas_en_vocabulario(mas_cercanas_en_vocabulario, modelos):

    # Itera sobre cada modelo
    for name, model in modelos.items():
        c, d = mas_cercanas_en_vocabulario(model)
        c0, d0 = mas_cercanas_en_vocabulario_sol(model)
        
        if str(sorted(c)) != str(sorted(c0)):
            print(f"En el modelo {name}\n")
            print(f"Se encontraron las palabras {c} con distancia {d}\n")
            print(f"Deberían ser las palabras {c0} con distancia {d0}")
            
            assert False
            