
from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np


# Funcion de reducción de dimensionalidad (PCA)
def reduccion_dimensionalidad(palabra, modelo, new_dim=2):
    """
    Reduce la dimensionalidad de una palabra palabra en 2D.
    ___________________________________
    Entrada:
    palabra: [1D-array] Palabra en forma de vector
    modelo:  [gensim.Word2Vec] Modelo utilizado
    new_dim: [int] nueva dimensión de la palabra
    ____________________________________
    Salida:
    palabra_red: [1D-array] Nueva representación vectorial de la palabra.
    """
    palabras = modelo.wv.index_to_key
    vectores = []
    for p in palabras:
        vectores.append(modelo.wv[p])
        
    vectores = np.array(vectores)
    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=new_dim)
    pca.fit(vectores)
    
    palabra_reducida = pca.transform(palabra.reshape(1, -1))[0]

    # Devuelve data en 2D
    return palabra_reducida



