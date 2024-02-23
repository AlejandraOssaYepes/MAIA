"""S2TP3_PCA.ipynb - SOLUCIONES

# **Ejercicio N°3:**  PCA

***Matemáticas para Machine Learning.***

Semana 2 - Tarea 3
"""


# LIBRERIAS ================================================
# Básicas
import numpy as np
import pandas as pd


def proyeccion_sol(X, V):
    X, V = np.array(X), np.array(V)
    Xp = np.dot(V.T, X.T).T
    return Xp


# 3.2) ================================================
def PCA_sol(X, ncomp):
    # Step-1
    X_meaned = X - np.mean(X, axis=0)

    # Step-2
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Step-5
    eigenvector_subset = sorted_eigenvectors[:, 0:ncomp]

    # Step-6
    X_PCA = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    X_PCA = proyeccion_sol(X_meaned, eigenvector_subset)

    return X_PCA


# 3.2) ================================================
def implementar_PCA_sol(data, n_comp):
    X = data.iloc[:, :4]
    Y = data.iloc[:, 4]
    X_PCA = PCA_sol(X, n_comp)
    columns = [f"x{i+1}" for i in range(n_comp)]
    df0 = pd.DataFrame(X_PCA, columns=columns)
    df = pd.concat([df0, pd.DataFrame(Y)], axis=1)
    return df

