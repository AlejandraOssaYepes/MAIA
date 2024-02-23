"""S2TP3_PCA.ipynb - PRUEBAS

# **Ejercicio N°3:**  PCA

***Matemáticas para Machine Learning.***

Semana 2 - Tarea 3
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
import pandas as pd
from sympy import Matrix
from IPython.display import display

# Soluciones
from soluciones import proyeccion_sol
from soluciones import PCA_sol
from soluciones import implementar_PCA_sol

# Auxiliar
from mtr import manipulate_ipython
from datos import generar_matrices_pd
from datos import generar_datos


PLINE = '__________________________________'



def listas_iguales(l1, l2, nrnd=2):
    l1 = [str([round(xi, nrnd) for xi in vi]) for vi in l1]
    l2 = [str([round(xi, nrnd) for xi in vi]) for vi in l2]
    
    out = [vi not in l2 for vi in l1] + [vi not in l1 for vi in l2]
    
    return not any(out)


def tablas_iguales(df1, df2, nrnd=2):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    df1n = df1.select_dtypes(include=numerics)
    df2n = df2.select_dtypes(include=numerics)
    
    df = pd.concat([df1n, df2n])
    df = df.reset_index(drop=True)
    df_gpby = df.groupby(list(df.columns))
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
    df.reindex(idx)
    
    different = len(df) > 0
    
    return not different



# 3.2) ================================================
def test_proyeccion(proyeccion):
    Alist = generar_matrices_pd(dim=[2, 3, 5], n=5, rnd=1, as_int=True)
    for A in Alist:
        A = np.array(A)
        _, V = np.linalg.eig(A)
        
        for n in [10, 50, 100]:
            X = generar_datos(n=10, dim=A.shape[0], nlog=3, seed=None)
            
            Xp = proyeccion(X, V)
            Xp0 = proyeccion_sol(X, V)
            
            if not listas_iguales(Xp, Xp0):
                print(f"Para los siguientes datos:\n\n{X}\n\n")
                print(f"Se obtuvo: \n\n{Xp}\n\n")
                print(f"Se esperaba: \n\n{Xp0}\n\n")
                print(f"Usando la base: \n\n{V}\n\n")
                
            
                assert False
                
            
def test_PCA(PCA):
    for dim in [2,3,5]:
        
        ncomps = [1,2,3]
        ncomps = [it for it in ncomps if it < dim]
        
        for ncomp in ncomps: 
            for n in [10,50,100]:
                X = generar_datos(n=n, dim=dim, nlog=3, seed=None)
                X_PCA = PCA(X, ncomp=ncomp)
                X_PCA0 = PCA_sol(X, ncomp=ncomp)
                
                if not listas_iguales(X_PCA, X_PCA0):
                    print(f"Para los siguientes datos:\n\n{X}\n\n")
                    print(f"Se obtuvo: \n\n{X_PCA}\n\n")
                    print(f"Se esperaba: \n\n{X_PCA0}\n\n")
                    
                    assert False
                    
            
def test_implementar_PCA(implementar_PCA, data, ncomp=2):
    df = implementar_PCA(data, ncomp)
    df0 = implementar_PCA_sol(data, ncomp)
    
    if not tablas_iguales(df, df0):
        print("Se obtuvo\n")
        display(df)
        print(PLINE)
        print("\n\nSe espera\n")
        display(df0)
        
        assert False