"""S2TP2 - Matrices Positivas Definidas.ipynb - SOLUCIONES

# **Ejercicio N°2:**  Matrices Positivas Definidas

***Matemáticas para Machine Learning.***

Semana 2 - Tarea 2
"""


# LIBRERIAS ================================================
# Básicas
import numpy as np


# 4.2) ================================================
def obtener_forma_simetrica_sol(A):
    """
    Obtiene la matriz simétrica cuya forma cuadrática es igual a la de A
    ___________________________________
    Entrada:
    A:   [2D-array] Matriz a transformar
    ___________________________________
    Salida:
    Aout: [2D-array] Forma simétrica de la matriz
    """
    A = np.array(A)
    Aout = 1/2*(A + A.T)
    return Aout


# 4.3) ================================================
def positiva_definida_vprop_sol(A):
    """
    Verifica si una matriz es positva definida
    utiliza el método de valores propioes
    ___________________________________
    Entrada:
    A:   [2D-array] Matriz a revisar
    ___________________________________
    Salida:
    w : [1D-array] lista de valores propios
    out: [boolean]  Respuesta de verificación. True: Es p.d. False: no es p.d.
    """
    A = np.array(A)
    try:
        w, v = np.linalg.eig(A)
    except:
        print("ERROR en",A, sep='\n')
        
    out = all([it > 0 for it in w])
    return out, w


# 4.4) ================================================
def positiva_definida_det_sol(A):
    """
    Verifica si una matriz es positva definida
    utiliza el método de determinantes
    ___________________________________
    Entrada:
     A:   [2D-array] Matriz a revisar
    ___________________________________
    Salida:
    dets : [1D-array] lista de valores determinantes de submatrices principales
    out: [boolean]  Respuesta de verificación. True: Es p.d. False: no es p.d.
    """
    A = np.array(A)
    dets = [np.linalg.det(A[:i+1, :i+1]) for i in range(len(A))]
    out = all([it > 0 for it in dets])
    return out, dets


# 4.6) ================================================
def obtener_matrices_covarianza_sol(data):
    """ 
    Obtiene matrices de correlación para submuestras de 4 a 10 elementos.
    ___________________________________
    Entrada:
    data:   [pd.DataFrame] Base de datos a revisar
    ___________________________________
    Salida:
    corr_matrices : [list] lista de matrices de correlación de las distintas submuestras
    """
    cov_matrices = []
    # Itera sobre diferentes subtablas
    for i in range(4,10):
        for j in range(len(data) - i - 1):
            # Selecciona datos
            df = data[j:j+i]
            
            cov = df.cov()
            
            # Guarda
            cov_matrices.append(cov)
            
    return cov_matrices


# 4.7) ================================================
def resultados_covarianza_sol(fun, corr_matrices, pivot_test=False):
    """ 
    Muestra los resultados de aplicar una función para detectar si una matriz es positiva definida
    sobre las diferentes matrices de correlación encontradas
    ___________________________________
    Entrada:
    fun:   [function] función a evaluar
    corr_matrices : [list] lista de matrices de correlación de las distintas submuestras
    """
    pd_list = []
    vals0 = []
    
    for A in corr_matrices:
        out, vals = fun(A)
        # Selecciona los valores relevantes para la función de pivotes
        if pivot_test:
            vals = np.array(vals)
            vals = [vals[i,i] for i in range(vals.shape[0])]
            
        # =====================================================
        # COMPLETAR ===========================================
        # Agrega un indicador si alguno de los valores es igual a 0
        vals0.append(any([np.abs(v) < 0.00001 for v in vals]))
        # =====================================================
        pd_list.append(out)
        
    return pd_list, vals0