"""S2TP2 - Matrices Positivas Definidas.ipynb - PRUEBAS

# **Ejercicio N°2:**  Matrices Positivas Definidas

***Matemáticas para Machine Learning.***

Semana 2 - Tarea 2
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np
from sympy import Matrix

# Soluciones
from soluciones import obtener_forma_simetrica_sol
from soluciones import positiva_definida_vprop_sol
from soluciones import positiva_definida_det_sol

# Otras
from datos import generar_matrices_balanceadas
from mtr import manipulate_ipython



# 3.3) ================================================
def test_positiva_definida_vprop(positiva_definida_vprop):
    """ Prueba """

    dim = [2, 3, 5]  # Dimensiones
    n = 100          # ELementos por dimensión
    rnd = 1          # Aleatoriedad
    nrnd = 3         # Aproximar

    Alist = generar_matrices_balanceadas(dim=dim, n=n, rnd=rnd, as_int=True)
    msg = 'ES positiva definida'

    for A in Alist:
        out0, w0 = positiva_definida_vprop_sol(A)
        out, w = positiva_definida_vprop(A)

        w.sort()
        w0.sort()
        
        w, w0 = [round(wi, nrnd) for wi in w], [round(wi, nrnd) for wi in w0]

        if out != out0 or str(w) != str(w0):

            out = msg if out else 'NO ' + msg
            out0 = msg if out0 else 'NO ' + msg

            print(f"\nla respuesta no se identifico correctamente para la matriz: \n\n{A}\n")
            print(f"Resultado encontrado: {out}")
            print(f"Resultado esperado  : {out0}\n")
            print(f"Valores propios encontrados: \n{w}")
            print(f"Valores propios esperados: \n{w0}")

            assert False


# 3.4) ================================================
def test_positiva_definida_det(positiva_definida_det):
    """ Prueba """
    dim = [2, 3, 5]
    n = 100
    rnd = 1
    nrnd = 3
    Alist = generar_matrices_balanceadas(dim=dim, n=n, rnd=rnd, as_int=True)
    msg = 'ES positiva definida'

    for A in Alist:
        out0, w0 = positiva_definida_det_sol(A)
        out, w = positiva_definida_det(A)

        # Redondea
        w0, w = [round(wi, nrnd) for wi in w0], [round(wi, nrnd) for wi in w]
        # Verifica igualdad
        diff_dets = any([wi not in w for wi in w0]) or any([wi not in w0 for wi in w])

        if out != out0 or diff_dets:

            out = msg if out else 'NO ' + msg
            out0 = msg if out0 else 'NO ' + msg

            print(f"\nLa respuesta no se identifico correctamente para la matriz \n\n{A}")
            print(f"\nResultado encontrado: {out}")
            print(f"Resultado esperado  : {out0}")
            print(f"\nValores de determinantes encontrados: \n{w}")
            print(f"Valores de determinantes esperados: \n{w0}")

            assert False


# 3.4) ================================================
def test_positiva_definida_pivots(positiva_definida_pivots):
    """ Prueba """
    dim = [2, 3, 5]
    n = 100
    rnd = 1
    Alist = generar_matrices_balanceadas(dim=dim, n=n, rnd=rnd, as_int=True)
    msg = 'ES positiva definida'

    for A in Alist:
        out, Ar = positiva_definida_pivots(A)
        
        # NO FUNCIONA???
        Acanon = Matrix(A).rref()[0]
        out0 = not(any([np.array(Acanon)[i, i] <= 0 for i in range(Acanon.shape[0])]))
        
        out0, _ = positiva_definida_det_sol(A)
        
        diff_reduced = str(Matrix(A).rref()) != str(Matrix(Ar).rref())

        if out != out0 or diff_reduced:

            out = msg if out else 'NO ' + msg
            out0 = msg if out0 else 'NO ' + msg

            print(f"\nLa respuesta no se identifico correctamente para la matriz \n\n{A}")
            print(f"\nResultado encontrado: {out}")
            print(f"Resultado esperado  : {out0}")
            #print(f"\nLa matriz A reducida en forma canónica: \n{np.array(Matrix(A).rref()[0])}")
            #print(f"La matriz Ar reducida en forma canónica: \n{np.array(Matrix(Ar).rref()[0])}")

            assert False