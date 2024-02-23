"""S1TP1 - Distancias y algoritmo KNN.ipynb - PRUEBAS

# **Ejercicio N°1:** Distancias y algoritmo KNN

***Matemáticas para Machine Learning.***

Semana 1 - Tarea 1
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np      
import pandas as pd
from itertools import product
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# Soluciones
from soluciones import norm_solucion
from soluciones import k_mas_cercanos_solucion
from soluciones import k_mas_cercanos_indice_solucion
from soluciones import encontrar_etiqueta_solucion
from soluciones import completar_tabla_solucion

# Otras
from datos import datos
from datos import GEN, EDA
from datos import generar_datos_basicos
from visualizar import visualizar_cercanos


PLINE = '__________________________________'
datos_conocidos, datos_desconocidos, datos_comparacion = datos()


def pretty_print(df):
    """ Muestra tabla con IPython.display """
    if isinstance(df, dict): df = pd.DataFrame(df)
    return display(HTML(df.to_html()))


# 1) ================================================
def test_norm(norm):
    """
    Prueba función norm
    ___________________________________
    Entrada:
    norm: [function] función a probar
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """
    n_vec = 10                # Vectores por dim
    nlog = 3                  # maximo número = 10^nlog
    dim_list = [2,3,8]        # Lista de dimensiones a revisar
    n_rnd = 5                 # # decimas aproxmar


    for dim in dim_list:      # Test para cada dimensión
        # Genera datos aleatorios
        V = np.around(np.random.rand(n_vec, dim), nlog)*10**(nlog)
        for v in V:
            # Halla normas
            sol = round(norm(v), n_rnd)
            sol0 = round(norm_solucion(v, typ=0), n_rnd)
            sol1 = round(norm_solucion(v, typ=1), n_rnd)

            # Evalua
            if sol != sol0 and sol != sol1:
                print('Para el vector:')
                print(v)
                print('\nResultado obtenido: ')
                print(f'{sol}')
                print('\nResultado esperado: ')
                print(f'{sol0}\n\n')

                # Error
                assert False, f'\n{PLINE}'

                
# 2) ================================================
def comparar_cercanos(c, c0, x, V, n_rnd=2):
    """ Compara las listas de cercanos c y c0 respecto al vector x """
    x = np.array(x)
    c = sorted(c, key=lambda k: np.linalg.norm(np.array(k)-x))
    c_dist = [round(np.linalg.norm(np.array(k)-x), n_rnd) for k in c]
    c0 = sorted(c0, key=lambda k: np.linalg.norm(np.array(k)-x))
    c0_dist = [round(np.linalg.norm(np.array(k)-x), n_rnd) for k in c0]

    # Caso Error
    if str(c_dist) != str(c0_dist):
        df = {
          'Vectores encontrados': c,
          'Vectores cercanos': c0,
          'Distancias encontradas': c_dist,
          'Distancias cercanos': c0_dist
        }
        print(f'Para k={len(c)} y x={x} se obtuvieron los siguientes resultados:\n\n')
        pretty_print(df)
        # print('\n\nSe guardaron los datos generados en V, x')

        # Dibuja visualización
        print(f'\n{PLINE}\n')
        print('Cercanos obtenidos:')
        _ = visualizar_cercanos(x, V, C=c)
        plt.show()
        print('Cercanos esperados:')
        _ = visualizar_cercanos(x, V, C=c0)
        plt.show()

        assert False, f'\n{PLINE}'

    # Continua operación
    else:
        pass
    
    
def test_k_mas_cercanos(k_mas_cercanos):
    """
    Prueba función k_mas_cercanos, interrumpe si hay error
    """
    print('test_k_mas_cercanos')

    n = 1000                  # Vectores por dim
    nlog = 3                  # Vectores en [0, ,10^nlog]]
    dim_list = [2, 3, 8]      # Lista de dimensiones a revisar
    n_rnd = 2                 # Decimas aproxmar
    k_list = [10, 5, 10]      # Número de vecinos

    # Prueba para distintas dimensiones
    for dim, k in product(dim_list, k_list):
        # dim, k = 2, 5
        # Generar datos
        V, x = generar_datos_basicos(n=n, dim=dim, nlog=nlog, seed=None)
        # Encontrar cercanos
        c = k_mas_cercanos(x=x, V=V, k=k, norm=norm_solucion)
        c0 = k_mas_cercanos_solucion(x=x, V=V, k=k, norm=norm_solucion)
        # Comparar y Evaluar
        comparar_cercanos(c=c, c0=c0, x=x, V=V, n_rnd=n_rnd)

        

# 3a) ================================================
def comparar_cercanos_idx(c, c0, x):
    """ Compara las listas de cercanos c y c0 respecto al vector x """
    c = sorted(c)
    c0 = sorted(c0)

    # Caso Error
    if str(c) != str(c0):
        df = {
            'Indices encontrados': c,
            'Indices cercanos': c0,
        }
        print(f'Para k={len(c)} y x={x} se obtuvieron los siguientes resultados:\n\n')
        pretty_print(df)
        assert False, f'\n{PLINE}'

    # Continua operación
    else:
        pass


def test_k_mas_cercanos_indice(k_mas_cercanos_indice):
    """
    Prueba función k_mas_cercanos_indice
    ___________________________________
    Entrada:
    k_mas_cercanos_indice: [function] función a probar
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """
    n = 1000                  # Vectores por dim
    nlog = 3                  # Vectores en [0, ,10^nlog]]
    dim_list = [2, 3, 8]        # Lista de dimensiones a revisar
    n_rnd = 2                 # Decimas aproxmar
    k_list = [1, 5, 10]         # Número de vecinos

    global V, x
    # Prueba para distintas dimensiones
    for dim, k in product(dim_list, k_list):
        # Generar datos
        V, x = generar_datos_basicos(n=n, dim=dim, nlog=nlog, seed=None)
        # Encontrar cercanos
        cercanos = k_mas_cercanos_indice(x=x, V=V, k=k, norm=norm_solucion)
        cercanos0 = k_mas_cercanos_indice_solucion(x=x, V=V, k=k, norm=norm_solucion)
        # Comparar y Evaluar
        comparar_cercanos_idx(cercanos, cercanos0, x)


# 3b) ================================================
def test_encontrar_etiqueta(encontrar_etiqueta):
    """
    Prueba función encontrar_etiqueta
    ___________________________________
    Entrada:
    encontrar_etiqueta: [function] función a probar
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """
    global datos_conocidos, datos_desconocidos

    datos = datos_conocidos

    for i in tqdm(range(len(datos_desconocidos))):
        d = datos_desconocidos.iloc[i]

        # Hallar etiquetas calculadas
        # et_gen, kinfo_gen = encontrar_etiqueta(d, datos, k=5, etiqueta=GEN)
        et_eda, kinfo_eda = encontrar_etiqueta(d, datos, k=5, etiqueta=EDA)

        # Hallar etiquetas solución
        # et_gen_sol, kinfo_gen_sol = encontrar_etiqueta_solucion(d, datos, k=5, etiqueta=GEN)
        et_eda_sol, kinfo_eda_sol = encontrar_etiqueta_solucion(d, datos, k=5, etiqueta=EDA)

        # Revisa que las etiquetas de los datos sean iguales
        # for et, et_sol, kinfo, kinfo_sol, ET in zip([et_gen, et_eda], [et_gen_sol, et_eda_sol],
        #                                             [kinfo_gen, kinfo_eda], [kinfo_gen_sol, kinfo_eda_sol],
        #                                             [GEN, EDA]):
        for et, et_sol, kinfo, kinfo_sol, ET in zip([et_eda], [et_eda_sol],
                                                    [kinfo_eda], [kinfo_eda_sol],
                                                    [EDA]):
            if et != et_sol:
                print(f'Dato [{i}]')
                print(f'{d}')
                print(PLINE)
                print(f'Etiqueta: {ET}')
                print(f'Calculado: {et}')
                print(f'Real     : {et_sol}')
                print(PLINE)
                print('\nCercanos Encontrados:')
                display(kinfo)
                print('\nCercanos Reales     :')
                display(kinfo_sol)
                assert False, f'\n{PLINE}'

                
# 3c) ================================================
def test_completar_tabla(completar_tabla):
    """
    Prueba función completar_tabla
    ___________________________________
    Entrada:
    completar_tabla: [function] función a probar
    ___________________________________
    Salida:
    Interrumpe ejecición en caso de error
    """

    print('\nGenerando tabla a partir de completar_tabla')
    datos_interpolados = completar_tabla(datos_conocidos, datos_desconocidos, k=5)

    print('\nGenerando tabla de comparación')
    datos_interpolados_sol = completar_tabla_solucion(datos_conocidos, datos_desconocidos, k=5)

    for i, d in datos_interpolados.iterrows():
        if str(d) != str(datos_interpolados_sol.iloc[i]):
            print('\nDato equivocado')
            pretty_print(pd.DataFrame(d, columns=[i]))
            print(PLINE)
            print('\nDato correcto')
            pretty_print(pd.DataFrame(datos_interpolados_sol.iloc[i], columns=[i]))

            # Error
            assert False, f'\n{PLINE}'

