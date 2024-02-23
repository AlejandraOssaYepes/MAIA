"""S3TP1 - Fuentes de Incertidumbre.ipynb - PRUEBAS
# **Taller N°1:** Fuentes de Incertidumbre

***Matemáticas para Machine Learning.***

Semana 1 - Actividad 1
"""

# LIBRERIAS ================================================
# Básicas
import numpy as np                # Matemáticas
import itertools                  # Manipulación iterables
import pandas as pd
import random
from functools import partial

# Visualización
import sys
from IPython.display import display, HTML
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")

# Soluciones
from soluciones import crear_histograma_sol
from soluciones import parametros_distribucion_sol
from soluciones import obtener_todos_los_parametros_sol
from soluciones import fdp_sol
from soluciones import clasificador_distribucion_muestra_sol
from soluciones import verosimilitud_sol
from soluciones import clasificador_distribucion_muestras_sol

# Adicionales
from mtr import manipulate_ipython

# Datos
from datos import obtener_datos
from datos import datos_dividir_en_clases

DATOS = obtener_datos()

def mostrar_tabla(df, transpose=True):
        """ Mostrar los resultados de una tabla
        ___________________________________
        Entrada:
        df : [pd.DataFrane/dict] tabla a mostrar
        transpose: [bool] Transponer visualización
        """
        if isinstance(df, dict): df = pd.DataFrame(df)
        if transpose: df = df.T
        return display(HTML(df.to_html()))
    
    
# 1) ================================================
def test_crear_histograma(crear_histograma):
    """
    Prueba el correcto funcionamiento de crear_histograma
    """
    Xclase = datos_dividir_en_clases(DATOS)
    fig = None
    
    for ngrupos in [3,5,10,20]:
        # Revisa histograma para cada clase
        for i, x in Xclase.items():

            val, cont = crear_histograma(x, ngrupos=ngrupos)
            val0, cont0 = crear_histograma_sol(x, ngrupos=ngrupos)      
            
            cont = [int(c) for c in cont]
            cont0 = [int(c) for c in cont0]
            
            if str(cont) != str(cont0): 
                print(f'Usando ngrupos={ngrupos} se obtiene: \n')
                df = {'Valores Hallados':val, 'Conteos Hallados':cont,
                      'Valores Reales':val0, 'Conteos Reales':cont0}
                df = pd.DataFrame(df, index=[f'Intervalo {i}' for i in range(len(val))])
                mostrar_tabla(df)
                assert False
            
            
            

# 2) ================================================
def test_parametros_distribucion(parametros_distribucion):
    """
    Prueba el correcto funcionamiento de crear_histograma
    """
    Xclase = datos_dividir_en_clases(DATOS)
    for i, x in Xclase.items():
        mu, var = parametros_distribucion(x)
        mu0, var0 = parametros_distribucion_sol(x)
        
        if (mu != mu0) or (var != var0):
            df = {'Media Hallada':mu, 'Varianza Muestral Hallada':var,
                  'Media Real':mu0, 'Varianza Muestral Real':var0}
            df = pd.DataFrame(df,index=[f'Prueba Clase {i}'])
            mostrar_tabla(df)
            assert False

# 3) ================================================
def test_obtener_todos_los_parametros(obtener_todos_los_parametros):
    """
    Prueba el correcto funcionamiento de obtener_todos_los_parametros
    """
    mus, sigmas = obtener_todos_los_parametros(DATOS)
    mus0, sigmas0 = obtener_todos_los_parametros_sol(DATOS)
    
    if mus!=mus0:
        print(f'Error en la estimación de las medias, se obtiene: \n')
        df = {'Valores Reales':[mus0], 'Valores Hallados':[mus]}
        df = pd.DataFrame(df,index=['Comparación de medias'])
        mostrar_tabla(df)
        assert False
        
    elif sigmas!=sigmas0:
        print(f'Error en la estimación de las varianzas, se obtiene: \n')
        df = {'Valores Reales':[sigmas0], 'Valores Hallados':[sigmas]}
        df = pd.DataFrame(df,index=['Comparación de varianzas'])
        mostrar_tabla(df)
        assert False
    

# 4) ================================================
def test_fdp(fdp):
    """
    Prueba el correcto funcionamiento de fdp
    """
    datos = DATOS
    Xclase = datos_dividir_en_clases(datos)
    
    ceros = random.sample(Xclase[0],10)
    
    fdp_sol_ceros = partial(fdp_sol,datos=DATOS,dist=0)
    fdp_ceros = partial(fdp,datos=DATOS,dist=0)
    
    unos = random.sample(Xclase[1],10)
    fdp_sol_unos = partial(fdp_sol,datos=DATOS,dist=1)
    fdp_unos = partial(fdp,datos=DATOS,dist=1)
    
    ps = list(map(fdp_sol_ceros,ceros))
    ps0 = list(map(fdp_ceros,ceros))
    
    if ps!=ps0:
        print('Clase 0 \n')
        df = {'Dato':ceros,'Probabilidad Real':ps, 'Probabilidad Hallada':ps0}
        df = pd.DataFrame(df, index=[f'Prueba {i+1}' for i in range(10)])
        mostrar_tabla(df)
        assert False
    
    ps = list(map(fdp_sol_unos,unos))
    ps0 = list(map(fdp_unos,unos))
    
    if ps!=ps0:
        print('Clase 1 \n')
        df = {'Dato':ceros,'Probabilidad Real':ps, 'Probabilidad Hallada':ps0}
        df = pd.DataFrame(df, index=[f'Prueba {i+1}' for i in range(10)])
        mostrar_tabla(df)
        assert False

# 5) ================================================
def test_clasificador_distribucion_muestra(clasificador_distribucion_muestra):
    """
    Prueba el correcto funcionamiento de clasificador_distribucion_muestra
    """
    q = 5
    Xclase = datos_dividir_en_clases(DATOS)

    test_data = random.sample(Xclase[0],q)
    test_data.extend(random.sample(Xclase[1],q))

    class_sol = []
    class_trial= []
    
    for i in test_data:
        class_sol.append(clasificador_distribucion_muestra_sol(i,DATOS))
        class_trial.append(clasificador_distribucion_muestra(i,DATOS))
    
    if class_sol!=class_trial:
        print('Comparación de Clases \n')
        df = {'Dato':test_data,'Clase Real':class_sol, 'Clase Hallada':class_trial}
        df = pd.DataFrame(df, index=[f'Prueba {i+1}' for i in range(q*2)])
        mostrar_tabla(df)
        assert False

# 6) ================================================
def test_verosimilitud(verosimilitud):
    """
    Prueba el correcto funcionamiento de verosimilitud
    """
    q = 20
    Xclase = datos_dividir_en_clases(DATOS)

    test_data = random.sample(Xclase[0],q)
    test_data.extend(random.sample(Xclase[1],q))
    
    vero_sol_cero = verosimilitud_sol(test_data,DATOS)
    vero_trial_cero = verosimilitud(test_data,DATOS)
    
    vero_sol_uno = verosimilitud_sol(test_data,DATOS,dist=1)
    vero_trial_uno = verosimilitud(test_data,DATOS,dist=1)
    
    vero_sol = [vero_sol_cero,vero_sol_uno]
    vero_trial = [vero_trial_cero,vero_trial_uno]
    
    if vero_trial != vero_sol: 
        df = {'Verosimilitud Real':vero_sol, 'Verosimilitud Hallada':vero_trial}
        df = pd.DataFrame(df, index=['Clase 0', 'Clase 1'])
        mostrar_tabla(df)
        assert False


# 7) ================================================
def test_clasificador_distribucion_muestras(clasificador_distribucion_muestras):
    """
    Prueba el correcto funcionamiento de clasificador_distribucion_muestras
    """
    q = 20
    Xclase = datos_dividir_en_clases(DATOS)

    test_data_cero = random.sample(Xclase[0],q)
    test_data_uno = random.sample(Xclase[1],q)

    class_sol = [clasificador_distribucion_muestras_sol(test_data_cero,DATOS),clasificador_distribucion_muestras_sol(test_data_uno,DATOS)]
    class_trial= [clasificador_distribucion_muestras(test_data_cero,DATOS),clasificador_distribucion_muestras(test_data_uno,DATOS)]

    
    if class_sol!=class_trial:
        print('Comparación de Clasificación \n')
        df = {'Clase Real':class_sol, 'Clase Hallada':class_trial}
        df = pd.DataFrame(df, index=['Prueba 1', 'Prueba 2'])
        mostrar_tabla(df)
        assert False                   