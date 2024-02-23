"""
***Matemáticas para Machine Learning.***
Manipulación de Traceback
"""
""" Manipulación de Traceback """

import sys

def manipulate_ipython(ipython):
    """ 
    Cambia el traceback para mostrar únicamente los errores obtenidos por Asserts
    En caso de que se encuentre en estra configuración cambia a la configuración inicial 
    ___________________________________
    Entrada:
    ipython:    [ipykernel] Kernel utilizado en la sesión actual
    ___________________________________
    Salida:
    toggle_traceback: [function] Función para manipular el traceback

    """
    if 'TRACEDEFAULT' not in globals():
        globals()['TRACEDEFAULT'] = ipython.showtraceback

    def hide_traceback(exc_tuple=None, filename=None, tb_offset=None,
                    exception_only=False, running_compiled_code=False):
        """ Oculta el reporte predeterminado de python respecto al error encontrado. """
        etype, value, tb = sys.exc_info()
        value._cause_ = None  # suppress chained exceptions
        return ipython._showtraceback(etype, value, ipython.InteractiveTB.get_exception_only(etype, value))


    def toggle_traceback():
        """ Cambia el reporte de traceback """
        if ipython.showtraceback == TRACEDEFAULT:
            ipython.showtraceback = hide_traceback
        else:
            ipython.showtraceback = TRACEDEFAULT

    return toggle_traceback


#a= maia.generar_datos_markov()

#print(a)

def cota_markov(q,p):
    """ 
    Retorna la cota de Markov para la probabilidad de obtener
    cara más de q por ciento para una moneda con probabilidad de exito p,
    para n experimentos.
    ___________________________________
    Entrada:
    q: [float] Porcentaje de los intentos que se desea, sean cara.
    p: [float] Probabilidad de exito de la moneda .
    n: [int] Cantidad de repeticiones del experimento.
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

#maia.calificar_cota_markov(cota_markov)
#maia.calificar_cota_chernof(cota_chernof)
#
#maia.calificar_cota(0,cota_chernof)


