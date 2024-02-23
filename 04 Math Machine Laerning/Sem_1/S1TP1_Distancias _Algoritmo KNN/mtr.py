"""
***Matemáticas para Machine Learning.***
Manipulación de Traceback
"""

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
        TRACEDEFAULT = ipython.showtraceback

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
