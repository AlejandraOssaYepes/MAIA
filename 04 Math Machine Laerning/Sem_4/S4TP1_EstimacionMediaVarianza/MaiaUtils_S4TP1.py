
# Librerias
import numpy as np
import sys


class MaiaUtils:

    # Constructor
    def __init__(self):
        self.__mu = 1729
        self.__sigma = 123
        self.__var = self.__sigma**2

    def generar_datos(self, n=10):
        x = np.random.normal(self.__mu, self.__sigma, n)
        return x

    def verificar_estimaciones(self, mu, var, sigma, tol=1.0):
        """
        Función para verificar que las estadisticas estimadas son correctas.

        """
        tol_mu = 10*tol
        tol_var = 2000*tol
        tol_sigma = 45*tol
        # Media
        print(mu)
        print(self.__mu - tol_mu)
        print(self.__mu + tol_mu)
        print("Estimación de la media +++NOTA DE ALEJANDRA")
        if (self.__mu - tol_mu < mu) and (mu < self.__mu + tol_mu):
            print("Muy bien! La estimación de la media parece ser correcta\n")
        else:
            assert False, "Es muy probable que la estimación de la media sea incorrecta, revise su implementación!\n"

        # Varianza
        print("Estimación de la varianza:")
        if (self.__var - tol_var < var) and (var < self.__var + tol_var):
            print("Muy bien! La estimación de la varianza parece ser correcta\n")
        else:
            assert False, ("Es muy probable que la estimación de la varianza sea incorrecta, "
                           "revise su implementación!\n")

        # Desviacion
        print("Verificación de la estimación de la desviación estándar:")
        if (self.__sigma - tol_sigma < sigma) and (sigma < self.__sigma + tol_sigma):
            print("Muy bien! La estimación de la la desviación estándar parece ser correcta\n")
        else:
            assert False, ("Es muy probable que la estimación de la desviación estándar sea incorrecta, "
                           "revise su implementación!\n")

    def verificar_estimacion_repetida(self, media, varianza, n_datos, n_rep):

        if len(media) != n_rep or len(varianza) != n_rep:
            assert False, "La función no está retornando vectores de media y varianza acorde al número de repeticiones!"

        # Verificación de los valores estimados
        if n_datos < 100 or n_rep < 1000:
            print("EL número de datos y/o repeticiones es muy bajo para evaluar las estimaciones")
        else:
            print("Verificación estimaciones repetidas:\n")
            self.verificar_estimaciones(media.mean(), varianza.mean(), np.sqrt(varianza.mean()), tol=0.2)

    @staticmethod
    def manipulate_ipython(ipython):
        """
        Cambia el traceback para mostrar únicamente los errores obtenidos por Asserts
        En caso de que se encuentre en estra configuración cambia a la configuración inicial
        ___________________________________
        Entrada:
        ipython:          [ipykernel] Kernel utilizado en la sesión actual
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



