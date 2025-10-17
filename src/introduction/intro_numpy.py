# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: numpy.py
# Descripción: Ejercicios básicos de manejo de numpy
# ============================================================
import numpy as np
#########################################################################
# NOTE: Revisa la API de Numpy en https://numpy.org/doc/1.21/reference/ #
#########################################################################

# Ejercicio 1
# 
# TODO: Crea una función "ten_zeros_array" que devuelva un arreglo de "n" ceros.
# NOTE: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
def ten_zeros_array(n: int) -> np.ndarray:
    """
    Genera un arreglo de n ceros.

    Parámetros:
    - n: int
        Número de elementos en el arreglo.

    Retorna:
    - np.ndarray
        Arreglo de ceros de longitud n.
    """
    arg = np.zeros(n)
    return arg

# Ejercicio 2
# 
# TODO: Crea una función "floats_array" que devuelva un arreglo de números continuos 
#       (float) de "start" a "stop".
# NOTE: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
def floats_array(start: int, stop: int) -> np.ndarray:
    """
    Genera un arreglo de números flotantes continuos desde start hasta stop (sin incluir).

    Parámetros:
    - start: int
        Valor inicial del arreglo.
    - stop: int
        Valor final (no incluido) del arreglo.

    Retorna:
    - np.ndarray
        Arreglo de números flotantes.
    """
    arg = np.arange(start, stop, dtype=float)
    return arg

# Ejercicio 3 
# TODO: Crea una funcion "invert_array" que devuelva el arreglo de entrada invertido.
#       El parámetro de entrada debe ser un array de numpy.
# NOTE: https://numpy.org/doc/stable/reference/generated/numpy.flip.html
def invert_array(arg: np.ndarray) -> np.ndarray:
    """
    Invierte el orden de los elementos de un arreglo.

    Parámetros:
    - arg: np.ndarray
        Arreglo de entrada.

    Retorna:
    - np.ndarray
        Arreglo invertido.
    """
    inverted = np.flip(arg)
    return inverted

# Ejercicio 4
# TODO: Crea una funcion "square_matrix" que devuelva una matriz cuadrada de orden n 
#       con valores (int) desde "start"(int) a "stop"(int).
# NOTE: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html
def square_matrix(n: int, start: int, stop: int) -> np.ndarray:
    """
    Genera una matriz cuadrada de orden n con valores enteros desde start hasta stop.

    Parámetros:
    - n: int
        Orden de la matriz cuadrada.
    - start: int
        Valor inicial.
    - stop: int
        Valor final (no incluido).

    Retorna:
    - np.ndarray
        Matriz cuadrada con valores enteros.
    """
    matrix = np.arange(start,stop). reshape(n, n)
    return matrix

# Ejercicio 5
# TODO: Genera una función "find_upper_five" que devuelva una matriz de los índices de elementos mayores a 5 de una matriz.
#       El parámetro de entrada debe ser una matriz de numpy.
#       El formato de salida debería ser así para la matriz 
#       [0,1,2]           
#       [3,4,5]    -->    [[2 1], [0 2], [1 2], [2, 2]]   
#       [6,7,8]           
# NOTE: https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
def find_upper_five(matrix: np.ndarray) -> np.ndarray:
    """
    Encuentra los índices de los elementos mayores a 5 en una matriz.

    Parámetros:
    - matrix: np.ndarray
        Matriz de entrada.

    Retorna:
    - np.ndarray
        Arreglo de índices donde los valores son mayores a 5.
    """
    indices = np.argwhere(matrix > 5)
    return indices

# Ejercicio 6
# TODO: Genera una función "get_statistics" que calcule el promedio, la media y la desviación estándar de un arreglo numpy.
#       El parámetro de entrada debe ser un arreglo de numpy y la salida será un tuple con las tres estadísticas
#       en el orden: promedio, mediana, desviación estándar
# NOTE: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
#       https://numpy.org/doc/stable/reference/generated/numpy.median.html
#       https://numpy.org/doc/stable/reference/generated/numpy.std.html
def get_statistics(arg: np.ndarray) -> tuple[float, float, float]:
    """
    Calcula el promedio, la mediana y la desviación estándar de un arreglo.

    Parámetros:
    - arg: np.ndarray
        Arreglo de entrada.

    Retorna:
    - tuple: (mean, median, std)
        Promedio, mediana y desviación estándar como flotantes.
    """
    mean = np.mean(arg)
    median = np.median(arg)
    standard = np.std(arg)
    return (mean, median, standard)

# Ejercicio 7
# TODO: Crea una función "identity_matrix" que devuelva una matriz identidad de orden n.
#       El parámetro de entrada debe ser "n" que es el orden de la matriz en formato (int).
# NOTE: https://numpy.org/doc/stable/reference/generated/numpy.identity.html
def identity_matrix(n: int) -> np.ndarray:
    """
    Genera una matriz identidad de orden n.

    Parámetros:
    - n: int
        Orden de la matriz identidad.

    Retorna:
    - np.ndarray
        Matriz identidad de tamaño n x n.
    """
    matrix = np.identity(n)
    return matrix

# Ejercicio 8
# TODO: Crea una función "multiply_matrices" que devuelva el resultado de la multiplicación de 
#       dos matrices de numpy "a" y "b" que se dan a la entrada.
# NOTE: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
def multiply_matrices(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    """
    Multiplica dos matrices de NumPy.

    Parámetros:
    - a: np.ndarray
        Primera matriz.
    - b: np.ndarray
        Segunda matriz.

    Retorna:
    - np.ndarray
        Resultado de la multiplicación matricial.
    """
    multiplication = np.matmul(a, b)
    return multiplication

# Ejercicio 9
# TODO: Crea una función "normalize" que normalice un arreglo usando la fórmula 
#       (x - min) / (max - min). El parámetro de entrada debe ser un arreglo de numpy y la salida 
#       debe ser el arreglo normalizado.
# NOTE: https://numpy.org/doc/stable//reference/generated/numpy.min.html
#       https://numpy.org/doc/stable/reference/generated/numpy.max.html
def normalize(arg:np.ndarray) -> np.ndarray:
    """
    Normaliza un arreglo usando la fórmula (x - min) / (max - min).

    Parámetros:
    - arg: np.ndarray
        Arreglo de entrada.

    Retorna:
    - np.ndarray
        Arreglo normalizado.
    """
    normalized = (arg - np.min(arg)) / (np.max(arg) - np.min(arg))
    return normalized

# Ejercicio 10
# TODO: Crea una función "count_in_range" que cuente cuántos elementos de un array de numpy están entre
#       un rango de "a" y "b". Los parámetros de entrada deben ser arreglos de numpy y la salida debe ser
#       el total de elementos que se contaron en formato (int).
def count_in_range(arr: np.ndarray, a: float, b: float) -> int:
    """
    Cuenta cuántos elementos de un arreglo están dentro del rango [a, b].

    Parámetros:
    - arr: np.ndarray
        Arreglo de entrada.
    - a: float
        Límite inferior del rango.
    - b: float
        Límite superior del rango.

    Retorna:
    - int
        Número de elementos dentro del rango.
    """
    count = np.sum((arr >= a) & (arr <= b))
    return count