# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: pandas.py
# Descripción: Ejercicios básicos de manejo de pandas
# ============================================================
import pandas as pd

#########################################################################
# NOTE: Revisa la API de Pandas en https://pandas.pydata.org/docs/      #
#########################################################################

# Ejercicio 1
#
# TODO: Crea una función "csv_registers" que reciba un CSV y devuelva la cantidad de registros (int).
#       así como el contenido en un DataFrame en un tuple(int, DataFrame).
# NOTE: https://pandas.pydata.org/docs/dev/reference/api/pandas.read_csv.html
def csv_registers(file: str) -> tuple[int, pd.DataFrame]:
    """
    Lee un archivo CSV y devuelve la cantidad de registros junto con su contenido.

    Parámetros:
    - file: str
        Ruta al archivo CSV.

    Retorna:
    - tuple: (n_registros, DataFrame)
        Número de registros y contenido como DataFrame.
    """
    data = pd.read_csv(file)
    registers = len(data)
    return (registers, data)

# Ejercicio 2
#
# TODO: Crea una función "json_registers" que reciba un JSON y devuelva la cantidad de registros (int).
#       así como el contenido en un DataFrame en un tuple(int, DataFrame).
# NOTE: https://pandas.pydata.org/docs/dev/reference/api/pandas.read_json.html
def json_registers(file: str) -> tuple[int, pd.DataFrame]:
    """
    Lee un archivo JSON y devuelve la cantidad de registros junto con su contenido.

    Parámetros:
    - file: str
        Ruta al archivo JSON.

    Retorna:
    - tuple: (n_registros, DataFrame)
        Número de registros y contenido como DataFrame.
    """
    data = pd.read_json(file)
    registers = len(data)
    return (registers, data)

# Ejercicio 3
#
# TODO: Crea una función "yaml_registers" que reciba un YAML y devuelva la cantidad de registros (int)
#       así como el contenido en un DataFrame en un tuple(int, DataFrame).
# NOTE: https://pyyaml.org/wiki/PyYAMLDocumentation
def yaml_registers(file: str) -> tuple[int, pd.DataFrame]:
    """
    Lee un archivo YAML y devuelve la cantidad de registros junto con su contenido.

    Parámetros:
    - file: str
        Ruta al archivo YAML.

    Retorna:
    - tuple: (n_registros, DataFrame)
        Número de registros y contenido como DataFrame.
    """
    data = pd.DataFrame(file)
    registers = len(data)
    return (registers, data)

# Ejercicio 4
# TODO: Crea una función "get_head" que devuelva un DataFrame solo con los primeros n registros de otro DataFrame.
# NOTE: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html
def get_head(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Devuelve los primeros n registros de un DataFrame.

    Parámetros:
    - df: pd.DataFrame
        DataFrame de entrada.
    - n: int
        Número de registros a extraer.

    Retorna:
    - pd.DataFrame
        Subconjunto con los primeros n registros.
    """
    df_head = df.head(n)
    return df_head

# Ejercicio 5
# TODO: Crea una función "get_above" que devuelva un Dataframe solo con los elementos que cumplan la cualidad de ser
#       mayores al valor de entrada "n" en la columna "col" del DataFrame "df" a la entrada. 
# NOTE: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#dataframe
def get_above(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """
    Filtra los registros de un DataFrame donde los valores de una columna son mayores a n.

    Parámetros:
    - df: pd.DataFrame
        DataFrame de entrada.
    - col: str
        Nombre de la columna a evaluar.
    - n: int
        Valor umbral para el filtrado.

    Retorna:
    - pd.DataFrame
        Registros que cumplen la condición col > n.
    """
    above = df[df[col] > n]
    return above

# Ejercicio 6
# TODO: Crea una función "group_and_average" que agrupe un DataFrame de entrada conforme a la columna "group" de entrada
#       y calcule el promedio de la columna "avg" de entrada. Dadas las cualidades de las columnas de 
#       entrada, debe devolver un objeto Series.
# NOTE: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
def group_and_average(df: pd.DataFrame, group: str, avg: str) -> pd.Series:
    """
    Agrupa un DataFrame por una columna y calcula el promedio de otra.

    Parámetros:
    - df: pd.DataFrame
        DataFrame de entrada.
    - group: str
        Columna por la cual agrupar.
    - avg: str
        Columna sobre la cual calcular el promedio.

    Retorna:
    - pd.Series
        Promedio por grupo.
    """
    grouped = df.groupby(group)[avg].mean()
    return grouped

# Ejercicio 7
# TODO: Crea una función "count_in_col" que cuente elementos "item" en un DataFrame de entrada 
#       de una columna "col" de entrada. La salida debe ser el conteo de elementos (int).
def count_in_col(df: pd.DataFrame, item: str, col: str) -> int:
    """
    Cuenta cuántas veces aparece un elemento en una columna de un DataFrame.

    Parámetros:
    - df: pd.DataFrame
        DataFrame de entrada.
    - item: str
        Elemento a contar.
    - col: str
        Columna donde buscar el elemento.

    Retorna:
    - int
        Número de ocurrencias del elemento.
    """
    count = (df[col] == item).sum()
    return count 

# Ejercicio 8
# TODO: Crea una función "export_data" que exporte un DataFrame de entrada a un archivo CSV "file" de entrada.
# NOTE: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
def export_data(df: pd.DataFrame, file: str) -> None:
    """
    Exporta un DataFrame a un archivo CSV.

    Parámetros:
    - df: pd.DataFrame
        DataFrame a exportar.
    - file: str
        Ruta destino del archivo CSV.

    Retorna:
    - None
    """
    df.to_csv(file, index=False)

# Ejercicio 9
# TODO: Crea una función "compare_dfs" que compare dos DataFrame de entrada y devuelva un True (bool) si son iguales
#       o bien, un False (bool) si no lo son.
def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Compara dos DataFrames y determina si son iguales.

    Parámetros:
    - df1: pd.DataFrame
        Primer DataFrame.
    - df2: pd.DataFrame
        Segundo DataFrame.

    Retorna:
    - bool
        True si son iguales, False si no lo son.
    """
    equal = df1.equals(df2)
    return equal

