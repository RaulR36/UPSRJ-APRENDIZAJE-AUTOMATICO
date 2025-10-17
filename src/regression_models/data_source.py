# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: data_source.py
# Descripción: Definición de clase DataSource
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataSource:
    def __init__(self, url: str, churn: bool=False):
        self.url = url
        self.data = self.fetch_url()
        self.relevant_features = self.set_relevant_features(churn=churn)
        
    def fetch_url(self) -> pd.DataFrame:
        """
        Extrae datos desde la URL proporcionada utilizando pandas y los devuelve como un DataFrame.
        También imprime un resumen estadístico de las columnas numéricas.

        Returns:
            pd.DataFrame o None: Los datos extraídos, o None si ocurre un error.
        """
        data = None
        try:
            # NOTE: https://pandas.pydata.org/docs/dev/reference/api/pandas.read_csv.html
            #       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
            data = pd.read_csv(self.url)
            data.describe()
            print(f"Información extraída del url {self.url}")
        except Exception as e:
            print(f"Error: no se pudo extraer la información del url {self.url}: {e}")
        return data

    def set_relevant_features(self, churn: bool=False) -> pd.DataFrame:
        """
        Selecciona características relevantes del conjunto de datos para un modelo de regresión lineal.
        Las características incluyen tamaño del motor, número de cilindros, consumo de combustible y emisiones de CO₂.

        Returns:
            pd.DataFrame o None: Un DataFrame con las características seleccionadas, o None si falla la extracción.
        """
        # Consider the real statement. If you are making a linear regression model for estimating
        # C02 consumption in different vehicles, some relevant features could be:
        #          - The engine size
        #          - The number of cylinders
        #          - The combined fuel consumption
        #          - The CO2 emissions  
        if churn:
            rf_cols = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']
        else:
            rf_cols = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']
        rf_data = None
        try:
            # NOTE: https://www.geeksforgeeks.org/python/different-ways-to-create-pandas-dataframe/#creating-a-dataframe-from-another-dataframe
            rf_data = self.data[rf_cols]
            if churn: 
                self.data['churn'] = self.data['churn'].astype('int')
            print(f"Características relevantes seleccionadas: {rf_cols}")
        except Exception as e:
            rf_data = None
            print(f"Error: no se pudo extraer las características relevantes: {e}")
        return rf_data 
    
    def set_histogram(self, out: str) -> None:
        """
        Genera histogramas para las características relevantes seleccionadas y guarda la imagen en el archivo indicado.

        Args:
            out (str): Ruta donde se guardará la imagen del histograma.

        Returns:
            None
        """
        rf_cols = ['CO2EMISSIONS', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG']
        try:
            # NOTE: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
            #       https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
            viz = self.relevant_features[rf_cols]
            viz.hist()
            plt.savefig(out)
            plt.close()
            print(f"Se creó gráfico de histograma en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico de histograma en {out}: {e}")
        
    def plot_relationship(self, x:np.ndarray, y:np.ndarray, x_label: str, y_label: str, out: str) -> None:
        """
        Crea un gráfico de dispersión entre dos características para visualizar su relación lineal.
        Guarda la imagen en el archivo indicado.

        Args:
            x (array-like): Datos para el eje X.
            y (array-like): Datos para el eje Y.
            x_label (str): Etiqueta del eje X.
            y_label (str): Etiqueta del eje Y.
            out (str): Ruta donde se guardará la imagen del gráfico.

        Returns:
            None
        """
        try:
            # NOTE: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html
            #       https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
            plt.scatter(x, y)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(out)
            plt.close()
            print(f"Se creó gráfico de relación lineal en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico de relación lineal en {out}: {e}")
            
    def get_relevant_feature(self, feature) -> np.ndarray:
        """
        Extrae los datos de una característica específica como arreglo NumPy desde el DataFrame de características relevantes.

        Args:
            feature (str): Nombre de la característica a extraer.

        Returns:
            np.ndarray o None: Los datos extraídos, o None si ocurre un error.
        """
        data = None
        try:
            # NOTE: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html
            data = self.relevant_features[feature].to_numpy()
            print(f"Información de '{feature}' extraída.")
        except Exception as e:
            data = None
            print(f"Error: no se pudo extraer la información de {feature}: {e}")
        return data
    
    def correlate_relevant_features(self) -> None:
        try:
            # Step 1: Absolute correlation with CO2EMISSIONS
            corr_matrix = self.relevant_features.corr()
            target_corr = corr_matrix["CO2EMISSIONS"].abs().sort_values(ascending=False)
            # Step 2: Umbral for filtering redundant variables
            redundancy_threshold = 0.95
            # Step 3: Smart filter
            selected_features = []
            excluded_features = set()
            for feature in target_corr.index:
                if feature in excluded_features or feature == "CO2EMISSIONS":
                    continue
                selected_features.append(feature)
                # Exclude highly correlated variables
                for other_feature in corr_matrix.columns:
                    if other_feature != feature and corr_matrix.loc[feature, other_feature] > redundancy_threshold:
                        excluded_features.add(other_feature)

            self.correlated_features = self.relevant_features[selected_features]
            print(f"Características correlacionadas seleccionadas: {list(self.correlated_features.columns)}")
        except Exception as e:
            self.correlated_features = None
            print(f"Error: no se pudo correlacionar las características: {e}")
                    
    def plot_correlation(self, out: str) -> None:
        try:
            axes = pd.plotting.scatter_matrix(self.correlated_features, alpha=0.2)
            # need to rotate axis labels so we can read them
            for ax in axes.flatten():
                ax.xaxis.label.set_rotation(90)
                ax.yaxis.label.set_rotation(0)
                ax.yaxis.label.set_ha('right')

            plt.tight_layout()
            plt.gcf().subplots_adjust(wspace=0, hspace=0)
            plt.savefig(out)
            plt.close()
            print(f"Se creó gráfico de correlación en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico de correlación en {out}: {e}")
        
    def get_correlation_columns(self, cols: list[int]) -> np.ndarray:
        data = None
        try:
            # NOTE: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html
            data = self.correlated_features.iloc[:, cols].to_numpy()
            print(f"Columna{'s ' if len(cols) > 1 else ' '}{cols} extraída{'s.' if len(cols) > 1 else '.'}")
        except Exception as e:
            data = None
            print(f"Error: no se pudo extraer la información de columna{'s ' if len(cols) > 1 else ' '}{cols}: {e}")
        return data