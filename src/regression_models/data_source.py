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
        Para churn, incluye características relacionadas con la retención de clientes.

        Args:
            churn (bool): Si True, selecciona características para modelo de churn.

        Returns:
            pd.DataFrame o None: Un DataFrame con las características seleccionadas, o None si falla la extracción.
        """
        if churn:
            rf_cols = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']
        else:
            rf_cols = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 
                      'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']
        
        rf_data = None
        try:
            # Verificar qué columnas existen en el dataset
            available_cols = self.data.columns.tolist()
            missing_cols = [col for col in rf_cols if col not in available_cols]
            
            if missing_cols:
                # Si faltan columnas, verificar si es el dataset correcto
                if churn and any(col in available_cols for col in ['ENGINESIZE', 'CO2EMISSIONS']):
                    # Está intentando usar datos de churn con URL de FuelConsumption
                    raise ValueError(
                        f"El dataset no contiene las columnas requeridas para análisis de churn: {missing_cols}. "
                        f"Asegúrate de usar la URL correcta del dataset de Churn."
                    )
                elif not churn and any(col in available_cols for col in ['tenure', 'churn']):
                    # Está intentando usar datos de combustible con URL de Churn
                    raise ValueError(
                        f"El dataset no contiene las columnas requeridas para análisis de combustible: {missing_cols}. "
                        f"Asegúrate de usar la URL correcta del dataset de FuelConsumption."
                    )
                else:
                    raise ValueError(
                        f"El dataset no contiene las columnas requeridas: {missing_cols}. "
                        f"Columnas disponibles: {available_cols}"
                    )
            
            rf_data = self.data[rf_cols]
            
            # Convertir 'churn' a entero si existe
            if churn and 'churn' in rf_data.columns: 
                rf_data['churn'] = rf_data['churn'].astype('int')
            
            print(f"Características relevantes seleccionadas: {rf_cols}")
            
        except Exception as e:
            rf_data = None
            print(f"Error: no se pudo extraer las características relevantes: {e}")
            raise  # Re-lanzar la excepción para que sea capturada por el código que llama
            
        return rf_data 
    
    def set_histogram(self, out: str) -> None:
        """
        Genera histogramas para las características relevantes seleccionadas y guarda la imagen en el archivo indicado.

        Args:
            out (str): Ruta donde se guardará la imagen del histograma.

        Returns:
            None
        """
        rf_cols = ['CO2EMISSIONS', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
                   'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG']
        try:
            viz = self.relevant_features[rf_cols]
            viz.hist()
            plt.savefig(out)
            plt.close()
            print(f"Se creó gráfico de histograma en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico de histograma en {out}: {e}")
        
    def plot_relationship(self, x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, out: str) -> None:
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
            data = self.correlated_features.iloc[:, cols].to_numpy()
            print(f"Columna{'s ' if len(cols) > 1 else ' '}{cols} extraída{'s.' if len(cols) > 1 else '.'}")
        except Exception as e:
            data = None
            print(f"Error: no se pudo extraer la información de columna{'s ' if len(cols) > 1 else ' '}{cols}: {e}")
        return data