# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: linear_regression.py
# Descripción: Definición de clase LinearRegression
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split

from regression_models.data_source import DataSource as ds

class MultipleLinearRegressionCompare:
    def __init__(self, url: str, corr: str, f1: str, f2: str, base: str, out: str):
        self.source = ds(url)
        # Característica base de comparación
        self.base = base
        # Obtener correlación
        self.source.correlate_relevant_features()
        self.source.plot_correlation(corr)
        # Opciones de características a correlacionar
        self.f1 = f1
        self.f2 = f2
        # Seleccionar las características para analisis de correlación
        self.x = self.source.get_correlation_columns(cols=[0,1])
        self.y = self.source.get_correlation_columns(cols=[2])
        # Preprocesamiento para estandarizar las características. De esta manera el modelo no se inclinará
        # a favor de ninguna característica debido a su magnitud.
        self.std_scaler, self.x_std = self.standarize(x=self.x)
        # Preparar información al 80% para entrenamiento y 20% para pruebas
        self.d = self.prepare_data(x=self.x, y=self.y, prc=0.2, random_state=42)
        # Creación de modelos de regresión lineal para las opciones
        self.m = self.create_model()
        # Entrenamiento de modelos
        self.train_model(self.m, self.d[0])
        # Coeficientes de regresor y la intercepción
        self.get_coef_and_int(self.m)
        # Gráficos de Regresión lineal múltiple
        self.plot_model_and_predict(model=self.m, x=self.d[1], y=self.y, x_label=self.f1.capitalize(), y_label=self.f1.capitalize(), z_label=self.base.capitalize(), 
                                    out=os.path.join(out, f"multiple_linear_regression_{self.f1.lower()}_{self.f2.lower()}_{self.base.lower()}.png")) 
        # Cortes verticales individuales del gráfico
        self.plot_variable(model=self.m, col=0, x=self.d[1], y=self.y, x_label=self.f1.capitalize(), y_label=self.base.capitalize(),
                        out=os.path.join(out, f"split_mlr_{self.f1.lower()}_{self.base.lower()}.png")) 
        self.plot_variable(model=self.m, col=1, x=self.d[1], y=self.y, x_label=self.f1.capitalize(), y_label=self.base.capitalize(),
                        out=os.path.join(out, f"split_mlr_{self.f2.lower()}_{self.base.lower()}.png")) 
        
    # TODO: Define un método que preprocese y estandarice las características correlacionadas. 
    #       La forma común de hacer esto es restar el promedio y dividir por la desviación estándar. 
    #       Scikit-learn tiene una implementación para esto.
    # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    #       https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit_transform
    def standarize(self, x: np.ndarray) -> tuple[preprocessing.StandardScaler, np.ndarray]:
        std_scaler = None
        x_std = None
        return std_scaler, x_std
    
    # TODO: Define un método que prepare la información para ser analizada por regresión lineal.
    #       Recuerda que al hacer un modelo de aprendizaje automático debemos dividir la información disponible en
    #       datos de entrenamiento y datos de pruebas, por lo tanto, a la salida debe 
    #       haber un tuple(x_train, x_test, y_train, y_test) de arreglos de numpy.
    # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    def prepare_data(self, x:np.ndarray, y:np.ndarray, prc: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Args:
            x (np.ndarray): Característica independiente.
            y (np.ndarray): Variable dependiente.
            prc (float): Proporción para prueba (entre 0 y 1).
            random_state (int): Semilla para reproducibilidad.

        Returns:
            tuple: (x_train, x_test, y_train, y_test)
        """
        x_train, x_test, y_train, y_test = None, None, None, None
        return (x_train, x_test, y_train, y_test)
    
    # TODO: Define un método que devuelva un objeto "linear_model.LinearRegression" de scikit-learn.
    # NOTE: https://scikit-learn.org/stable/modules/linear_model.html
    def create_model(self) -> linear_model.LinearRegression:
        """
        Crea un modelo de regresión lineal.

        Returns:
            LinearRegression: Modelo vacío listo para entrenar.
        """
        return None
    
    # TODO: Define un método que entrene un modelo de entrada "linear_model" de scikit-learn
    #       con la información de entrada "data".
    # NOTE: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html
    #       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit
    def train_model(self, model: linear_model.LinearRegression, data: np.ndarray) -> None:
        """
        Entrena el modelo con los datos de entrenamiento.

        Args:
            model (LinearRegression): Modelo a entrenar.
            data (tuple): (x_train, x_test, y_train, y_test)
        """
        pass
    
    # TODO: Define un método que obtenga los coeficientes de regresor y la intercepción
    #       de un modelo de entrada "linear_model" de scikit-learn.
    # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
    def get_coef_and_int(self, model: linear_model.LinearRegression) -> None:
        """
        Imprime los coeficientes y la intercepción del modelo.

        Args:
            model (LinearRegression): Modelo entrenado.
        """
        pass
        
    def plot_model_and_predict(self, model: linear_model.LinearRegression, x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, z_label: str, out: str) -> None:
        try:
            # Ensure X1, X2, and y_test have compatible shapes for 3D plotting
            X1 = x[:, 0] if x.ndim > 1 else x
            X2 = x[:, 1] if x.ndim > 1 else np.zeros_like(X1)

            # Create a mesh grid for plotting the regression plane
            x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), 
                                        np.linspace(X2.min(), X2.max(), 100))

            y_surf = model.intercept_ +  model.coef_[0,0] * x1_surf  +  model.coef_[0,1] * x2_surf

            # Predict y values using trained regression model to compare with actual y_test for above/below plane colors
            y_pred = model.predict(x.reshape(-1, 1)) if x.ndim == 1 else model.predict(x)
            above_plane = y >= y_pred
            below_plane = y < y_pred
            above_plane = above_plane[:,0]
            below_plane = below_plane[:,0]

            # Plotting
            fig = plt.figure(figsize=(20, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the data points above and below the plane in different colors
            ax.scatter(X1[above_plane], X2[above_plane], y[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
            ax.scatter(X1[below_plane], X2[below_plane], y[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')

            # Plot the regression plane
            ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')

            # Set view and labels
            ax.view_init(elev=10)

            ax.legend(fontsize='x-large',loc='upper center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect(None, zoom=0.75)
            ax.set_xlabel(x_label, fontsize='xx-large')
            ax.set_ylabel(y_label, fontsize='xx-large')
            ax.set_zlabel(z_label, fontsize='xx-large')
            ax.set_title(f'Multiple Linear Regression of {z_label}', fontsize='xx-large')
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print(f"Se creó gráfico de regresión lineal múltiple en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico del modelo: {e}")
    
    def plot_variable(self, model: linear_model.LinearRegression, col: int, x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, out: str) -> None:
        try:
            plt.scatter(x[:,col], y,  color='blue')
            plt.plot(x[:,col], model.coef_[0,col] * x[:,0] + model.intercept_[0], '-r')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(out)
            plt.close()
            print(f"Se creó gráfico de corte vertical de regresión lineal múltiple en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico de corte vertical de regresión lineal múltiple: {e}")