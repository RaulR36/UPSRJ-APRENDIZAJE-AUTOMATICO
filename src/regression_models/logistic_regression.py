# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: logistic_regression.py
# Descripción: Definición de clase LogisticRegression
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

from regression_models.data_source import DataSource as ds

class LogisticRegressionCompare:
    def __init__(self, url: str, base: str, out: str):
        self.source = ds(url, churn=True)
        self.x = np.asarray(self.source.data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
        self.y = np.asarray(self.source.data[base])
        # Preprocesamiento para estandarizar las características. De esta manera el modelo no se inclinará
        # a favor de ninguna característica debido a su magnitud.
        self.std_scaler, self.x_std = self.standarize(x=self.x)
        self.d = self.prepare_data(x=self.x_std, y=self.y, prc=0.2, random_state=4)
        self.m = self.create_model()
        self.train_model(self.m, self.d)
        self.plot_model_and_predict(self.m, index=self.source.data.columns[:-1], x=self.d[1], 
                                    y=self.d[3], out=os.path.join(out, "logistic_regression_churn_coefficients.png"))
        
    # TODO: Define un método que preprocese y estandarice las características correlacionadas. 
    #       La forma común de hacer esto es restar el promedio y dividir por la desviación estándar. 
    #       Scikit-learn tiene una implementación para esto.
    # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    #       https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit_transform
    def standarize(self, x: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
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

    # TODO: Define un método que devuelva un objeto "linear_model.LogisticRegression" de scikit-learn.
    # NOTE: https://scikit-learn.org/stable/modules/linear_model.html
    def create_model(self) -> LogisticRegression:
        """
        Crea un modelo de regresión lineal.

        Returns:
            LogisticRegression: Modelo vacío listo para entrenar.
        """
        return None
    
    # TODO: Define un método que entrene un modelo de entrada "linear_model" de scikit-learn
    #       con la información de entrada "data".
    # NOTE: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html
    #       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit
    def train_model(self, model: LogisticRegression, data: np.ndarray) -> None:
        """
        Entrena el modelo con los datos de entrenamiento.

        Args:
            model (LogisticRegression): Modelo a entrenar.
            data (tuple): (x_train, x_test, y_train, y_test)
        """
        pass
    
    def plot_model_and_predict(self, model: LogisticRegression, index, x: np.ndarray, y:np.ndarray, out: str) -> None:
        try:
            yhat_prob = model.predict_proba(x)
            coefficients = pd.Series(model.coef_[0], index=index)
            coefficients.sort_values().plot(kind='barh')
            plt.title("Feature Coefficients in Logistic Regression Churn Model")
            plt.xlabel("Coefficient Value")
            plt.savefig(out)
            plt.close()
            log_loss(y, yhat_prob)
            print(f"Se creó gráfico de coeficientes en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico de coeficientes en {out}: {e}")