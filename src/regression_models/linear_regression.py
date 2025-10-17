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
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regression_models.data_source import DataSource as ds

class LinearRegressionCompare:
    def __init__(self, url: str, hist: str, base: str, f1: str, f2: str, out: str):
        self.source = ds(url)
        self.histogram = hist
        # Característica base de comparación
        self.base = base
        # Opciones de características a comparar
        self.f1 = f1
        self.f2 = f2
        # Histograma
        self.source.set_histogram(self.histogram)
        # Seleccionar las características para regresión lineal
        self.x1 = self.source.get_relevant_feature(self.f1)
        self.x2 = self.source.get_relevant_feature(self.f2)
        self.y = self.source.get_relevant_feature(self.base)
        # Relación entre opciones y base
        self.source.plot_relationship(x=self.x1, y=self.y, x_label=self.f1.capitalize(), y_label=self.base.capitalize(), 
                             out=os.path.join(out, f"relationship_{self.f1.lower()}_{self.base.lower()}.png"))
        self.source.plot_relationship(x=self.x2, y=self.y, x_label=self.f2.capitalize(), y_label=self.base.capitalize(), 
                             out=os.path.join(out, f"relationship_{self.f2.lower()}_{self.base.lower()}.png"))
        # Preparar información al 80% para entrenamiento y 20% para pruebas
        self.d1 = self.prepare_data(x=self.x1, y=self.y, prc=0.2, random_state=42)
        self.d2 = self.prepare_data(x=self.x2, y=self.y, prc=0.2, random_state=42)
        # Creación de modelos de regresión lineal para las opciones
        self.m1 = self.create_model()
        self.m2 = self.create_model()
        # Entrenamiento de modelos
        self.train_model(self.m1, self.d1)
        self.train_model(self.m2, self.d2)
        # Coeficientes de regresor y la intercepción
        self.get_coef_and_int(self.m1)
        self.get_coef_and_int(self.m2)
        # Predicciones con modelos
        self.p1 = self.predict(self.m1, self.x1)
        self.p2 = self.predict(self.m2, self.x2)
        # Evaluación de modelos
        self.evaluate(self.y, self.p1)
        self.evaluate(self.y, self.p2)
        # Gráfico de Regresión lineal
        self.plot_model(model=self.m1, x=self.x1, y=self.y, x_label=self.f1.capitalize(), y_label=self.base.capitalize(),
                        out=os.path.join(out, f"linear_regression_{self.f1.lower()}_{self.base.lower()}.png"))        
        self.plot_model(model=self.m2, x=self.x2, y=self.y, x_label=self.f2.capitalize(), y_label=self.base.capitalize(),
                        out=os.path.join(out, f"linear_regression_{self.f2.lower()}_{self.base.lower()}.png"))
        
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
        x_reshaped = x.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x_reshaped, y, test_size=prc, random_state=random_state
        )
        
        return (x_train, x_test, y_train, y_test)
    
    def create_model(self) -> linear_model.LinearRegression:
        """
        Crea un modelo de regresión lineal.

        Returns:
            LinearRegression: Modelo vacío listo para entrenar.
        """
        return linear_model.LinearRegression()
    
    def train_model(self, model: linear_model.LinearRegression, data: tuple) -> None:
        """
        Entrena el modelo con los datos de entrenamiento.

        Args:
            model (LinearRegression): Modelo a entrenar.
            data (tuple): (x_train, x_test, y_train, y_test)
        """
        x_train, x_test, y_train, y_test = data
        model.fit(x_train, y_train)
        
        if len(model.coef_.shape) == 1:
            model.coef_ = model.coef_.reshape(1, -1)
    
    def get_coef_and_int(self, model: linear_model.LinearRegression) -> None:
        """
        Imprime los coeficientes y la intercepción del modelo.

        Args:
            model (LinearRegression): Modelo entrenado.
        """
        print(f"Coeficiente(s): {model.coef_}")
        print(f"Intercepción: {model.intercept_}")
    
    def predict(self, model: linear_model.LinearRegression, x: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo.

        Args:
            model (LinearRegression): Modelo entrenado.
            x (np.ndarray): Datos de entrada.

        Returns:
            np.ndarray: Predicciones.
        """
        x_reshaped = x.reshape(-1, 1)
        prediction = model.predict(x_reshaped)
        return prediction
    
    def evaluate(self, y: np.ndarray, prediction: np.ndarray) -> None:
        print("Evaluación de modelo:")
        print("- Mean absolute error: %.2f" % mean_absolute_error(y, prediction))
        print("- Mean squared error: %.2f" % mean_squared_error(y, prediction))
        print("- Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y, prediction)))
        print("- R2-score: %.2f" % r2_score(y, prediction))
        
    def plot_model(self, model: linear_model.LinearRegression, x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, out: str) -> None:
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, color='blue', alpha=0.6, label='Datos reales')
            
            # Crear la línea de regresión
            x_reshaped = x.reshape(-1, 1)
            y_pred = model.predict(x_reshaped)
            plt.plot(x, y_pred, '-r', linewidth=2, label='Línea de regresión')
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print(f"Se creó gráfico de regresión lineal en {out}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico del modelo: {e}")