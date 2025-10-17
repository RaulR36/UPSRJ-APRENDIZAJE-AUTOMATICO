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
        # Cargar datos con churn=True para obtener dataset de churn
        self.source = ds(url, churn=True)
        
        # Definir columnas esperadas para el modelo de churn
        required_cols = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
        available_cols = self.source.data.columns.tolist()
        
        # Validar que existan las columnas necesarias
        valid_cols = [col for col in required_cols if col in available_cols]
        
        if not valid_cols:
            raise ValueError(f"El dataset no contiene ninguna columna requerida. Se esperaban: {required_cols}, pero se encontraron: {available_cols}")
        
        # Verificar que la columna base (target) existe
        if base not in available_cols:
            raise ValueError(f"La columna objetivo '{base}' no existe en el dataset. Columnas disponibles: {available_cols}")
        
        # Extraer características y variable objetivo
        self.x = np.asarray(self.source.data[valid_cols])
        self.y = np.asarray(self.source.data[base])
        
        # Preprocesamiento para estandarizar las características
        self.std_scaler, self.x_std = self.standarize(x=self.x)
        
        # Preparar datos de entrenamiento y prueba
        self.d = self.prepare_data(x=self.x_std, y=self.y, prc=0.2, random_state=4)
        
        # Crear y entrenar modelo
        self.m = self.create_model()
        self.train_model(self.m, self.d)
        
        # Generar visualización
        self.plot_model_and_predict(
            self.m, 
            index=valid_cols, 
            x=self.d[1], 
            y=self.d[3], 
            out=os.path.join(out, "logistic_regression_churn_coefficients.png")
        )
        
    def standarize(self, x: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
        """
        Estandariza las características usando StandardScaler.
        
        Args:
            x (np.ndarray): Array de características a estandarizar
            
        Returns:
            tuple: (StandardScaler ajustado, array estandarizado)
        """
        std_scaler = StandardScaler()
        x_std = std_scaler.fit_transform(x)
        return std_scaler, x_std
    
    def prepare_data(self, x: np.ndarray, y: np.ndarray, prc: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Args:
            x (np.ndarray): Características independientes.
            y (np.ndarray): Variable dependiente.
            prc (float): Proporción para prueba (entre 0 y 1).
            random_state (int): Semilla para reproducibilidad.

        Returns:
            tuple: (x_train, x_test, y_train, y_test)
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=prc, random_state=random_state
        )
        return x_train, x_test, y_train, y_test

    def create_model(self) -> LogisticRegression:
        """
        Crea un modelo de regresión logística.

        Returns:
            LogisticRegression: Modelo vacío listo para entrenar.
        """
        return LogisticRegression(C=0.01, solver='liblinear')
    
    def train_model(self, model: LogisticRegression, data: tuple) -> None:
        """
        Entrena el modelo con los datos de entrenamiento.

        Args:
            model (LogisticRegression): Modelo a entrenar.
            data (tuple): (x_train, x_test, y_train, y_test)
        """
        x_train, x_test, y_train, y_test = data
        model.fit(x_train, y_train)
    
    def plot_model_and_predict(self, model: LogisticRegression, index, x: np.ndarray, y: np.ndarray, out: str) -> None:
        """
        Genera gráfico de coeficientes y calcula log loss.
        
        Args:
            model: Modelo entrenado
            index: Nombres de las características
            x: Datos de prueba
            y: Etiquetas verdaderas
            out: Ruta de salida para el gráfico
        """
        try:
            yhat_prob = model.predict_proba(x)
            coefficients = pd.Series(model.coef_[0], index=index)
            coefficients.sort_values().plot(kind='barh')
            plt.title("Feature Coefficients in Logistic Regression Churn Model")
            plt.xlabel("Coefficient Value")
            plt.savefig(out)
            plt.close()
            loss = log_loss(y, yhat_prob)
            print(f"Se creó gráfico de coeficientes en {out}")
            print(f"Log Loss: {loss:.4f}")
        except Exception as e:
            print(f"Error: no se pudo crear gráfico de coeficientes en {out}: {e}")