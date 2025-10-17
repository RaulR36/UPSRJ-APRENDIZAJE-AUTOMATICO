# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: main.py
# Descripción: Script principal del proyecto
# ============================================================
import introduction as intro
import regression_models as rm
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Ruta al archivo de entrada
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "estudiantes.csv")

# Ruta a la información de entrada
SOURCE_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

SEPARATOR = f"{'='*50}"

def introduction():

    # Status: OK
    status = os.EX_OK
    
    try:
        print(f"Procesando lista de estudiantes.")
        
        # Cargar estudiantes
        total, df = intro.csv_registers(CSV_FILE)
        if total is None or df is None:
            print("Error: No se pudo cargar el archivo CSV.\n")
            return os.EX_SOFTWARE
        print(f"- Estudiantes cargados: {total}")

        # Filtrar estudiantes con calificación > 8
        aprobados = intro.get_above(df, col="promedio", n=8)
        if aprobados is None or aprobados.empty:
            print("Error: No se pudo filtrar estudiantes aprobados.\n")
            return os.EX_SOFTWARE
        print(f"- Estudiantes aprobados:\n{aprobados}")

        # Agrupar por carrera y calcular promedio
        promedio_por_carrera = intro.group_and_average(aprobados, group="carrera", avg="promedio")
        if promedio_por_carrera is None or promedio_por_carrera.empty:
            print("Error: No se pudo calcular el promedio por carrera.\n")
            return os.EX_SOFTWARE
        print(f"- Promedio por carrera:\n{promedio_por_carrera}")

        # Exportar resultados
        OUTPUT = os.path.join(os.path.dirname(__file__), "outputs", "aprobados.csv")
        try:
            if not os.path.exists(os.path.dirname(OUTPUT)):
                os.mkdir(os.path.dirname(OUTPUT))
            intro.export_data(aprobados, OUTPUT)
            print(f"- Datos exportados a: {OUTPUT}")
        except:
            print("Error: No se pudo exportar el archivo CSV.\n")
            return os.EX_SOFTWARE

        # Comparar DataFrames
        son_iguales = intro.compare_dfs(df, aprobados)
        if son_iguales is None:
            print("Error: No se pudo comparar los DataFrames.\n")
            return os.EX_SOFTWARE
        print(f"- ¿Original y filtrado son iguales?: {son_iguales}")

        # Obtener columna de promedios
        if "promedio" not in df.columns:
            print("Error: La columna 'promedio' no existe en el DataFrame.\n")
            return os.EX_SOFTWARE
        calificaciones = df["promedio"].to_numpy()

        # Obtener estadísticas con NumPy
        mean, median, std = intro.get_statistics_numpy(calificaciones)
        if mean is None or median is None or std is None:
            print("Error: No se pudieron calcular las estadísticas.\n")
            return os.EX_SOFTWARE
        print(f"Estadísticas generales:\n- Promedio: {mean:.2f}\n- Mediana: {median:.2f}\n- Desviación estándar: {std:.2f}")

        # Simular señal de calificaciones y aplicar filtro pasa-bajas
        fs = 100
        t = np.linspace(0, 1, len(calificaciones), endpoint=False)
        ruido = 0.5 * np.sin(2 * np.pi * 50 * t)
        curva_suavizada = intro.low_pass_filter(calificaciones + ruido, fs)
        if curva_suavizada is None:
            print("Error: No se pudo aplicar el filtro pasa-bajas.\n")
            return os.EX_SOFTWARE

        # Reasignar calificaciones suavizadas
        df["calificacion curvada"] = curva_suavizada

        # Visualizar la calificación real vs. curvada
        # Analisis de justicia evaluativa, variabilidad y decisiones pedagógicas basadas en datos.
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(calificaciones, label="Calificación real", marker='o')
            plt.plot(curva_suavizada, label="Calificación curvada", marker='x', linestyle='--')
            plt.title("Comparación: Calificación real vs. curvada")
            plt.xlabel("Estudiante")
            plt.ylabel("Calificación")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(OUTPUT), "analisis.png"), dpi=300)
            plt.show()
            print("Gráfica guardada como 'analisis.png'\n")
        except:
            print("Error: No se pudo generar la gráfica.\n")
            return os.EX_SOFTWARE

        print(f"Completado.\n")
        
    except Exception as e:
        # Status: Error de software
        status = os.EX_SOFTWARE
        print(f"Error inesperado: {e}\n")
    
    # Return de la función: status EX_OK (0) | EX_SOFTWARE (70)
    return status   

def linear_regression():
    
    # Status: OK
    status = os.EX_OK
    
    # Definición de parámetros para comparación de modelos de regresión lineal
    OUTPUT = os.path.join(os.path.dirname(__file__), "output")
    HISTOGRAM = os.path.join(OUTPUT, "histogram.png")
    BASE = "CO2EMISSIONS"
    FEATURE_1 = "ENGINESIZE"
    FEATURE_2 = "FUELCONSUMPTION_COMB"
    
    try:
        # Creación de directorio
        if not os.path.exists(OUTPUT):
            os.mkdir(OUTPUT)
        
        # Comparación de modelos de regresión lineal
        rm.LinearRegressionCompare(url=SOURCE_URL, hist=HISTOGRAM, base=BASE, f1=FEATURE_1, f2=FEATURE_2, out=OUTPUT)
        print(f"Comparación de modelos: [{FEATURE_1} {FEATURE_2}] con {BASE} completada.\n")
    except Exception as e:
        # Status: Error de software
        status = os.EX_SOFTWARE
        print(f"Error inesperado: {e}\n")
        
    # Return de la función: status EX_OK (0) | EX_SOFTWARE (70)
    return status  

def multiple_linear_regression():
    # Status: OK
    status = os.EX_OK
    
    # Definición de parámetros para comparación de modelos de regresión lineal
    OUTPUT = os.path.join(os.path.dirname(__file__), "output")
    CORRELATION = os.path.join(OUTPUT, "correlation.png")
    BASE = "CO2EMISSIONS"
    FEATURE_1 = "ENGINESIZE"
    FEATURE_2 = "FUELCONSUMPTION_COMB"
    
    try:
        # Creación de directorio
        if not os.path.exists(OUTPUT):
            os.mkdir(OUTPUT)
        
        # Comparación de modelos de regresión lineal
        rm.MultipleLinearRegressionCompare(url=SOURCE_URL, corr=CORRELATION, base=BASE, f1=FEATURE_1, f2=FEATURE_2, out=OUTPUT)
        print(f"Correlación: [{FEATURE_1} {FEATURE_2}] con {BASE} completada.\n")
    except Exception as e:
        # Status: Error de software
        status = os.EX_SOFTWARE
        print(f"Error inesperado: {e}\n")
        
    # Return de la función: status EX_OK (0) | EX_SOFTWARE (70)
    return status  

def logistic_regression():
    # Status: OK
    status = os.EX_OK
    
    # Definición de parámetros para comparación de modelos de regresión logistica
    CHURN_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
    OUTPUT = os.path.join(os.path.dirname(__file__), "output")
    BASE = "churn"
        
    try:
        # Creación de directorio
        if not os.path.exists(OUTPUT):
            os.mkdir(OUTPUT)
        
        # Comparación de modelos de regresión lineal
        rm.LogisticRegressionCompare(url=CHURN_URL, base=BASE, out=OUTPUT)
        print(f"Coeficientes: '{BASE}' completada.\n")
    except Exception as e:
        # Status: Error de software
        status = os.EX_SOFTWARE
        print(f"Error inesperado: {e}\n")
        
    # Return de la función: status EX_OK (0) | EX_SOFTWARE (70)
    return status  

def main():
    
    # Status: OK
    status = os.EX_OK
    
    # Evaluación de primer ejercicio
    print(f"EJERCICIO 1")
    print(SEPARATOR)    
    status = introduction()
    
    # Evaluación de segundo ejercicio
    print(f"EJERCICIO 2")
    print(SEPARATOR)
    status = linear_regression()
    
    # Evaluación de tercer ejercicio
    print(f"EJERCICIO 3")
    print(SEPARATOR)
    status = multiple_linear_regression()
    
    # Evaluación de cuarto ejercicio
    print(f"EJERCICIO 4")
    print(SEPARATOR)
    status = logistic_regression()
    
    # Return de la función: status EX_OK (0) | EX_SOFTWARE (70)
    return status  

if __name__ == "__main__":
    sys.exit(main())