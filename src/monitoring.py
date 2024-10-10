import pandas as pd
import mlflow
import numpy as np
from scipy.stats import entropy
from collections import deque
import threading
import time
import os
from dotenv import load_dotenv

# Cargar variables de entorno (si es necesario)
stage = os.getenv('ENV', 'dev')
load_dotenv(f".env.{stage}")

# Clase para la detección de data drift
class DataDriftMonitor:
    def __init__(self, X_train_path, buffer_size=100, drift_threshold=0.1):
        self.X_train = pd.read_csv(X_train_path)
        self.buffer = deque(maxlen=buffer_size)  # Buffer circular para los datos en vivo
        self.buffer_lock = threading.Lock()  # Bloqueo para asegurar la sincronización en el acceso al buffer
        self.drift_threshold = drift_threshold  # Umbral para reportar el data drift significativo

    def add_live_data(self, live_data):
        """Agrega datos en vivo al buffer."""
        with self.buffer_lock:
            self.buffer.append(live_data)

    def kl_divergence(self, p, q):
        """Calcula la divergencia KL entre dos distribuciones."""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        p = p / p.sum()  # Normaliza
        q = q / q.sum()  # Normaliza
        return entropy(p, q)

    def check_data_drift(self):
        """Compara los datos en vivo almacenados en el buffer con los datos de entrenamiento y calcula la divergencia KL."""
        drift_metrics = {}
        with self.buffer_lock:
            if len(self.buffer) == self.buffer.maxlen:  # Solo si el buffer está lleno
                X_live_buffer = pd.concat(list(self.buffer), ignore_index=True)
                for column in self.X_train.columns:
                    if column in X_live_buffer.columns:
                        kl_div = self.kl_divergence(self.X_train[column], X_live_buffer[column])
                        drift_metrics[column] = kl_div
        return drift_metrics

    def detect_significant_drift(self, drift_metrics):
        """Detecta si alguno de los valores de drift es mayor que el umbral."""
        significant_drift = {col: val for col, val in drift_metrics.items() if val > self.drift_threshold}
        return significant_drift

# Clase para manejar el monitoreo en segundo plano
class MonitoringTask:
    def __init__(self, X_train_path, experiment_name="Monitoring", buffer_size=100, drift_threshold=0.1):
        self.data_monitor = DataDriftMonitor(X_train_path, buffer_size, drift_threshold)
        self.experiment_name = experiment_name

    def background_task(self):
        """Realiza el monitoreo en segundo plano, registrando métricas en MLflow."""
        while True:
            time.sleep(5)  # Esperar entre chequeos de drift
            drift_metrics = self.data_monitor.check_data_drift()
            if drift_metrics:
                significant_drift = self.data_monitor.detect_significant_drift(drift_metrics)

                if significant_drift:
                    # Loguea las métricas de drift en MLflow
                    mlflow.set_experiment(self.experiment_name)
                    with mlflow.start_run():
                        mlflow.log_metrics(significant_drift)
                    print(f"Significant data drift detected and logged: {significant_drift}")

# Ejecución del monitoreo
if __name__ == "__main__":
    X_train_path = "data/processed/X_train.csv"  # Ruta a los datos de entrenamiento

    # Crear una instancia del monitoreo en segundo plano
    monitoring_task = MonitoringTask(X_train_path)

    # Iniciar la tarea en un hilo separado para el monitoreo continuo
    monitoring_thread = threading.Thread(target=monitoring_task.background_task)
    monitoring_thread.start()
