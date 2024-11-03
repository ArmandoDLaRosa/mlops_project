import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from io import StringIO
from DataDriftMonitor import DataDriftMonitor

class TestDataDriftMonitor(unittest.TestCase):

    def setUp(self):
        # Crea un conjunto de datos de entrenamiento simulado
        train_data = StringIO("""
        col1,col2,col3
        1,2,3
        4,5,6
        7,8,9
        10,11,12
        """)
        self.train_path = 'train_data.csv'
        pd.DataFrame({
            'col1': [1, 4, 7, 10],
            'col2': [2, 5, 8, 11],
            'col3': [3, 6, 9, 12]
        }).to_csv(self.train_path, index=False)

        self.monitor = DataDriftMonitor(X_train_path=self.train_path, buffer_size=2, drift_threshold=0.05)

    def test_add_live_data(self):
        live_data = pd.DataFrame({'col1': [2], 'col2': [3], 'col3': [4]})
        self.monitor.add_live_data(live_data)

        with self.monitor.buffer_lock:
            self.assertEqual(len(self.monitor.buffer), 1)

    def test_check_data_drift_no_drift(self):
        # Agrega datos en vivo que son similares a los de entrenamiento
        live_data1 = pd.DataFrame({'col1': [4], 'col2': [5], 'col3': [6]})
        live_data2 = pd.DataFrame({'col1': [7], 'col2': [8], 'col3': [9]})
        
        self.monitor.add_live_data(live_data1)
        self.monitor.add_live_data(live_data2)
        
        drift_metrics = self.monitor.check_data_drift(force=True)
        
        # Verifica que se generen métricas de drift
        self.assertTrue('col1' in drift_metrics)
        self.assertTrue('col2' in drift_metrics)
        self.assertTrue('col3' in drift_metrics)

        # Verifica que las métricas de drift sean menores que el umbral (simula sin drift)
        for col, metrics in drift_metrics.items():
            self.assertLess(metrics['KL_Divergence'], self.monitor.drift_threshold)
            self.assertLess(metrics['KS_Statistic'], self.monitor.drift_threshold)

    def test_detect_significant_drift(self):
        # Agrega datos en vivo diferentes para provocar drift
        live_data1 = pd.DataFrame({'col1': [100], 'col2': [200], 'col3': [300]})
        live_data2 = pd.DataFrame({'col1': [150], 'col2': [250], 'col3': [350]})
        
        self.monitor.add_live_data(live_data1)
        self.monitor.add_live_data(live_data2)
        
        drift_metrics = self.monitor.check_data_drift(force=True)
        significant_drift = self.monitor.detect_significant_drift(drift_metrics)
        
        # Verifica que se detecte drift significativo
        self.assertTrue(len(significant_drift) > 0)
