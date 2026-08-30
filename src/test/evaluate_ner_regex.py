import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ner.extractor import MedicalExtractor

def ordenar(s):
    """Ordena archivos que contienen números de forma lógica (e.g., Analitica2 antes que Analitica10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def cargar_analiticas(directorio='analiticas/lote1'):
    """Carga todos los archivos txt del directorio a un dataframe, 
       manejando múltiples codificaciones de texto."""
    datos = []
    if not os.path.exists(directorio):
        print(f"Directorio '{directorio}' no encontrado.")
        return pd.DataFrame()

    archivos = os.listdir(directorio)
    for nombre_archivo in archivos:
        ruta_completa = os.path.join(directorio, nombre_archivo)
        
        if nombre_archivo.endswith(".txt") and os.path.isfile(ruta_completa):
            # 1. Intentamos leer en UTF-8
            try:
                with open(ruta_completa, 'r', encoding='utf-8') as archivo:
                    contenido = archivo.read()
            # 2. Si falla por la codificación, pasamos a Latin-1 (común en Windows/España)
            except UnicodeDecodeError:
                with open(ruta_completa, 'r', encoding='latin-1') as archivo:
                    contenido = archivo.read()
                    
            datos.append({
                'id': nombre_archivo.replace(".txt", ""),
                'texto': contenido
            })

    df = pd.DataFrame(datos)
    if not df.empty:
        df = df.sort_values(by='id', key=lambda x: x.map(ordenar)).reset_index(drop=True)
        return df
    return pd.DataFrame()


# Etiquetas reales con Gemini
df_lote1 = [
    # Analítica 1
    {'Hb': 14.5, 'Glucosa': 92, 'Creatinina': 0.50, 'Plaquetas': 147, 'Neutrófilos': 3.41, 'Linfocitos': 2.91, 'Calcio': 9.6, 'Potasio': 4.7, 'Sodio': 139, 'Magnesio': None, 'ALT': 83, 'AST': 51, 'LDH': 274, 'GGT': 297, 'FA': 180, 'Bilirrubina': 0.3, 'Albumina': 4.0, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 2
    {'Hb': 14.6, 'Glucosa': 65, 'Creatinina': 0.71, 'Plaquetas': 243, 'Neutrófilos': 6.22, 'Linfocitos': 1.68, 'Calcio': 9.0, 'Potasio': 4.5, 'Sodio': 140, 'Magnesio': 2.0, 'ALT': 14, 'AST': 18, 'LDH': 313, 'GGT': 17, 'FA': 46, 'Bilirrubina': 0.4, 'Albumina': 3.9, 'Proteinas': 6.7, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 178, 'HDL': None, 'Trigliceridos': 93, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': 5.0},
    # Analítica 3
    {'Hb': 11.1, 'Glucosa': 99, 'Creatinina': 0.61, 'Plaquetas': 203, 'Neutrófilos': 2.42, 'Linfocitos': 1.59, 'Calcio': 9.8, 'Potasio': 3.9, 'Sodio': 138, 'Magnesio': 2.0, 'ALT': 15, 'AST': 18, 'LDH': 309, 'GGT': 13, 'FA': 66, 'Bilirrubina': 0.8, 'Albumina': 4.2, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 4
    {'Hb': 13.6, 'Glucosa': 95, 'Creatinina': 0.82, 'Plaquetas': 284, 'Neutrófilos': 3.84, 'Linfocitos': 2.25, 'Calcio': 9.9, 'Potasio': 4.1, 'Sodio': 140, 'Magnesio': None, 'ALT': 51, 'AST': 38, 'LDH': 344, 'GGT': 15, 'FA': 82, 'Bilirrubina': 0.7, 'Albumina': 4.0, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 5
    {'Hb': 13.3, 'Glucosa': 95, 'Creatinina': 0.90, 'Plaquetas': 238, 'Neutrófilos': 3.77, 'Linfocitos': 2.54, 'Calcio': 9.5, 'Potasio': 4.3, 'Sodio': 139, 'Magnesio': 2.2, 'ALT': 31, 'AST': 35, 'LDH': None, 'GGT': 22, 'FA': 58, 'Bilirrubina': None, 'Albumina': 3.9, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 6
    {'Hb': 13.5, 'Glucosa': 79, 'Creatinina': 0.83, 'Plaquetas': 327, 'Neutrófilos': 4.77, 'Linfocitos': 2.03, 'Calcio': 9.8, 'Potasio': 5.0, 'Sodio': 138, 'Magnesio': None, 'ALT': 16, 'AST': 16, 'LDH': 313, 'GGT': 14, 'FA': 41, 'Bilirrubina': 0.4, 'Albumina': 4.0, 'Proteinas': 6.7, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 180, 'HDL': None, 'Trigliceridos': 47, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 7
    {'Hb': None, 'Glucosa': 96, 'Creatinina': 0.84, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 9.4, 'Potasio': 4.2, 'Sodio': 142, 'Magnesio': None, 'ALT': 13, 'AST': 19, 'LDH': 179, 'GGT': 19, 'FA': 115, 'Bilirrubina': 0.3, 'Albumina': 4.0, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': 5.0},
    # Analítica 8
    {'Hb': None, 'Glucosa': 100, 'Creatinina': 0.78, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 9.6, 'Potasio': 4.0, 'Sodio': 140, 'Magnesio': None, 'ALT': 21, 'AST': 16, 'LDH': 157, 'GGT': 19, 'FA': 98, 'Bilirrubina': None, 'Albumina': 4.3, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': 6.0},
    # Analítica 9
    {'Hb': None, 'Glucosa': 103, 'Creatinina': 0.75, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 9.4, 'Potasio': 4.2, 'Sodio': 142, 'Magnesio': None, 'ALT': 25, 'AST': 16, 'LDH': 237, 'GGT': 22, 'FA': 84, 'Bilirrubina': None, 'Albumina': 4.3, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 10
    {'Hb': None, 'Glucosa': 68, 'Creatinina': 0.68, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 9.0, 'Potasio': None, 'Sodio': 137, 'Magnesio': None, 'ALT': 12, 'AST': 20, 'LDH': None, 'GGT': 5, 'FA': 47, 'Bilirrubina': 0.5, 'Albumina': 4.3, 'Proteinas': 7.2, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 188, 'HDL': 59, 'Trigliceridos': 56, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 11
    {'Hb': None, 'Glucosa': 84, 'Creatinina': 0.62, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 9.8, 'Potasio': 4.5, 'Sodio': 141, 'Magnesio': None, 'ALT': 28, 'AST': 30, 'LDH': 185, 'GGT': 10, 'FA': 63, 'Bilirrubina': 0.5, 'Albumina': 4.4, 'Proteinas': 7.3, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 208, 'HDL': 63, 'Trigliceridos': 72, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 12
    {'Hb': None, 'Glucosa': 78, 'Creatinina': 0.62, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 9.2, 'Potasio': 4.3, 'Sodio': 140, 'Magnesio': None, 'ALT': 19, 'AST': 24, 'LDH': 163, 'GGT': 6, 'FA': 64, 'Bilirrubina': 0.8, 'Albumina': 4.4, 'Proteinas': 7.2, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 213, 'HDL': 77, 'Trigliceridos': 70, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 13
    {'Hb': None, 'Glucosa': 104, 'Creatinina': 0.58, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 10.1, 'Potasio': 5.1, 'Sodio': 140, 'Magnesio': None, 'ALT': 20, 'AST': 20, 'LDH': 157, 'GGT': 33, 'FA': 68, 'Bilirrubina': 0.7, 'Albumina': 4.6, 'Proteinas': 7.4, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 218, 'HDL': 71, 'Trigliceridos': 113, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 14
    {'Hb': None, 'Glucosa': 103, 'Creatinina': 0.69, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 9.9, 'Potasio': 4.9, 'Sodio': 139, 'Magnesio': None, 'ALT': 15, 'AST': 19, 'LDH': 159, 'GGT': 40, 'FA': 72, 'Bilirrubina': 0.7, 'Albumina': 4.7, 'Proteinas': 7.2, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 211, 'HDL': 62, 'Trigliceridos': 99, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 15
    {'Hb': 13.9, 'Glucosa': 114, 'Creatinina': 0.92, 'Plaquetas': 173, 'Neutrófilos': 4.1, 'Linfocitos': 1.5, 'Calcio': None, 'Potasio': 4.3, 'Sodio': 143.0, 'Magnesio': None, 'ALT': 35, 'AST': 28, 'LDH': 343, 'GGT': 33, 'FA': 53, 'Bilirrubina': 0.5, 'Albumina': None, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 16
    {'Hb': 11.1, 'Glucosa': None, 'Creatinina': 0.76, 'Plaquetas': 358000.0, 'Neutrófilos': 4.9, 'Linfocitos': 0.9, 'Calcio': None, 'Potasio': 4.9, 'Sodio': 133.0, 'Magnesio': None, 'ALT': 114, 'AST': 67, 'LDH': 258, 'GGT': 686, 'FA': 452, 'Bilirrubina': 0.76, 'Albumina': None, 'Proteinas': None, 'Lipasa': None, 'Amilasa': 112, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 17
    {'Hb': 12.0, 'Glucosa': None, 'Creatinina': None, 'Plaquetas': 471, 'Neutrófilos': 3.3, 'Linfocitos': None, 'Calcio': None, 'Potasio': None, 'Sodio': None, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': None, 'GGT': 222, 'FA': None, 'Bilirrubina': None, 'Albumina': None, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 18
    {'Hb': None, 'Glucosa': 115, 'Creatinina': 0.81, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 9.7, 'Potasio': 4.71, 'Sodio': 141.8, 'Magnesio': None, 'ALT': 21, 'AST': 22, 'LDH': 328, 'GGT': 39, 'FA': 68, 'Bilirrubina': 0.94, 'Albumina': None, 'Proteinas': 7.08, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 19
    {'Hb': None, 'Glucosa': 75, 'Creatinina': 0.57, 'Plaquetas': None, 'Neutrófilos': None, 'Linfocitos': None, 'Calcio': 8.9, 'Potasio': 4.3, 'Sodio': 135, 'Magnesio': None, 'ALT': 69, 'AST': 63, 'LDH': 143, 'GGT': 467, 'FA': 423, 'Bilirrubina': 2.07, 'Albumina': 3.2, 'Proteinas': 6.4, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 20
    {'Hb': 12.4, 'Glucosa': 89, 'Creatinina': 0.93, 'Plaquetas': 258, 'Neutrófilos': 3.81, 'Linfocitos': 3.47, 'Calcio': 9.6, 'Potasio': 5.1, 'Sodio': 140, 'Magnesio': None, 'ALT': 43, 'AST': 38, 'LDH': 182, 'GGT': 87, 'FA': 125, 'Bilirrubina': 2.92, 'Albumina': 4.0, 'Proteinas': 6.8, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 21
    {'Hb': 11.9, 'Glucosa': 89, 'Creatinina': 0.44, 'Plaquetas': 297, 'Neutrófilos': 4.4, 'Linfocitos': 2.0, 'Calcio': 9.3, 'Potasio': 4.0, 'Sodio': 142, 'Magnesio': None, 'ALT': None, 'AST': 19, 'LDH': 196, 'GGT': 10, 'FA': 51, 'Bilirrubina': 0.2, 'Albumina': 4.2, 'Proteinas': 6.5, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 22
    {'Hb': 8.7, 'Glucosa': 104, 'Creatinina': 1.21, 'Plaquetas': 237, 'Neutrófilos': 5.7, 'Linfocitos': 2.5, 'Calcio': 9.7, 'Potasio': None, 'Sodio': None, 'Magnesio': None, 'ALT': 20, 'AST': 27, 'LDH': 244, 'GGT': 114, 'FA': 132, 'Bilirrubina': 0.3, 'Albumina': None, 'Proteinas': 68.0, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 23
    {'Hb': 11.7, 'Glucosa': 105, 'Creatinina': 1.39, 'Plaquetas': 577, 'Neutrófilos': 9.52, 'Linfocitos': 2.04, 'Calcio': 9.6, 'Potasio': 4.6, 'Sodio': 139, 'Magnesio': None, 'ALT': 29, 'AST': 23, 'LDH': 172, 'GGT': 27, 'FA': 88, 'Bilirrubina': 0.3, 'Albumina': 3.4, 'Proteinas': 6.7, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 133, 'HDL': 33, 'Trigliceridos': 139, 'Creatinquinasa': None, 'TTPA': 20.8, 'Fibrinogeno': None, 'INR': 1.09, 'PH': None},
    # Analítica 24
    {'Hb': 13.2, 'Glucosa': 246, 'Creatinina': 0.56, 'Plaquetas': 162, 'Neutrófilos': 1.44, 'Linfocitos': 1.64, 'Calcio': 9.3, 'Potasio': 3.6, 'Sodio': 139, 'Magnesio': None, 'ALT': 21, 'AST': 23, 'LDH': 182, 'GGT': 79, 'FA': 59, 'Bilirrubina': 0.49, 'Albumina': None, 'Proteinas': 6.5, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 144, 'HDL': 56, 'Trigliceridos': 126, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    # Analítica 25
    {'Hb': 13.2, 'Glucosa': 246, 'Creatinina': 0.56, 'Plaquetas': 162, 'Neutrófilos': 1.44, 'Linfocitos': 1.64, 'Calcio': 9.3, 'Potasio': 3.6, 'Sodio': 139, 'Magnesio': None, 'ALT': 21, 'AST': 23, 'LDH': 182, 'GGT': 79, 'FA': 59, 'Bilirrubina': 0.49, 'Albumina': None, 'Proteinas': 6.5, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 144, 'HDL': 56, 'Trigliceridos': 126, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None}
]

df_lote2 = [
    {'Hb': 11.2, 'Glucosa': 103, 'Creatinina': 1.14, 'Plaquetas': 188, 'Neutrófilos': 4.08, 'Linfocitos': 1.33, 'Calcio': 9.0, 'Potasio': 3.86, 'Sodio': 143, 'Magnesio': None, 'ALT': 18, 'AST': None, 'LDH': 193, 'GGT': None, 'FA': None, 'Bilirrubina': 0.78, 'Albumina': 4.12, 'Proteinas': 7.10, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 13.7, 'Glucosa': 92, 'Creatinina': 1.18, 'Plaquetas': 197, 'Neutrófilos': 2.22, 'Linfocitos': 1.71, 'Calcio': 9.8, 'Potasio': 4.88, 'Sodio': 143, 'Magnesio': None, 'ALT': 19, 'AST': None, 'LDH': 209, 'GGT': None, 'FA': None, 'Bilirrubina': 0.49, 'Albumina': 4.07, 'Proteinas': 7.10, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 9.6, 'Glucosa': 88, 'Creatinina': 0.74, 'Plaquetas': 320, 'Neutrófilos': 2.44, 'Linfocitos': 1.50, 'Calcio': 9.6, 'Potasio': 4.91, 'Sodio': 143, 'Magnesio': None, 'ALT': 19, 'AST': None, 'LDH': 233, 'GGT': None, 'FA': None, 'Bilirrubina': 0.25, 'Albumina': 3.81, 'Proteinas': 7.10, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 14.4, 'Glucosa': 138, 'Creatinina': 0.81, 'Plaquetas': 356, 'Neutrófilos': 7.49, 'Linfocitos': 0.83, 'Calcio': 9.9, 'Potasio': 4.00, 'Sodio': 139, 'Magnesio': None, 'ALT': 20, 'AST': None, 'LDH': 230, 'GGT': None, 'FA': None, 'Bilirrubina': 0.93, 'Albumina': 3.82, 'Proteinas': 7.50, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.1, 'Glucosa': 118, 'Creatinina': 0.67, 'Plaquetas': 439, 'Neutrófilos': 5.04, 'Linfocitos': 1.46, 'Calcio': 9.3, 'Potasio': 4.52, 'Sodio': 142, 'Magnesio': None, 'ALT': 19, 'AST': None, 'LDH': 243, 'GGT': None, 'FA': None, 'Bilirrubina': 0.71, 'Albumina': 3.00, 'Proteinas': 7.10, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.3, 'Glucosa': 97, 'Creatinina': 0.72, 'Plaquetas': 128, 'Neutrófilos': 2.13, 'Linfocitos': 1.31, 'Calcio': 9.7, 'Potasio': 6.74, 'Sodio': 139, 'Magnesio': None, 'ALT': 35, 'AST': None, 'LDH': 371, 'GGT': None, 'FA': None, 'Bilirrubina': 0.57, 'Albumina': 4.09, 'Proteinas': 7.70, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.2, 'Glucosa': 82, 'Creatinina': 0.77, 'Plaquetas': 255, 'Neutrófilos': 2.94, 'Linfocitos': 0.79, 'Calcio': 9.7, 'Potasio': 5.56, 'Sodio': 143, 'Magnesio': None, 'ALT': 9, 'AST': None, 'LDH': 210, 'GGT': None, 'FA': None, 'Bilirrubina': 0.32, 'Albumina': 3.54, 'Proteinas': 6.90, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None}, 
    {'Hb': 13.7, 'Glucosa': 112, 'Creatinina': 0.73, 'Plaquetas': 419, 'Neutrófilos': 8.04, 'Linfocitos': 2.01, 'Calcio': 9.5, 'Potasio': 4.88, 'Sodio': 141, 'Magnesio': None, 'ALT': 78, 'AST': 27, 'LDH': 191, 'GGT': None, 'FA': None, 'Bilirrubina': 0.28, 'Albumina': 3.46, 'Proteinas': 7.30, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 10.8, 'Glucosa': 90, 'Creatinina': 0.78, 'Plaquetas': 236, 'Neutrófilos': 3.60, 'Linfocitos': 1.20, 'Calcio': 9.8, 'Potasio': 3.8, 'Sodio': 139, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 156, 'GGT': None, 'FA': None, 'Bilirrubina': 0.52, 'Albumina': 4.0, 'Proteinas': 7.8, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 11.9, 'Glucosa': 104, 'Creatinina': 0.82, 'Plaquetas': 334, 'Neutrófilos': 1.70, 'Linfocitos': 1.40, 'Calcio': 9.6, 'Potasio': 3.83, 'Sodio': 143, 'Magnesio': None, 'ALT': 49, 'AST': 40, 'LDH': 216, 'GGT': None, 'FA': None, 'Bilirrubina': 0.55, 'Albumina': 3.98, 'Proteinas': 7.60, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 14.1, 'Glucosa': 215, 'Creatinina': 0.75, 'Plaquetas': 266, 'Neutrófilos': 11.18, 'Linfocitos': 1.54, 'Calcio': 9.0, 'Potasio': 4.21, 'Sodio': 134, 'Magnesio': None, 'ALT': 150, 'AST': 19, 'LDH': 280, 'GGT': None, 'FA': None, 'Bilirrubina': 0.81, 'Albumina': 3.05, 'Proteinas': 6.00, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    ## Analitica 11:
    {'Hb': 12.8, 'Glucosa': 97, 'Creatinina': 0.91, 'Plaquetas': 320, 'Neutrófilos': 1.38, 'Linfocitos': 1.13, 'Calcio': 10.1, 'Potasio': 4.78, 'Sodio': 142, 'Magnesio': None, 'ALT': 21, 'AST': None, 'LDH': 249, 'GGT': None, 'FA': None, 'Bilirrubina': 0.42, 'Albumina': 4.00, 'Proteinas': 7.30, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 11.0, 'Glucosa': 85, 'Creatinina': 0.68, 'Plaquetas': 296, 'Neutrófilos': 4.22, 'Linfocitos': 1.13, 'Calcio': 9.3, 'Potasio': 4.2, 'Sodio': 140, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 246, 'GGT': None, 'FA': None, 'Bilirrubina': 0.52, 'Albumina': 3.6, 'Proteinas': 7.1, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.0, 'Glucosa': 91, 'Creatinina': 0.64, 'Plaquetas': 275, 'Neutrófilos': 1.66, 'Linfocitos': 1.25, 'Calcio': 9.8, 'Potasio': 4.8, 'Sodio': 142, 'Magnesio': None, 'ALT': None, 'AST': 27, 'LDH': 197, 'GGT': None, 'FA': None, 'Bilirrubina': 0.39, 'Albumina': 3.7, 'Proteinas': 6.7, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 10.4, 'Glucosa': 104, 'Creatinina': 0.86, 'Plaquetas': 213, 'Neutrófilos': 1.52, 'Linfocitos': 1.01, 'Calcio': 9.6, 'Potasio': 5.1, 'Sodio': 141, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 172, 'GGT': None, 'FA': None, 'Bilirrubina': 0.62, 'Albumina': 3.2, 'Proteinas': 7.1, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 8.6, 'Glucosa': 207, 'Creatinina': 0.55, 'Plaquetas': 119, 'Neutrófilos': 2.55, 'Linfocitos': 0.89, 'Calcio': 9.1, 'Potasio': 3.6, 'Sodio': 141, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 207, 'GGT': None, 'FA': None, 'Bilirrubina': 0.52, 'Albumina': 3.2, 'Proteinas': 6.2, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 10.8, 'Glucosa': 126, 'Creatinina': 0.75, 'Plaquetas': 67, 'Neutrófilos': 1.54, 'Linfocitos': 1.83, 'Calcio': 9.6, 'Potasio': 4.0, 'Sodio': 143, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 241, 'GGT': None, 'FA': None, 'Bilirrubina': 0.42, 'Albumina': 4.0, 'Proteinas': 7.1, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 13.1, 'Glucosa': 86, 'Creatinina': 0.68, 'Plaquetas': 327, 'Neutrófilos': 3.58, 'Linfocitos': 1.60, 'Calcio': 9.8, 'Potasio': 3.9, 'Sodio': 141, 'Magnesio': None, 'ALT': 11, 'AST': None, 'LDH': 156, 'GGT': None, 'FA': None, 'Bilirrubina': 0.39, 'Albumina': None, 'Proteinas': 7.3, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 10.8, 'Glucosa': 95, 'Creatinina': 0.74, 'Plaquetas': 209, 'Neutrófilos': 2.88, 'Linfocitos': 1.29, 'Calcio': 9.5, 'Potasio': 4.3, 'Sodio': 139, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 228, 'GGT': None, 'FA': None, 'Bilirrubina': 0.44, 'Albumina': 3.2, 'Proteinas': 7.3, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 10.9, 'Glucosa': 93, 'Creatinina': 0.56, 'Plaquetas': 348, 'Neutrófilos': 4.28, 'Linfocitos': 1.79, 'Calcio': 9.6, 'Potasio': 4.7, 'Sodio': 139, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 266, 'GGT': None, 'FA': None, 'Bilirrubina': 0.45, 'Albumina': 3.6, 'Proteinas': 7.6, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.7, 'Glucosa': 92, 'Creatinina': 0.91, 'Plaquetas': 206, 'Neutrófilos': 1.84, 'Linfocitos': 2.22, 'Calcio': 9.3, 'Potasio': 3.9, 'Sodio': 141, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 189, 'GGT': None, 'FA': None, 'Bilirrubina': 0.57, 'Albumina': 3.8, 'Proteinas': 7.1, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    ## Analitica 21:
    {'Hb': 11.6, 'Glucosa': 101, 'Creatinina': 0.76, 'Plaquetas': 259, 'Neutrófilos': 2.94, 'Linfocitos': 1.02, 'Calcio': 9.5, 'Potasio': 4.13, 'Sodio': 139, 'Magnesio': None, 'ALT': 50, 'AST': 21, 'LDH': 193, 'GGT': None, 'FA': None, 'Bilirrubina': 0.43, 'Albumina': 3.48, 'Proteinas': 6.90, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.4, 'Glucosa': 121, 'Creatinina': 2.18, 'Plaquetas': 293, 'Neutrófilos': 5.28, 'Linfocitos': 1.53, 'Calcio': 6.9, 'Potasio': 3.6, 'Sodio': 137, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 245, 'GGT': None, 'FA': None, 'Bilirrubina': 0.58, 'Albumina': 3.2, 'Proteinas': 6.2, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.2, 'Glucosa': 195, 'Creatinina': 0.97, 'Plaquetas': 380, 'Neutrófilos': 4.13, 'Linfocitos': 2.49, 'Calcio': 9.9, 'Potasio': 3.8, 'Sodio': 136, 'Magnesio': None, 'ALT': None, 'AST': 34, 'LDH': 223, 'GGT': None, 'FA': None, 'Bilirrubina': 0.42, 'Albumina': 4.0, 'Proteinas': 7.8, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 11.6, 'Glucosa': 91, 'Creatinina': 0.65, 'Plaquetas': 113, 'Neutrófilos': 1.31, 'Linfocitos': 0.87, 'Calcio': 9.2, 'Potasio': 3.9, 'Sodio': 139, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 216, 'GGT': None, 'FA': None, 'Bilirrubina': 0.43, 'Albumina': 3.6, 'Proteinas': 7.0, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.4, 'Glucosa': 91, 'Creatinina': 0.77, 'Plaquetas': 174, 'Neutrófilos': 1.50, 'Linfocitos': 0.51, 'Calcio': 9.3, 'Potasio': 4.8, 'Sodio': 141, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 226, 'GGT': None, 'FA': None, 'Bilirrubina': 0.48, 'Albumina': 4.0, 'Proteinas': 6.6, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 9.9, 'Glucosa': 91, 'Creatinina': 0.77, 'Plaquetas': 300, 'Neutrófilos': 2.06, 'Linfocitos': 1.79, 'Calcio': 9.1, 'Potasio': 3.8, 'Sodio': 142, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 207, 'GGT': None, 'FA': None, 'Bilirrubina': 0.33, 'Albumina': 3.0, 'Proteinas': 6.1, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 14.0, 'Glucosa': 95, 'Creatinina': 0.67, 'Plaquetas': 189, 'Neutrófilos': 2.81, 'Linfocitos': 1.58, 'Calcio': 9.0, 'Potasio': 4.5, 'Sodio': 142, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 225, 'GGT': None, 'FA': None, 'Bilirrubina': 0.72, 'Albumina': 3.7, 'Proteinas': 6.7, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.1, 'Glucosa': 98, 'Creatinina': 0.84, 'Plaquetas': 287, 'Neutrófilos': 0.56, 'Linfocitos': 2.15, 'Calcio': 9.1, 'Potasio': 4.7, 'Sodio': 143, 'Magnesio': None, 'ALT': None, 'AST': None, 'LDH': 162, 'GGT': None, 'FA': None, 'Bilirrubina': 0.35, 'Albumina': 3.3, 'Proteinas': 6.2, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 13.5, 'Glucosa': 104, 'Creatinina': 0.61, 'Plaquetas': 257, 'Neutrófilos': 2.51, 'Linfocitos': 2.12, 'Calcio': 10.5, 'Potasio': 4.2, 'Sodio': 137, 'Magnesio': None, 'ALT': 38, 'AST': None, 'LDH': 205, 'GGT': 26, 'FA': 133, 'Bilirrubina': 0.57, 'Albumina': 3.9, 'Proteinas': 7.0, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 205, 'HDL': None, 'Trigliceridos': 71, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 10.9, 'Glucosa': 223, 'Creatinina': 0.98, 'Plaquetas': 412, 'Neutrófilos': 10.45, 'Linfocitos': 3.10, 'Calcio': 9.3, 'Potasio': 5.4, 'Sodio': 143, 'Magnesio': None, 'ALT': 16, 'AST': None, 'LDH': 194, 'GGT': 29, 'FA': 154, 'Bilirrubina': 0.22, 'Albumina': 3.6, 'Proteinas': 6.9, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 161, 'HDL': None, 'Trigliceridos': 107, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    ## Analitica 31:
    {'Hb': 12.1, 'Glucosa': 96, 'Creatinina': 0.85, 'Plaquetas': 197, 'Neutrófilos': 2.46, 'Linfocitos': 2.15, 'Calcio': 9.5, 'Potasio': 5.30, 'Sodio': 144, 'Magnesio': None, 'ALT': 35, 'AST': None, 'LDH': 171, 'GGT': 14, 'FA': 73, 'Bilirrubina': 0.45, 'Albumina': 3.83, 'Proteinas': 6.90, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 138, 'HDL': None, 'Trigliceridos': 119, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 13.3, 'Glucosa': 123, 'Creatinina': 0.77, 'Plaquetas': 261, 'Neutrófilos': 3.36, 'Linfocitos': 2.01, 'Calcio': None, 'Potasio': 4.59, 'Sodio': 143, 'Magnesio': None, 'ALT': 21, 'AST': 22, 'LDH': None, 'GGT': 30, 'FA': 74, 'Bilirrubina': None, 'Albumina': 3.46, 'Proteinas': None, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 161, 'HDL': 42, 'Trigliceridos': 106, 'Creatinquinasa': 95, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': 5.5},
    {'Hb': 13.5, 'Glucosa': 107, 'Creatinina': 0.68, 'Plaquetas': 224, 'Neutrófilos': 2.91, 'Linfocitos': 1.23, 'Calcio': 9.2, 'Potasio': 4.47, 'Sodio': 140, 'Magnesio': None, 'ALT': 69, 'AST': 34, 'LDH': 178, 'GGT': None, 'FA': None, 'Bilirrubina': 0.59, 'Albumina': 4.25, 'Proteinas': 6.80, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.6, 'Glucosa': 87, 'Creatinina': 0.92, 'Plaquetas': 262, 'Neutrófilos': 4.91, 'Linfocitos': 1.61, 'Calcio': 9.6, 'Potasio': 4.52, 'Sodio': 140, 'Magnesio': None, 'ALT': 31, 'AST': None, 'LDH': 170, 'GGT': 22, 'FA': 71, 'Bilirrubina': 0.54, 'Albumina': 3.88, 'Proteinas': 6.75, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 187, 'HDL': None, 'Trigliceridos': 135, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 13.0, 'Glucosa': 88, 'Creatinina': 0.78, 'Plaquetas': 207, 'Neutrófilos': 2.54, 'Linfocitos': 2.42, 'Calcio': 9.7, 'Potasio': 4.83, 'Sodio': 142, 'Magnesio': None, 'ALT': 32, 'AST': None, 'LDH': 194, 'GGT': 18, 'FA': 82, 'Bilirrubina': 0.85, 'Albumina': 3.76, 'Proteinas': 7.80, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 213, 'HDL': None, 'Trigliceridos': 61, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 11.6, 'Glucosa': 126, 'Creatinina': 0.84, 'Plaquetas': 164, 'Neutrófilos': 3.52, 'Linfocitos': 0.79, 'Calcio': 9.3, 'Potasio': 4.99, 'Sodio': 139, 'Magnesio': None, 'ALT': 13, 'AST': None, 'LDH': 154, 'GGT': None, 'FA': None, 'Bilirrubina': 0.86, 'Albumina': 3.68, 'Proteinas': 7.30, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 10.1, 'Glucosa': 94, 'Creatinina': 0.92, 'Plaquetas': 170, 'Neutrófilos': 0.82, 'Linfocitos': 1.30, 'Calcio': 9.3, 'Potasio': 5.34, 'Sodio': 139, 'Magnesio': None, 'ALT': 16, 'AST': None, 'LDH': 195, 'GGT': None, 'FA': None, 'Bilirrubina': 0.34, 'Albumina': 3.59, 'Proteinas': 7.50, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 11.8, 'Glucosa': 80, 'Creatinina': 1.05, 'Plaquetas': 237, 'Neutrófilos': 1.50, 'Linfocitos': 1.04, 'Calcio': 9.8, 'Potasio': 4.99, 'Sodio': 139, 'Magnesio': None, 'ALT': 11, 'AST': None, 'LDH': 311, 'GGT': None, 'FA': None, 'Bilirrubina': 0.40, 'Albumina': 3.42, 'Proteinas': 7.70, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 13.8, 'Glucosa': 83, 'Creatinina': 0.83, 'Plaquetas': 154, 'Neutrófilos': 2.07, 'Linfocitos': 1.90, 'Calcio': 8.7, 'Potasio': 4.30, 'Sodio': 144, 'Magnesio': None, 'ALT': 20, 'AST': None, 'LDH': 181, 'GGT': None, 'FA': None, 'Bilirrubina': 0.70, 'Albumina': 3.50, 'Proteinas': 6.30, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.5, 'Glucosa': 78, 'Creatinina': 0.80, 'Plaquetas': 242, 'Neutrófilos': 2.34, 'Linfocitos': 1.51, 'Calcio': 9.9, 'Potasio': 4.91, 'Sodio': 139, 'Magnesio': None, 'ALT': 14, 'AST': None, 'LDH': 196, 'GGT': None, 'FA': None, 'Bilirrubina': 0.46, 'Albumina': 3.59, 'Proteinas': 7.00, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    ## Analitica 41:
    {'Hb': 13.2, 'Glucosa': 80, 'Creatinina': 0.83, 'Plaquetas': 180, 'Neutrófilos': 4.26, 'Linfocitos': 0.78, 'Calcio': 9.5, 'Potasio': 4.07, 'Sodio': 141, 'Magnesio': None, 'ALT': 9, 'AST': None, 'LDH': 166, 'GGT': None, 'FA': None, 'Bilirrubina': 0.69, 'Albumina': 3.94, 'Proteinas': 6.70, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.0, 'Glucosa': 86, 'Creatinina': 0.88, 'Plaquetas': 240, 'Neutrófilos': 0.97, 'Linfocitos': 1.57, 'Calcio': 10.1, 'Potasio': 4.58, 'Sodio': 144, 'Magnesio': None, 'ALT': 34, 'AST': None, 'LDH': 175, 'GGT': None, 'FA': None, 'Bilirrubina': 0.21, 'Albumina': 3.84, 'Proteinas': 7.30, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 12.4, 'Glucosa': 110, 'Creatinina': 0.69, 'Plaquetas': 225, 'Neutrófilos': 1.16, 'Linfocitos': 1.62, 'Calcio': 9.4, 'Potasio': 4.21, 'Sodio': 141, 'Magnesio': None, 'ALT': 24, 'AST': None, 'LDH': 159, 'GGT': 23, 'FA': 72, 'Bilirrubina': 0.38, 'Albumina': 3.76, 'Proteinas': 7.14, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 172, 'HDL': None, 'Trigliceridos': 123, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 13.5, 'Glucosa': 89, 'Creatinina': 0.63, 'Plaquetas': 352, 'Neutrófilos': 2.99, 'Linfocitos': 2.68, 'Calcio': 9.5, 'Potasio': 4.94, 'Sodio': 143, 'Magnesio': None, 'ALT': 21, 'AST': 20, 'LDH': 178, 'GGT': 18, 'FA': 117, 'Bilirrubina': 0.48, 'Albumina': 3.44, 'Proteinas': 6.60, 'Lipasa': None, 'Amilasa': None, 'Colesterol': 193, 'HDL': None, 'Trigliceridos': 106, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None},
    {'Hb': 11.9, 'Glucosa': 106, 'Creatinina': 0.88, 'Plaquetas': 158, 'Neutrófilos': 1.37, 'Linfocitos': 0.90, 'Calcio': 9.4, 'Potasio': 4.00, 'Sodio': 142, 'Magnesio': None, 'ALT': 14, 'AST': None, 'LDH': 191, 'GGT': None, 'FA': None, 'Bilirrubina': 0.27, 'Albumina': 4.00, 'Proteinas': 7.30, 'Lipasa': None, 'Amilasa': None, 'Colesterol': None, 'HDL': None, 'Trigliceridos': None, 'Creatinquinasa': None, 'TTPA': None, 'Fibrinogeno': None, 'INR': None, 'PH': None}
]

def generar_predicciones_regex(df_textos, columns):
    """Corre el MedicalExtractor sobre todos los textos y los formatea para hacer compatible con el df de Gemini."""
    extractor = MedicalExtractor()
    predicciones = []

    for index, row in df_textos.iterrows():
        texto = row['texto']
        resultados_regex = extractor.analizar_texto(texto)
        
        fila_prediccion = {}
        for col in columns:
            if col in resultados_regex:
                fila_prediccion[col] = resultados_regex[col]['valor']
            else:
                fila_prediccion[col] = None
                
        predicciones.append(fila_prediccion)
        
    return pd.DataFrame(predicciones)


def evaluar_extraccion(reg, gem):
    if pd.isna(reg) and pd.isna(gem):
        return "VN"
    if pd.isna(reg) and pd.notna(gem):
        return "FN"
    if pd.notna(reg) and pd.isna(gem):
        return "FP"
    
    if reg == gem:
        return "VP"
    else:
        return "Erróneo"

def evaluar_lote(df_textos, lista_gold, nombre_lote):
    """Encapsula toda la lógica de evaluación y generación de gráficos por lote."""
    if df_textos.empty:
        print(f"[{nombre_lote}] Parando ejecución porque no se encuentran textos.")
        return
        
    df_gold = pd.DataFrame(lista_gold)
    
    print(f"[{nombre_lote}] Corriendo el motor regex sobre las analíticas...")
    df_regex = generar_predicciones_regex(df_textos, df_gold.columns)
    
    print(f"[{nombre_lote}] Evaluando las predicciones...")
    df_comparacion = pd.DataFrame(index=df_gold.index, columns=df_gold.columns)

    for col in df_gold.columns:
        df_comparacion[col] = [evaluar_extraccion(r, g) for r, g in zip(df_regex[col], df_gold[col])]

    # Construir matriz de confusión
    valores = df_comparacion.values.flatten()
    vp = np.sum(valores == "VP")
    vn = np.sum(valores == "VN")
    fn = np.sum(valores == "FN")
    fp = np.sum(valores == "FP") + np.sum(valores == "Erróneo")

    cm = [[vn, fp], 
          [fn, vp]]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ausente (Pred)', 'Presente (Pred)'],
                yticklabels=['Ausente (Real)', 'Presente (Real)'])

    plt.title(f'Regex vs Gemini (NER) - {nombre_lote.upper()}')
    plt.xlabel('Predicción (Regex)')
    plt.ylabel('Realidad (Gemini)')
    
    os.makedirs('output', exist_ok=True)
    ruta_grafico = f'output/mc_ner_regex_{nombre_lote}.png'
    plt.savefig(ruta_grafico, dpi=300)
    plt.close() # Cierra la figura para que no se solape con la del siguiente lote
    
    df_cm = pd.DataFrame(cm, index=['Ausente (Real)', 'Presente (Real)'], columns=['Ausente (Pred)', 'Presente (Pred)'])
    ruta_cm_csv = f'output/mc_ner_regex_{nombre_lote}_data.csv'
    df_cm.to_csv(ruta_cm_csv)
    print(f"[{nombre_lote}] Matriz guardada en '{ruta_grafico}' y datos en '{ruta_cm_csv}'")

    # Calcular métricas
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metricas_ner = {
        "lote": nombre_lote,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "total_entidades": int(len(valores))
    }

    ruta_json = f'output/metricas_ner_{nombre_lote}.json'
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(metricas_ner, f, indent=4)
    print(f"[{nombre_lote}] Métricas guardadas en '{ruta_json}'")

    # Extraer casos donde falló el motor regex (Falsos Negativos)
    casos_fn = []
    for col in df_gold.columns:
        for idx in df_comparacion.index:
            if df_comparacion.loc[idx, col] == "FN":
                casos_fn.append({
                    'id_analitica': df_textos.iloc[idx]['id'] if 'id' in df_textos.columns else idx,
                    'marcador': col,
                    'valor_gemini': df_gold.loc[idx, col]
                })
    
    if casos_fn:
        df_fallos = pd.DataFrame(casos_fn)
        ruta_fallos = f'output/fallos_regex_fn_{nombre_lote}.csv'
        df_fallos.to_csv(ruta_fallos, index=False, encoding='utf-8')
        print(f"[{nombre_lote}] Casos donde Regex falló (FN) guardados en '{ruta_fallos}'")

    print(f"\n--- MÉTRICAS GLOBALES DEL MOTOR REGEX ({nombre_lote.upper()}) ---")
    print(f"Total entidades evaluadas: {len(valores)}")
    print(f"- Precisión: {precision:.4f}")
    print(f"- Recall:    {recall:.4f}")
    print(f"- F1-Score:  {f1:.4f}\n")


if __name__ == "__main__":
    
    # ------------------ LOTE 1 ------------------
    print(">>> INICIANDO PROCESO PARA EL LOTE 1 <<<")
    df_textos_lote1 = cargar_analiticas('analiticas/lote1')
    evaluar_lote(df_textos_lote1, df_lote1, "lote1")

    # ------------------ LOTE 2 ------------------
    print(">>> INICIANDO PROCESO PARA EL LOTE 2 <<<")
    df_textos_lote2 = cargar_analiticas('analiticas/lote2')
    evaluar_lote(df_textos_lote2, df_lote2, "lote2")