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
    """Carga todos los archivos txt del directorio a un dataframe"""
    datos = []
    if not os.path.exists(directorio):
        print(f"Directorio '{directorio}' no encontrado.")
        return pd.DataFrame()

    archivos = os.listdir(directorio)
    for nombre_archivo in archivos:
        ruta_completa = os.path.join(directorio, nombre_archivo)
        
        if nombre_archivo.endswith(".txt") and os.path.isfile(ruta_completa):
            with open(ruta_completa, 'r', encoding='utf-8') as archivo:
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
df_gemini = [
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


def generar_predicciones_regex(df_textos, columns):
    """Corre el MedicalExtractor sobre todos los textos y los formatea para hacer compatible con el df de Gemini."""
    print("Corriendo el motor regex sobre las analíticas...")
    extractor = MedicalExtractor()
    predicciones = []

    for index, row in df_textos.iterrows():
        texto = row['texto']
        # Extraer biomarcadores con el motor regex
        resultados_regex = extractor.analizar_texto(texto)
        
        # Formatear para hacer compatible con el df gemini
        fila_prediccion = {}
        for col in columns:
            if col in resultados_regex:
                # Extraemos solo el valor numérico
                fila_prediccion[col] = resultados_regex[col]['valor']
            else:
                fila_prediccion[col] = None
                
        predicciones.append(fila_prediccion)
        
    return pd.DataFrame(predicciones)


def evaluar_extraccion(reg, gem):
    # Caso 1: Ambos son nulos o None
    if pd.isna(reg) and pd.isna(gem):
        return "VN"
    
    # Caso 2: Regex no encontró nada, pero Gemini sí
    if pd.isna(reg) and pd.notna(gem):
        return "FN"
    
    # Caso 3: Regex encontró algo, pero Gemini no 
    if pd.notna(reg) and pd.isna(gem):
        return "FP"
    
    # Caso 4: Ambos tienen valores
    if reg == gem:
        return "VP" # Coincidencia exacta
    else:
        return "Erróneo"  # Valores distintos
 

if __name__ == "__main__":
    # Cargar textos
    df_textos = cargar_analiticas()
    
    if df_textos.empty:
        print("Parando ejecución porque no se encuentran textos.")
    else:
        df_gemini = pd.DataFrame(df_gemini)
        
        # Hacer las predicciones
        df_regex = generar_predicciones_regex(df_textos, df_gemini.columns)
        
        # Crear el df comparativo
        print("Evaluando las predicciones sobre el conjunto de test...")
        df_comparacion = pd.DataFrame(index=df_gemini.index, columns=df_gemini.columns)

        for col in df_gemini.columns:
            df_comparacion[col] = [evaluar_extraccion(r, g) for r, g in zip(df_regex[col], df_gemini[col])]

        # Construir matriz de confusión
        valores = df_comparacion.values.flatten()
        vp = np.sum(valores == "VP")
        vn = np.sum(valores == "VN")
        fn = np.sum(valores == "FN")
        fp = np.sum(valores == "FP") + np.sum(valores == "Erróneo_FP")

        cm = [[vn, fp], 
              [fn, vp]]

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ausente (Pred)', 'Presente (Pred)'],
                    yticklabels=['Ausente (Real)', 'Presente (Real)'])

        plt.title('Regex vs Gemini (NER)')
        plt.xlabel('Predicción (Regex)')
        plt.ylabel('Realidad (Gemini)')
        
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/mc_ner_regex.png', dpi=300)
        print("Matriz guardada en 'output/mc_ner_regex.png'")

        # Calcular métricas de evaluación
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0
        recall = vp / (vp + fn) if (vp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metricas_ner = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "total_entidades": int(len(valores))
        }

        with open('output/metricas_ner.json', 'w', encoding='utf-8') as f:
            json.dump(metricas_ner, f, indent=4)
        print("Métricas NER guardadas en 'output/metricas_ner.json'")

        print("\nMÉTRICAS GLOBALES DEL MOTOR REGEX")
        print(f"Total entidades evaluadas: {len(valores)}")
        print(f"- Precisión: {precision:.4f} (Fiabilidad de lo extraído)")
        print(f"- Recall:    {recall:.4f} (Capacidad de encontrar todo)")
        print(f"- F1-Score:  {f1:.4f} (Equilibrio general)")