import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_pdqi9(base_dir=None):
    if base_dir is None:
        # Asume que el script está en TFG/test/
        root_dir = Path(__file__).parent.parent.parent
        base_dir = root_dir / "analiticas" / "lote1" / "Processed"
    else:
        base_dir = Path(base_dir)
        
    data = []
    
    metrics = [
        "precisa", "exhaustiva", "util", "organizada", 
        "comprensible", "concisa", "sintetizada", "consistencia_interna"
    ]
    
    if not base_dir.exists():
        print(f"Error: El directorio {base_dir} no existe.")
        return None
        
    for item in base_dir.iterdir():
        if item.is_dir():
            json_file = item / f"{item.name}.json"
            txt_file = item / f"NG_{item.name}.txt"
            
            if json_file.exists() and txt_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as jf:
                        pdqi9_data = json.load(jf)
                        
                    with open(txt_file, 'r', encoding='utf-8') as tf:
                        text_content = tf.read()
                        
                    # Aproximación de longitud en tokens (por palabras/espacios)
                    tokens = len(text_content.split())
                    
                    row = {"id": item.name, "tokens": tokens}
                    total_score = 0
                    valid_metrics = 0
                    
                    for m in metrics:
                        if m in pdqi9_data and "score" in pdqi9_data[m]:
                            score = pdqi9_data[m]["score"]
                            row[m] = score
                            total_score += score
                            valid_metrics += 1
                            
                    if valid_metrics > 0:
                        row["average_score"] = total_score / valid_metrics
                    else:
                        row["average_score"] = np.nan
                        
                    row["error_seguridad_grave"] = pdqi9_data.get("error_seguridad_grave", False)
                    row["contradicciones_clinicas_insalvables"] = pdqi9_data.get("contradicciones_clinicas_insalvables", False)
                    row["human_generated"] = pdqi9_data.get("human_generated", False)
                    
                    data.append(row)
                except Exception as e:
                    print(f"Error procesando {item.name}: {e}")
                    
    if not data:
        print("No se encontraron datos válidos.")
        return None
        
    df = pd.DataFrame(data)
    
    print("\n" + "="*50)
    print("ANÁLISIS DE EVALUACIÓN PDQI-9 (LOTE 1)")
    print("="*50)
    print(f"Total de analíticas analizadas: {len(df)}")
    
    print("\n[+] Puntuación Promedio por Métrica")
    print("-" * 35)
    for m in metrics:
        if m in df.columns:
            print(f"  - {m.capitalize():<22}: {df[m].mean():.2f} (std: {df[m].std():.2f})")
            
    print("\n[+] Estadísticas Generales")
    print("-" * 35)
    print(f"  - Puntuación media global: {df['average_score'].mean():.2f} (std: {df['average_score'].std():.2f})")
    print(f"  - Puntuación máxima obtenida: {df['average_score'].max():.2f} (Documento ID: {df.loc[df['average_score'].idxmax(), 'id']})")
    print(f"  - Puntuación mínima obtenida: {df['average_score'].min():.2f} (Documento ID: {df.loc[df['average_score'].idxmin(), 'id']})")
    print(f"  - Longitud media (tokens aprox): {df['tokens'].mean():.1f} (std: {df['tokens'].std():.1f})")
    
    print("\n[+] Cruce: Longitud de la Nota vs Puntuación")
    print("-" * 35)
    correlation = df['tokens'].corr(df['average_score'])
    print(f"  - Correlación de Pearson: {correlation:.3f}")
    
    interpretacion = ""
    if correlation > 0.5:
        interpretacion = "Fuerte correlación positiva (notas más largas tienden a tener mejor puntuación)."
    elif correlation < -0.5:
        interpretacion = "Fuerte correlación negativa (notas más largas tienden a tener peor puntuación)."
    elif correlation > 0.3:
        interpretacion = "Correlación positiva moderada."
    elif correlation < -0.3:
        interpretacion = "Correlación negativa moderada."
    else:
        interpretacion = "No hay correlación lineal significativa."
    print(f"  - Interpretación: {interpretacion}")
    
    print("\n[+] Métricas de Seguridad y Calidad")
    print("-" * 35)
    errores_graves = df["error_seguridad_grave"].sum()
    contradicciones = df["contradicciones_clinicas_insalvables"].sum()
    humanos = df["human_generated"].sum()
    
    print(f"  - Notas con error de seguridad grave: {errores_graves} ({(errores_graves/len(df))*100:.1f}%)")
    print(f"  - Notas con contradicciones insalvables: {contradicciones} ({(contradicciones/len(df))*100:.1f}%)")
    print(f"  - Notas marcadas como 'human_generated': {humanos} ({(humanos/len(df))*100:.1f}%)")
    
    print("="*50 + "\n")
    
    return df

if __name__ == "__main__":
    analyze_pdqi9()
