import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def parse_pdqi9_metrics(pdqi9_data):
    metrics = [
        "precisa", "exhaustiva", "util", "organizada", 
        "comprensible", "concisa", "sintetizada", "consistencia_interna"
    ]
    # En biomistral usamos "enfocada" conceptualmente pero mantuvimos la key "sintetizada" en JSON
    
    row = {}
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
    return row

def get_qwen_cot_fs_data(base_dir):
    data = []
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return data
        
    for item in base_dir.iterdir():
        if item.is_dir():
            json_file = item / f"{item.name}.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as jf:
                        pdqi9_data = json.load(jf)
                    row = parse_pdqi9_metrics(pdqi9_data)
                    row["model"] = "qwen2.5-7B"
                    row["estrategia"] = "CoT+FS"
                    row["id"] = item.name
                    data.append(row)
                except Exception as e:
                    print(f"Error parseando {json_file}: {e}")
    return data

def get_json_eval_data(filepath, model_name, id_key):
    data = []
    path = Path(filepath)
    if not path.exists():
        return data
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            records = json.load(f)
            
        for rec in records:
            row = parse_pdqi9_metrics(rec)
            row["model"] = model_name
            row["estrategia"] = rec.get("estrategia", "Unknown")
            row["id"] = str(rec.get(id_key, "Unknown"))
            data.append(row)
    except Exception as e:
        print(f"Error parseando {filepath}: {e}")
    return data

def analyze_pdqi9():
    root_dir = Path(__file__).parent.parent.parent
    
    # 1. Datos antiguos de Qwen CoT+FS
    qwen_cot_fs = get_qwen_cot_fs_data(root_dir / "analiticas" / "lote1" / "Processed")
    
    # 2. Datos de Qwen (ZS, FS, CoT)
    qwen_gemini = get_json_eval_data(root_dir / "output" / "evaluacion_qwen_gemini.json", "qwen2.5-7B", "id_analitica")
    
    # 3. Datos de Biomistral (ZS, FS, CoT, CoT+FS)
    biomistral_gemini = get_json_eval_data(root_dir / "output" / "evaluacion_biomistral_gemini.json", "biomistral-7B", "pregunta_id")
    
    # Combinar todos
    all_data = qwen_cot_fs + qwen_gemini + biomistral_gemini
    
    if not all_data:
        print("No se encontraron datos para analizar.")
        return None
        
    df = pd.DataFrame(all_data)
    
    print("\n" + "="*70)
    print("ANÁLISIS DE EVALUACIÓN PDQI-9 EMPÍRICA (Múltiples Estrategias)")
    print("="*70)
    print(f"Total de evaluaciones procesadas: {len(df)}")
    
    # Agrupar por modelo y estrategia
    grouped = df.groupby(["model", "estrategia"])
    
    metrics_to_print = ["average_score", "precisa", "exhaustiva", "util", "consistencia_interna"]
    
    for (model, strategy), group in grouped:
        print("\n" + "-"*50)
        print(f"Modelo: {model} | Estrategia: {strategy} (n={len(group)})")
        print("-" * 50)
        
        # Puntuaciones medias
        print("[+] Puntuaciones (Media ± Std):")
        for m in metrics_to_print:
            if m in group.columns:
                print(f"  - {m.capitalize():<22}: {group[m].mean():.2f} ± {group[m].std():.2f}")
                
        # Booleanas
        errores_graves = group["error_seguridad_grave"].sum()
        contradicciones = group["contradicciones_clinicas_insalvables"].sum()
        humanos = group["human_generated"].sum()
        
        print("\n[+] Calidad y Seguridad:")
        print(f"  - Error Seguridad Grave : {errores_graves} ({(errores_graves/len(group))*100:.1f}%)")
        print(f"  - Contradicc. Insalvable: {contradicciones} ({(contradicciones/len(group))*100:.1f}%)")
        print(f"  - Pasa test humano      : {humanos} ({(humanos/len(group))*100:.1f}%)")
        
    print("="*70 + "\n")
    
    # Guardar dataframe final
    output_path = root_dir / "output" / "evaluacion_pdqi9_completa.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset completo exportado a {output_path}")
    
    return df

if __name__ == "__main__":
    analyze_pdqi9()
