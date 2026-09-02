import json
import pandas as pd
import numpy as np
from pathlib import Path

def parse_pdqi9_metrics(pdqi9_data):
    metrics = [
        "precisa", "exhaustiva", "util", "organizada", 
        "comprensible", "concisa", "sintetizada", "consistencia_interna"
    ]
    
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

def analyze_pdqi9_lote2():
    root_dir = Path(__file__).parent.parent.parent
    
    # Datos de Qwen (Lote 2)
    qwen_gemini = get_json_eval_data(root_dir / "output" / "evaluacion_qwen_lote2_gemini.json", "qwen2.5-7B", "id_analitica")
    
    # Datos de Biomistral (Lote 2)
    biomistral_gemini = get_json_eval_data(root_dir / "output" / "evaluacion_biomistral_lote2_gemini.json", "biomistral-7B", "pregunta_id")
    
    all_data = qwen_gemini + biomistral_gemini
    
    if not all_data:
        print("No se encontraron datos para analizar en lote 2.")
        return None
        
    df = pd.DataFrame(all_data)
    
    output_str = []
    output_str.append("======================================================================")
    output_str.append("ANÁLISIS DE EVALUACIÓN PDQI-9 EMPÍRICA (Múltiples Estrategias) - LOTE 2")
    output_str.append("======================================================================")
    output_str.append(f"Total de evaluaciones procesadas: {len(df)}\n")
    
    grouped = df.groupby(["model", "estrategia"])
    metrics_to_print = ["average_score", "precisa", "exhaustiva", "util", "consistencia_interna"]
    
    for model in sorted(df["model"].unique()):
        for strategy in ["Zero-Shot", "Few-Shot", "CoT", "CoT+FS"]:
            try:
                group = grouped.get_group((model, strategy))
                output_str.append("-" * 50)
                output_str.append(f"Modelo: {model} | Estrategia: {strategy} (n={len(group)})")
                output_str.append("-" * 50)
                
                output_str.append("[+] Puntuaciones (Media ± Std):")
                for m in metrics_to_print:
                    output_str.append(f"  - {m.capitalize():<22}: {group[m].mean():.2f} ± {group[m].std():.2f}")
                
                err_count = group['error_seguridad_grave'].sum()
                contra_count = group['contradicciones_clinicas_insalvables'].sum()
                hum_count = group['human_generated'].sum()
                
                output_str.append("\n[+] Calidad y Seguridad:")
                output_str.append(f"  - Error Seguridad Grave : {err_count} ({(err_count/len(group))*100:.1f}%)")
                output_str.append(f"  - Contradicc. Insalvable: {contra_count} ({(contra_count/len(group))*100:.1f}%)")
                output_str.append(f"  - Pasa test humano      : {hum_count} ({(hum_count/len(group))*100:.1f}%)\n")
                
            except KeyError:
                pass
                
    output_str.append("======================================================================\n")
    
    full_output = "\n".join(output_str)
    print(full_output)
    
    # Exportar a CSV
    csv_path = root_dir / "output" / "evaluacion_pdqi9_completa_lote2.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset exportado a {csv_path}")
    
    # Exportar a Markdown
    md_path = root_dir / "output" / "pdqi9_lote2.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_output)
    print(f"Archivo generado en {md_path}")
    
    return df

if __name__ == "__main__":
    analyze_pdqi9_lote2()
