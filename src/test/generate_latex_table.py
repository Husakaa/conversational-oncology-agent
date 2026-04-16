import pandas as pd
import numpy as np
import json
import os

def cargar_json(ruta):
    if os.path.exists(ruta):
        with open(ruta, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def generar_tabla():
    # 1. Cargar métricas NER (Regex)
    ner = cargar_json('output/metricas_ner.json')
    p_ner = ner['precision'] if ner else 0.0
    r_ner = ner['recall'] if ner else 0.0
    f_ner = ner['f1'] if ner else 0.0

    # 2. Cargar latencias de SLM (CSV)
    csv_path = 'output/latencias_secuenciales.csv'
    q_mean, q_p95, b_mean, b_p95 = 0, 0, 0, 0
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        q_mean = df['latencia_qwen'].mean()
        q_p95 = np.percentile(df['latencia_qwen'], 95)
        b_mean = df['latencia_bio'].mean()
        b_p95 = np.percentile(df['latencia_bio'], 95)

    # 3. Cargar Calidad (JSON de la batería)
    bateria = cargar_json('output/bateria_resultados.json')
    q_qual, b_qual = 0.0, 0.0
    if bateria:
        notas_q = [d['puntuacion_qwen'] for d in bateria if d.get('puntuacion_qwen') is not None]
        notas_b = [d['puntuacion_bio'] for d in bateria if d.get('puntuacion_bio') is not None]
        if notas_q: q_qual = np.mean(notas_q)
        if notas_b: b_qual = np.mean(notas_b)

    # 4. Construcción de la tabla LaTeX
    latex_table = r"""
\begin{tabular}{@{}lllc@{}}
\toprule
\textbf{Fase del Pipeline} & \textbf{Tecnología} & \textbf{Métrica de Evaluación} & \textbf{Resultado} \\ \midrule

% BLOQUE 1: EXTRACCIÓN
\multirow{3}{*}{1. Extracción} & \multirow{3}{*}{Motor Regex} & Precisión Global & """ + f"{p_ner:.4f}" + r""" \\
 & & Recall Global & """ + f"{r_ner:.4f}" + r""" \\
 & & F1-Score & """ + f"{f_ner:.4f}" + r""" \\ \midrule

% BLOQUE 2: SÍNTESIS
\multirow{3}{*}{2. Síntesis Clínica} & \multirow{3}{*}{Qwen 2.5 (7B)} & Latencia Media & """ + f"{q_mean:.2f} s" + r""" \\
 & & Latencia P95 & """ + f"{q_p95:.2f} s" + r""" \\
 & & Calidad Media & """ + f"{q_qual:.1f} / 5.0" + r""" \\ \midrule

% FASE CONSULTA
\multirow{3}{*}{3. Consulta} & \multirow{3}{*}{BioMistral (7B)} & Latencia Media & """ + f"{b_mean:.2f} s" + r""" \\
 & & Latencia P95 & """ + f"{b_p95:.2f} s" + r""" \\
 & & Calidad Media & """ + f"{b_qual:.1f} / 5.0" + r""" \\ \bottomrule
\end{tabular}
"""

    with open('output/tabla_resumen.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"Tabla generada con éxito usando {len(bateria) if bateria else 0} casos de prueba.")

if __name__ == "__main__":
    generar_tabla()