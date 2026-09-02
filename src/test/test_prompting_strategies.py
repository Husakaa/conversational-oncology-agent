import requests
import time
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.test.evaluate_ner_regex import cargar_analiticas
from src.test.evaluate_slm import PREGUNTAS_BIOMISTRAL, PREGUNTAS_BIOMISTRAL_LOTE2, forzar_descarga_vram

URL_SYNTHESIZE = "http://127.0.0.1:8000/api/v1/synthesize"
URL_CONSULT = "http://127.0.0.1:8000/api/v1/consult"

ESTRATEGIAS_QWEN = ["Zero-Shot", "Few-Shot", "CoT"]
ESTRATEGIAS_BIOMISTRAL = ["Zero-Shot", "Few-Shot", "CoT", "CoT+FS"]

def probar_estrategias_sintesis():
    print(">>> Iniciando pruebas de estrategias de prompting para Síntesis (Qwen2.5-7B) <<<")
    df_analiticas = cargar_analiticas('analiticas/lote1')
    if df_analiticas.empty:
        print("No se encontraron analíticas en lote 1.")
        return

    resultados = []
    
    for _, row in df_analiticas.iterrows():
        id_analitica = row['id']
        texto = row['texto']
        
        for estrategia in ESTRATEGIAS_QWEN:
            print(f"[{id_analitica}] Evaluando con {estrategia}...", end=" ", flush=True)
            
            payload = {
                "texto_clinico": texto,
                "metodologia_prompt": estrategia
            }
            
            start_time = time.time()
            try:
                res = requests.post(URL_SYNTHESIZE, json=payload, timeout=120)
                if res.ok:
                    latencia = time.time() - start_time
                    sintesis = res.json().get("sintesis_clinica", "ERROR")
                    print(f"OK ({latencia:.2f}s)")
                else:
                    latencia = -1
                    sintesis = f"HTTP ERROR: {res.status_code}"
                    print("ERROR HTTP")
            except Exception as e:
                latencia = -1
                sintesis = f"EXCEPTION: {str(e)}"
                print(f"ERROR EXCEPTION: {e}")
                
            resultados.append({
                "id_analitica": id_analitica,
                "estrategia": estrategia,
                "latencia_s": latencia,
                "sintesis": sintesis
            })
            
    # Guardar resultados
    os.makedirs("output", exist_ok=True)
    df_res = pd.DataFrame(resultados)
    df_res.to_csv("output/qwen_prompting_lote1.csv", index=False)
    print("Guardado en output/qwen_prompting_lote1.csv\n")
    
    # Liberar VRAM
    forzar_descarga_vram("qwen2.5-7B")

def probar_estrategias_consultas():
    print(">>> Iniciando pruebas de estrategias de prompting para Consultas (Biomistral-7B) <<<")
    
    resultados = []
    
    for i, pregunta in enumerate(PREGUNTAS_BIOMISTRAL, 1):
        for estrategia in ESTRATEGIAS_BIOMISTRAL:
            print(f"[Pregunta {i}] Evaluando con {estrategia}...", end=" ", flush=True)
            
            payload = {
                "consulta": pregunta,
                "metodologia_prompt": estrategia
            }
            
            start_time = time.time()
            try:
                res = requests.post(URL_CONSULT, json=payload, timeout=120)
                if res.ok:
                    latencia = time.time() - start_time
                    respuesta = res.json().get("respuesta", "ERROR")
                    print(f"OK ({latencia:.2f}s)")
                else:
                    latencia = -1
                    respuesta = f"HTTP ERROR: {res.status_code}"
                    print("ERROR HTTP")
            except Exception as e:
                latencia = -1
                respuesta = f"EXCEPTION: {str(e)}"
                print(f"ERROR EXCEPTION: {e}")
                
            resultados.append({
                "pregunta_id": i,
                "pregunta": pregunta,
                "estrategia": estrategia,
                "latencia_s": latencia,
                "respuesta": respuesta
            })
            
    # Guardar resultados
    os.makedirs("output", exist_ok=True)
    df_res = pd.DataFrame(resultados)
    df_res.to_csv("output/biomistral_prompting.csv", index=False)
    print("Guardado en output/biomistral_prompting.csv\n")
    
    # Liberar VRAM
    forzar_descarga_vram("biomistral-7B")

def probar_estrategias_sintesis_lote2():
    print(">>> Iniciando pruebas de estrategias de prompting para Síntesis (Qwen2.5-7B) LOTE 2 <<<")
    df_analiticas = cargar_analiticas('analiticas/lote2')
    if df_analiticas.empty:
        print("No se encontraron analíticas en lote 2.")
        return

    resultados = []
    
    estrategias_qwen_lote2 = ["Zero-Shot", "Few-Shot", "CoT", "CoT+FS"]
    
    for _, row in df_analiticas.iterrows():
        id_analitica = row['id']
        texto = row['texto']
        
        for estrategia in estrategias_qwen_lote2:
            print(f"[{id_analitica}] Evaluando con {estrategia}...", end=" ", flush=True)
            
            payload = {
                "texto_clinico": texto,
                "metodologia_prompt": estrategia
            }
            
            start_time = time.time()
            try:
                res = requests.post(URL_SYNTHESIZE, json=payload, timeout=120)
                if res.ok:
                    latencia = time.time() - start_time
                    sintesis = res.json().get("sintesis_clinica", "ERROR")
                    print(f"OK ({latencia:.2f}s)")
                else:
                    latencia = -1
                    sintesis = f"HTTP ERROR: {res.status_code}"
                    print("ERROR HTTP")
            except Exception as e:
                latencia = -1
                sintesis = f"EXCEPTION: {str(e)}"
                print(f"ERROR EXCEPTION: {e}")
                
            resultados.append({
                "id_analitica": id_analitica,
                "estrategia": estrategia,
                "latencia_s": latencia,
                "sintesis": sintesis
            })
            
    # Guardar resultados
    os.makedirs("output", exist_ok=True)
    df_res = pd.DataFrame(resultados)
    df_res.to_csv("output/qwen_prompting_lote2.csv", index=False)
    print("Guardado en output/qwen_prompting_lote2.csv\n")
    
    # Liberar VRAM
    forzar_descarga_vram("qwen2.5-7B")

def probar_estrategias_consultas_lote2():
    print(">>> Iniciando pruebas de estrategias de prompting para Consultas (Biomistral-7B) LOTE 2 <<<")
    
    resultados = []
    
    for i, pregunta in enumerate(PREGUNTAS_BIOMISTRAL_LOTE2, 1):
        for estrategia in ESTRATEGIAS_BIOMISTRAL:
            print(f"[Pregunta Lote2 {i}] Evaluando con {estrategia}...", end=" ", flush=True)
            
            payload = {
                "consulta": pregunta,
                "metodologia_prompt": estrategia
            }
            
            start_time = time.time()
            try:
                res = requests.post(URL_CONSULT, json=payload, timeout=120)
                if res.ok:
                    latencia = time.time() - start_time
                    respuesta = res.json().get("respuesta", "ERROR")
                    print(f"OK ({latencia:.2f}s)")
                else:
                    latencia = -1
                    respuesta = f"HTTP ERROR: {res.status_code}"
                    print("ERROR HTTP")
            except Exception as e:
                latencia = -1
                respuesta = f"EXCEPTION: {str(e)}"
                print(f"ERROR EXCEPTION: {e}")
                
            resultados.append({
                "pregunta_id": i,
                "pregunta": pregunta,
                "estrategia": estrategia,
                "latencia_s": latencia,
                "respuesta": respuesta
            })
            
    # Guardar resultados
    os.makedirs("output", exist_ok=True)
    df_res = pd.DataFrame(resultados)
    df_res.to_csv("output/biomistral_prompting_lote2.csv", index=False)
    print("Guardado en output/biomistral_prompting_lote2.csv\n")
    
    # Liberar VRAM
    forzar_descarga_vram("biomistral-7B")

if __name__ == "__main__":
    # Comentamos las del lote 1 para no repetirlas si no se necesita
    # probar_estrategias_sintesis()
    # probar_estrategias_consultas()
    # probar_estrategias_sintesis_lote2()
    probar_estrategias_consultas_lote2()
