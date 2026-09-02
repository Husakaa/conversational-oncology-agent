import requests
import time
import json
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

URL_SYNTHESIZE = "http://127.0.0.1:8000/api/v1/synthesize"
URL_CONSULT = "http://127.0.0.1:8000/api/v1/consult"
URL_OLLAMA_NATIVA = "http://127.0.0.1:11434/api/chat"

MODELO_QWEN = "qwen-oncologo"
MODELO_BIO = "biomistral-oncologo"

PREGUNTAS_BIOMISTRAL = [
    "¿Cuál es la definición clínica de neutropenia febril y qué riesgo implica?",
    "Explica la diferencia entre una toxicidad por inmunoterapia (irAE) y una citotóxica.",
    "Describe los criterios para clasificar una anemia como Grado 3 según el CTCAE v5.0.",
    "¿Qué medidas de soporte se recomiendan ante una trombocitopenia de Grado 2 tras carboplatino?",
    "¿Cómo influye una elevación de creatinina Grado 2 en la dosificación de cisplatino?",
    "Describe la fisiopatología de la mucositis oral inducida por radioterapia.",
    "¿Cuál es el significado clínico de una elevación aislada de la LDH en linfomas?",
    "Explica la importancia de la mutación BRAF V600E en el melanoma metastásico.",
    "¿Qué relación existe entre los niveles de Albúmina sérica y el pronóstico nutricional?",
    "Define el concepto de TMB y su valor predictivo en inmunoterapia.",
    "¿Por qué es crítico monitorizar la FEVI en pacientes tratados con antraciclinas?",
    "Explica la utilidad de la PCR como marcador de inflamación sistémica en cáncer.",
    "¿Cómo actúan los anticuerpos monoclonales anti-PD-1 en el microambiente tumoral?",
    "Describe el mecanismo de acción de los inhibidores de la aromatasa.",
    "¿Qué es el síndrome de lisis tumoral y qué hallazgos analíticos lo caracterizan?",
    "Diferencia de mecanismo entre un TKI y un antimetabolito como el 5-FU.",
    "¿Cuáles son los principales efectos adversos de los inhibidores anti-VEGF?",
    "Define el concepto de ventana terapéutica en fármacos citostáticos.",
    "¿Qué es la caquexia tumoral y cuáles son los mediadores implicados?",
    "Explica el estado de hipercoagulabilidad asociado al cáncer (Síndrome de Trousseau).",
    "¿Cómo se debe realizar la corrección de una hiponatremia severa (122 mEq/L)?",
    "Describe la importancia de la barrera hematoencefálica en metástasis cerebrales.",
    "¿Qué implicaciones tiene una hipercalcemia maligna en el estado de consciencia?",
    "Describe el papel del sistema linfático en la diseminación de carcinomas.",
    "Explica el concepto de resistencia adquirida en terapias contra EGFR."
]

def forzar_descarga_vram(nombre_modelo):
    print(f"\nVaciando {nombre_modelo} de la VRAM...")
    try:
        requests.post(URL_OLLAMA_NATIVA, json={"model": nombre_modelo, "keep_alive": 0})
        time.sleep(3)
        print("VRAM liberada.")
    except Exception as e:
        print(f"Error liberando VRAM: {e}")

def ejecutar_bateria_secuencial(df_analiticas):
    os.makedirs('output', exist_ok=True)
    num_casos = min(len(df_analiticas), len(PREGUNTAS_BIOMISTRAL), 25)
    
    resultados_maestros = {i: {"iteracion": i+1} for i in range(num_casos)}
    
    print(f"Iniciando evaluación secuencial (N={num_casos})")

    # FASE 1: QWEN 2.5
    print("\nFASE 1: Qwen 2.5 (Síntesis)")
    for i in range(num_casos):
        id_analitica = df_analiticas.iloc[i]['id']
        texto_analitica = df_analiticas.iloc[i]['texto']
        
        print(f"[{i+1}/{num_casos}] Evaluando {id_analitica}...", end=" ", flush=True)
        
        start_qwen = time.time()
        res_qwen = requests.post(URL_SYNTHESIZE, json={
            "texto_clinico": texto_analitica, "metodologia_prompt": "CoT+FS"
        })
        lat_qwen = time.time() - start_qwen
        sintesis = res_qwen.json().get("sintesis_clinica", "ERROR")
        
        resultados_maestros[i].update({
            "analitica_id": id_analitica,
            "latencia_qwen": lat_qwen,
            "output_qwen": sintesis,
            "puntuacion_qwen": None
        })
        print(f"{lat_qwen:.2f}s")

    forzar_descarga_vram(MODELO_QWEN)

    # FASE 2: BIOMISTRAL
    print("\nFASE 2: BioMistral (Consulta)")
    for i in range(num_casos):
        pregunta = PREGUNTAS_BIOMISTRAL[i]
        
        print(f"[{i+1}/{num_casos}] Pregunta {i+1}...", end=" ", flush=True)
        
        start_bio = time.time()
        res_bio = requests.post(URL_CONSULT, json={"consulta": pregunta})
        lat_bio = time.time() - start_bio
        respuesta = res_bio.json().get("respuesta", "ERROR")
        
        resultados_maestros[i].update({
            "pregunta_medica": pregunta,
            "latencia_bio": lat_bio,
            "output_biomistral": respuesta,
            "puntuacion_bio": None
        })
        print(f"{lat_bio:.2f}s")

    forzar_descarga_vram(MODELO_BIO)

    # GUARDADO
    lista_resultados = list(resultados_maestros.values())
    
    with open("output/bateria_resultados.json", "w", encoding="utf-8") as f:
        json.dump(lista_resultados, f, indent=4, ensure_ascii=False)

    df_latencias = pd.DataFrame(lista_resultados)[["iteracion", "latencia_qwen", "latencia_bio"]]
    df_latencias.to_csv("output/latencias_secuenciales.csv", index=False)
    
    # Calcular promedios omitiendo la primera iteración
    if len(df_latencias) > 1:
        avg_qwen = df_latencias["latencia_qwen"][1:].mean()
        avg_bio = df_latencias["latencia_bio"][1:].mean()
        print(f"\n[Promedios de Latencia (omitiendo la primera iteración - carga VRAM)]")
        print(f"Qwen2.5-7B   : {avg_qwen:.2f}s")
        print(f"Biomistral-7B: {avg_bio:.2f}s")

    print("\nProceso finalizado. Datos en carpeta output/")

if __name__ == "__main__":
    from src.test.evaluate_ner_regex import cargar_analiticas
    df = cargar_analiticas('analiticas/lote1')
    if not df.empty:
        ejecutar_bateria_secuencial(df)