"""
Evalúa con Gemini (LLM-as-a-Judge, rúbrica PDQI-9) síntesis narrativas generadas por
src/test/benchmark_sintesis_slms.py, comparándolas contra la analítica original.

Reutiliza el prompt de juez (PROMPT_QWEN) y el cargador de biomarcadores originales
(cargar_bo) ya usados en evaluate_with_gemini.py, para mantener la misma rúbrica/
metodología de evaluación en todo el TFG.

Genérico: sirve tanto para el benchmark de confirmación a escala completa
(benchmark_sintesis_lote1.csv / lote2.csv) como para los barridos exploratorios de
metodología/temperature (sweep_*.csv), pasando la ruta de CSV/JSON y el lote de
analíticas de origen (para cargar_bo) como argumentos.

Uso:
    python -m src.test.evaluate_benchmark_gemini [csv_path] [json_path] [--lote lote1]

Sin argumentos, evalúa el benchmark de confirmación de lote1 (comportamiento histórico).
Guardado incremental (reanudable): identifica cada fila evaluada por sus columnas
"identificadoras" (todas las del CSV salvo latencia_s/sintesis), así que sirve igual para
un CSV con columnas (model_key, estrategia, id_analitica) que para uno con
(model_key, temperature, id_analitica).
"""
import argparse
import os
import json
import time
import pandas as pd

from src.test.evaluate_with_gemini import client, MODEL_NAME, PROMPT_QWEN, cargar_bo
from google.genai import types

CSV_PATH_DEFAULT = "output/benchmark_sintesis_lote1.csv"
JSON_PATH_DEFAULT = "output/evaluacion_benchmark_gemini.json"

# Columnas que solo son metadatos de la generación local, no identifican la fila
COLUMNAS_NO_ID = {"latencia_s", "sintesis"}


def evaluar(csv_path: str = CSV_PATH_DEFAULT, json_path: str = JSON_PATH_DEFAULT, lote: str = "lote1"):
    print(f">>> Evaluando {csv_path} con Gemini LLM-as-a-Judge (analíticas de {lote}) <<<")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"No se encontró {csv_path}")
        return

    columnas_id = [c for c in df.columns if c not in COLUMNAS_NO_ID]

    resultados = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                resultados = json.load(f)
            print(f"Retomando desde {len(resultados)} evaluaciones previas...")
        except Exception:
            pass

    evaluados = {tuple(r.get(c) for c in columnas_id) for r in resultados}

    for idx, row in df.iterrows():
        clave = tuple(row[c] for c in columnas_id)
        sintesis = row["sintesis"]
        etiqueta = " | ".join(f"{c}={row[c]}" for c in columnas_id)

        if clave in evaluados:
            continue

        if pd.isna(sintesis) or "ERROR" in str(sintesis) or "EXCEPTION" in str(sintesis) or "Asegúrate de que Ollama" in str(sintesis):
            print(f"[{etiqueta}] - Saltando por error o ausencia en síntesis")
            continue

        bo_text = cargar_bo(row["id_analitica"], f"analiticas/{lote}")
        prompt = PROMPT_QWEN.format(bo=bo_text, ng=sintesis)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[{etiqueta}] - Evaluando (intento {attempt+1})...", end="", flush=True)
                res = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                data = json.loads(res.text)
                for c in columnas_id:
                    data[c] = row[c]
                resultados.append(data)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(resultados, f, indent=2, ensure_ascii=False)

                print(" ✅ Hecho")
                time.sleep(4.1)  # Respetar rate limits de Gemini free tier (15 RPM -> 4s por request)
                break
            except Exception as e:
                print(f" ❌ Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)

    if resultados:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print(f"\nGuardado en {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default=CSV_PATH_DEFAULT)
    parser.add_argument("json_path", nargs="?", default=JSON_PATH_DEFAULT)
    parser.add_argument("--lote", default="lote1", help="Carpeta de analiticas/ de origen (para cargar_bo)")
    args = parser.parse_args()
    evaluar(args.csv_path, args.json_path, lote=args.lote)
