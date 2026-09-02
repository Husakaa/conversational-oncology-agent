"""
Benchmark comparativo de los 3 SLMs candidatos a la tarea de síntesis narrativa
(qwen2.5-7B, gemma4-nano-e2b, qwen3-0.6B) tras la optimización de prompts (system
prompt del Modelfile + prompt de usuario en src/config.py) que corrige el conflicto
de formato entre ambas capas y el manejo del canal "thinking" nativo de Ollama.

Genera 1 informe por analítica del lote1, usando la configuración (metodología de
prompting + hiperparámetros) considerada óptima para cada modelo tras la fase de
exploración manual:
    - qwen2.5-7B:      CoT+FS  (config. de producción ya validada)
    - gemma4-nano-e2b: CoT     (mejor relación fidelidad/latencia observada)
    - qwen3-0.6B:      CoT+FS  (mejor esfuerzo; se documenta como resultado negativo)

No sobrescribe los CSV de qwen2.5-7B ya existentes de la evaluación de estrategias de
prompting (output/qwen_prompting_lote1.csv) — este script es un benchmark aparte, con
su propio conjunto de ficheros de salida (prefijo "benchmark_sintesis_").

Requiere el backend arrancado con DEV_MODE=true (para poder fijar model_key por request).
"""
import requests
import time
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.test.evaluate_ner_regex import cargar_analiticas
from src.test.evaluate_slm import forzar_descarga_vram
from src.config import METODOLOGIAS_PROMPT

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8010")
URL_SYNTHESIZE = f"{BACKEND_URL}/api/v1/synthesize"

# Subconjunto fijo de lote1 usado en las fases exploratorias (barrido de metodologías e
# hiperparámetros): mantiene el coste acotado y la selección reproducible.
SUBSET_EXPLORATORIO = [f"Analitica{i}" for i in range(1, 9)]

# model_key -> metodología óptima elegida tras la exploración manual
# (qwen3.5-4B: CoT+FS contamina el grado CTCAE copiándolo del ejemplo few-shot —
# p. ej. reporta "Grado 3" para una neutropenia que es Grado 4 — así que se usa CoT sin ejemplos)
CONFIG_OPTIMA = {
    "qwen2.5-7B": "CoT+FS",
    "gemma4-nano-e2b": "CoT",
    "qwen3-0.6B": "CoT+FS",
    "qwen3.5-4B": "CoT",
}


def _llamar_synthesize(payload: dict, timeout: int = 180):
    """POST a /synthesize devolviendo (sintesis, latencia_s). Nunca lanza: los fallos de
    red/HTTP quedan codificados en el propio texto de síntesis (empieza por "HTTP ERROR"
    o "EXCEPTION"), como ya hacían los distintos scripts de benchmark de src/test/."""
    start = time.time()
    try:
        res = requests.post(URL_SYNTHESIZE, json=payload, timeout=timeout)
        latencia = time.time() - start
        if res.ok:
            return res.json().get("sintesis_clinica", "ERROR"), latencia
        return f"HTTP ERROR: {res.status_code} {res.text}", latencia
    except Exception as e:
        return f"EXCEPTION: {str(e)}", -1


def ejecutar_benchmark(lote="lote1", modelos=None):
    """modelos: subconjunto de CONFIG_OPTIMA a ejecutar (None = todos). Los resultados se
    AÑADEN al CSV existente (si lo hay) en vez de sobrescribirlo, para poder incorporar un
    modelo nuevo sin repetir los ya evaluados."""
    df_analiticas = cargar_analiticas(f"analiticas/{lote}")
    if df_analiticas.empty:
        print(f"No se encontraron analíticas en {lote}.")
        return

    os.makedirs("output", exist_ok=True)
    csv_path = f"output/benchmark_sintesis_{lote}.csv"

    previos = []
    if os.path.exists(csv_path):
        previos = pd.read_csv(csv_path).to_dict("records")
        print(f"Retomando {len(previos)} resultados previos de {csv_path}")

    resultados = list(previos)
    config = {k: v for k, v in CONFIG_OPTIMA.items() if modelos is None or k in modelos}

    for model_key, estrategia in config.items():
        print(f"\n>>> Modelo: {model_key} | Estrategia: {estrategia} <<<")
        for _, row in df_analiticas.iterrows():
            id_analitica = row["id"]
            texto = row["texto"]

            payload = {
                "texto_clinico": texto,
                "metodologia_prompt": estrategia,
                "model_key": model_key,
            }

            print(f"[{model_key}] {id_analitica}...", end=" ", flush=True)
            sintesis, latencia = _llamar_synthesize(payload)
            print(f"OK ({latencia:.2f}s)" if not str(sintesis).startswith(("HTTP ERROR", "EXCEPTION")) else f"ERROR: {sintesis[:80]}")

            resultados.append({
                "model_key": model_key,
                "estrategia": estrategia,
                "id_analitica": id_analitica,
                "latencia_s": latencia,
                "sintesis": sintesis,
            })

            # Guardado incremental por si el proceso se interrumpe
            pd.DataFrame(resultados).to_csv(csv_path, index=False)

        forzar_descarga_vram(model_key)

    print(f"\nBenchmark completo. Guardado en {csv_path}")

    # Resumen rápido de latencias
    df_res = pd.DataFrame(resultados)
    resumen = df_res[df_res["latencia_s"] > 0].groupby("model_key")["latencia_s"].agg(["mean", "min", "max", "count"])
    print("\nResumen de latencias:")
    print(resumen)


def sweep_metodologias(ids_subset=None, modelos=None, lote="lote1"):
    """Fase A (exploratoria): las 4 metodologías x cada modelo, sobre un subconjunto
    pequeño de analíticas — para elegir la metodología ganadora de cada modelo con datos
    en vez de por inspección manual de 1-2 casos. Hiperparámetros: los de MODEL_OPTIONS_MAP
    tal cual (no se tocan aquí, solo se compara el efecto del prompt)."""
    ids_subset = ids_subset or SUBSET_EXPLORATORIO
    df_analiticas = cargar_analiticas(f"analiticas/{lote}")
    df_analiticas = df_analiticas[df_analiticas["id"].isin(ids_subset)]
    if df_analiticas.empty:
        print(f"No se encontraron analíticas de {ids_subset} en {lote}.")
        return

    os.makedirs("output", exist_ok=True)
    csv_path = f"output/sweep_metodologias_{lote}_subset.csv"

    resultados = []
    if os.path.exists(csv_path):
        resultados = pd.read_csv(csv_path).to_dict("records")
        print(f"Retomando {len(resultados)} resultados previos de {csv_path}")

    # Por defecto, solo los modelos candidatos a síntesis (CONFIG_OPTIMA) — NO OLLAMA_MODELS
    # completo, que también incluye biomistral-7B (agente conversacional, fuera de alcance:
    # sus prompts no tienen la clave "context_block" fuera del flujo de consulta y el
    # endpoint /synthesize devuelve 500 si se le pide una síntesis).
    modelos_a_probar = modelos or list(CONFIG_OPTIMA.keys())
    ya_hechos = {(r["model_key"], r["estrategia"], r["id_analitica"]) for r in resultados}

    for model_key in modelos_a_probar:
        for estrategia in METODOLOGIAS_PROMPT:
            print(f"\n>>> [Sweep metodologías] {model_key} | {estrategia} <<<")
            for _, row in df_analiticas.iterrows():
                id_analitica = row["id"]
                if (model_key, estrategia, id_analitica) in ya_hechos:
                    continue

                payload = {
                    "texto_clinico": row["texto"],
                    "metodologia_prompt": estrategia,
                    "model_key": model_key,
                }
                print(f"[{model_key}/{estrategia}] {id_analitica}...", end=" ", flush=True)
                sintesis, latencia = _llamar_synthesize(payload)
                print(f"OK ({latencia:.2f}s)" if not str(sintesis).startswith(("HTTP ERROR", "EXCEPTION")) else f"ERROR: {sintesis[:80]}")

                resultados.append({
                    "model_key": model_key,
                    "estrategia": estrategia,
                    "id_analitica": id_analitica,
                    "latencia_s": latencia,
                    "sintesis": sintesis,
                })
                pd.DataFrame(resultados).to_csv(csv_path, index=False)

        forzar_descarga_vram(model_key)

    print(f"\nSweep de metodologías completo. Guardado en {csv_path}")


def sweep_temperature(model_key: str, estrategia: str, valores: list, ids_subset=None, lote="lote1"):
    """Fase B (exploratoria): barrido ligero de `temperature` para un modelo/metodología ya
    ganadores, sobre el mismo subconjunto de analíticas. El resto de hiperparámetros del
    modelo se mantienen (solo se sobreescribe `temperature` vía el override de
    hiperparámetros que ya soporta /synthesize en DEV_MODE)."""
    ids_subset = ids_subset or SUBSET_EXPLORATORIO
    df_analiticas = cargar_analiticas(f"analiticas/{lote}")
    df_analiticas = df_analiticas[df_analiticas["id"].isin(ids_subset)]
    if df_analiticas.empty:
        print(f"No se encontraron analíticas de {ids_subset} en {lote}.")
        return

    os.makedirs("output", exist_ok=True)
    csv_path = f"output/sweep_temperature_{lote}_subset.csv"

    resultados = []
    if os.path.exists(csv_path):
        resultados = pd.read_csv(csv_path).to_dict("records")
        print(f"Retomando {len(resultados)} resultados previos de {csv_path}")

    ya_hechos = {(r["model_key"], r["temperature"], r["id_analitica"]) for r in resultados}

    for temperature in valores:
        print(f"\n>>> [Sweep temperature] {model_key} | {estrategia} | temperature={temperature} <<<")
        for _, row in df_analiticas.iterrows():
            id_analitica = row["id"]
            if (model_key, temperature, id_analitica) in ya_hechos:
                continue

            payload = {
                "texto_clinico": row["texto"],
                "metodologia_prompt": estrategia,
                "model_key": model_key,
                "hyperparams": {"temperature": temperature},
            }
            print(f"[{model_key}/T={temperature}] {id_analitica}...", end=" ", flush=True)
            sintesis, latencia = _llamar_synthesize(payload)
            print(f"OK ({latencia:.2f}s)" if not str(sintesis).startswith(("HTTP ERROR", "EXCEPTION")) else f"ERROR: {sintesis[:80]}")

            resultados.append({
                "model_key": model_key,
                "temperature": temperature,
                "id_analitica": id_analitica,
                "latencia_s": latencia,
                "sintesis": sintesis,
            })
            pd.DataFrame(resultados).to_csv(csv_path, index=False)

    forzar_descarga_vram(model_key)
    print(f"\nSweep de temperature completo. Guardado en {csv_path}")


if __name__ == "__main__":
    # Uso:
    #   python -m src.test.benchmark_sintesis_slms                     -> confirmación lote1, todos los modelos
    #   python -m src.test.benchmark_sintesis_slms qwen2.5-7B ...       -> confirmación lote1, subconjunto
    #   python -m src.test.benchmark_sintesis_slms --sweep-metodologias -> fase A
    #   python -m src.test.benchmark_sintesis_slms --sweep-temperature MODEL_KEY ESTRATEGIA T1 T2 ... -> fase B
    #   python -m src.test.benchmark_sintesis_slms --lote2              -> confirmación lote2, todos los modelos
    args = sys.argv[1:]
    if args and args[0] == "--sweep-metodologias":
        sweep_metodologias(modelos=args[1:] or None)
    elif args and args[0] == "--sweep-temperature":
        model_key, estrategia, *temps = args[1:]
        sweep_temperature(model_key, estrategia, [float(t) for t in temps])
    elif args and args[0] == "--lote2":
        ejecutar_benchmark("lote2", modelos=args[1:] or None)
    else:
        ejecutar_benchmark("lote1", modelos=args or None)
