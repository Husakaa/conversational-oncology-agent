# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Agente Conversacional Oncológico (TFG - Trabajo de Fin de Grado). Sistema híbrido que combina:
1. Un **motor determinista de extracción (NER por regex)** que estructura biomarcadores de analíticas de laboratorio en texto libre y les asigna un grado de toxicidad CTCAE v5.0.
2. Un **motor generativo (LLM local vía Ollama)** que sintetiza esos datos estructurados en un informe clínico en prosa, o responde consultas conversacionales libres.

La API (FastAPI) expone ambos motores; el frontend (Streamlit) es un chat que permite subir una analítica, extraer entidades, generar un informe completo o preguntar libremente.

## Comandos

Todos los comandos se ejecutan desde `TFG/` (la raíz del repo).

```bash
# Instalar dependencias
pip install -r requirements.txt

# Backend (FastAPI) — expone /docs con Swagger
uvicorn src.main:app --reload --port 8000

# Frontend (Streamlit)
streamlit run app.py

# Con Docker (backend + frontend + Ollama, requiere GPU nvidia para el contenedor ollama)
docker compose up --build

# Tests (pytest)
pytest src/test/test_metrics_recomendadas.py
pytest src/test/test_prompting_strategies.py
pytest src/test/test_metrics_recomendadas.py::test_calcular_metricas_recomendadas  # un solo test
```

`BACKEND_URL` (env var, default `http://127.0.0.1:8000`) apunta el frontend al backend. Ollama debe estar corriendo (`OLLAMA_HOST`, default `http://localhost:11434` fuera de Docker) con los modelos de `OLLAMA_MODELS` (`src/config.py`) ya descargados (`ollama pull <modelo>`), o las llamadas a `/synthesize` y `/consult` devolverán un mensaje de error controlado en vez de lanzar excepción.

Los scripts en `src/test/` fuera de los dos `test_*.py` (p. ej. `evaluate_ner_regex.py`, `evaluate_slm.py`, `analyze_pdqi9.py`, `graph_results.py`, `generate_latex_table.py`) no son suites de pytest: son scripts de evaluación/análisis que se ejecutan directamente (`python -m src.test.evaluate_ner_regex`) y generan CSVs/PNGs en `src/test/plots/` o JSON en `output/`, usados para las métricas y gráficas de la memoria del TFG.

## Arquitectura

### Flujo de datos end-to-end

```
texto analítica (.txt) 
  → MedicalExtractor.analizar_texto()  [src/ner/extractor.py]  → dict de biomarcadores + grado CTCAE
  → clinical_context() / biomarkers_to_markdown()  [src/ui/formatters.py]  → texto para prompt / tabla markdown
  → LLMService._build_prompt() + ollama.chat()  [src/ollama/llm_service.py]  → síntesis clínica en prosa
```

### `src/ner/` — motor determinista (sin LLM)
- `patterns.py`: un `re.compile(...)` por biomarcador (Hb, Plaquetas, Neutrófilos, Creatinina, GGT, ALT, etc.), con grupos nombrados `valor`, `unidad`, `inferior`, `superior`. También define `REGLAS_CTCAE`: por cada biomarcador, un dict `{"High"/"Low": {"desc_base", "reglas": [(grado, condición_lambda, descripción), ...]}}` que codifica los umbrales CTCAE v5.0.
- `extractor.py` (`MedicalExtractor`): aplica cada patrón regex al texto, y **filtra falsos positivos** con dos validaciones antes de aceptar un valor: (1) `RANGOS_BIOLOGICOS` — rango numérico plausible por biomarcador, y (2) `UNIDADES_VALIDAS` — la unidad extraída debe ser coherente con el biomarcador (se relaja si hay rango de referencia junto al valor, porque entonces el contexto del regex es más fiable). Si el documento no alcanza `MIN_BIOMARKERS` (3) ni `MIN_CAPTURE_RATE` (10% de los patrones), lanza `InvalidDocumentError` — así se rechazan textos que no son analíticas.
- Añadir un biomarcador nuevo requiere tocar tres sitios en sincronía: el patrón regex en `patterns.py`, sus reglas en `REGLAS_CTCAE`, y la entrada correspondiente en `self.mapa_nombres` de `MedicalExtractor` (que traduce la clave del patrón a la clave clínica usada por `REGLAS_CTCAE`) — y normalmente también `RANGOS_BIOLOGICOS`/`UNIDADES_VALIDAS`.

### `src/ollama/llm_service.py` — motor generativo
- `LLMService` mantiene un registro de modelos (`OLLAMA_MODELS`) y sus hiperparámetros de inferencia (`options_qwen25`, `options_qwen3`, `options_gemma4`, `options_biomistral`, todos en `src/config.py`).
- `_build_prompt()` selecciona la plantilla desde `MODEL_PROMPTS[model_id][methodology]` (fallback a `MODEL_PROMPTS["default"]`). Metodologías soportadas: `Zero-Shot`, `Few-Shot`, `CoT`, `CoT+FS`. Los prompts de síntesis (`generate_synthesis`) usan datos estructurados como contexto; los de consulta (`generate_response`, `consult=True`) usan además `self.last_biomarkers` (última tabla extraída, guardada tras la última síntesis) como contexto adicional en `context_block`.
- El modelo por defecto de síntesis es `qwen2.5-7B`; el de consulta libre es `biomistral-7B`. Cada familia de modelo tiene su propia plantilla de prompt en `src/config.py::MODEL_PROMPTS`, ajustada a cómo responde ese modelo (p. ej. BioMistral necesita instrucciones explícitas para no filtrar su cadena de razonamiento en `Zero-Shot`/`Few-Shot`, y usa las etiquetas `<razonamiento>`/`<respuesta>` en `CoT`/`CoT+FS`).

### `src/api/` — capa HTTP (FastAPI)
- `routes.py` instancia `MedicalExtractor` y `LLMService` una sola vez a nivel de módulo (no por request).
- `POST /api/v1/extract`: solo motor regex, texto → JSON estructurado.
- `POST /api/v1/synthesize`: acepta `datos_estructurados` ya extraídos o `texto_clinico` crudo (en cuyo caso extrae primero); llama al LLM para redactar la síntesis.
- `POST /api/v1/consult`: consulta libre en lenguaje natural al LLM, reutilizando el contexto de la última analítica extraída si existe.
- `models.py` define los esquemas Pydantic de request/response.

### `app.py` — frontend Streamlit
Chat con estado en `st.session_state` (no persistente entre sesiones). Consume la API por HTTP (`requests`), nunca importa `src/ner` o `src/ollama` directamente. `src/ui/formatters.py` sí se importa directamente en el frontend para formatear las respuestas de la API en DataFrames/markdown para pantalla.

### `src/test/` — evaluación y métricas del TFG
Scripts que corren el pipeline completo (o partes) contra los corpus de analíticas (`analiticas/lote1`, `analiticas/lote2`) para producir las métricas, tablas LaTeX y gráficas usadas en la memoria: distribución de biomarcadores, comparación de estrategias de prompting, evaluación PDQI-9, comparación con Gemini como juez, etc. Los datos de salida (CSVs, PNGs) viven en `src/test/plots/`.

## Notas importantes

- Los corpus de analíticas reales (`analiticas/`) y la carpeta `output/` están en `.gitignore` — contienen datos clínicos y no se versionan.
- `.env` contiene una clave de API (Gemini) real — nunca la muestres ni la incluyas en commits, logs o salidas.
- Los prompts en `src/config.py` son el resultado de iteración/tuning específico para cada modelo; si se modifica uno, conviene revisar si las reglas de formato (extensión, ausencia de viñetas, no inventar datos del paciente) siguen intactas, ya que son parte del diseño anti-alucinación del sistema.
