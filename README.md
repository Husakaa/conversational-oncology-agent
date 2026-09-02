# Agente Conversacional Oncológico

Trabajo de Fin de Grado (Ingeniería de la Salud, UMA). Sistema híbrido de apoyo a la
decisión clínica en oncología que combina:

1. Un **motor determinista de extracción (NER por regex)** que estructura biomarcadores
   de analíticas de laboratorio en texto libre y les asigna un grado de toxicidad
   [CTCAE v5.0](https://ctep.cancer.gov/protocoldevelopment/electronic_applications/ctc.htm).
2. Un **motor generativo (LLM local vía [Ollama](https://ollama.com))** que sintetiza esos
   datos estructurados en un informe clínico en prosa, o responde consultas
   conversacionales libres.

La API (FastAPI) expone ambos motores; el frontend (Streamlit) es un chat que permite
subir una analítica, extraer entidades, generar un informe completo o preguntar
libremente.

```
texto analítica (.txt)
  → MedicalExtractor.analizar_texto()   [src/ner/extractor.py]
  → dict de biomarcadores + grado CTCAE
  → clinical_context() / biomarkers_to_markdown()   [src/ui/formatters.py]
  → LLMService.generate_synthesis() + ollama.chat()   [src/ollama/llm_service.py]
  → síntesis clínica en prosa
```

## Puesta en marcha

```bash
# Dependencias
pip install -r requirements.txt

# Backend (FastAPI) — expone /docs con Swagger
uvicorn src.main:app --reload --port 8000

# Frontend (Streamlit), en otra terminal
streamlit run app.py

# Con Docker (backend + frontend + Ollama, requiere GPU nvidia para el contenedor ollama)
docker compose up --build
```

Ollama debe estar corriendo (`OLLAMA_HOST`, por defecto `http://localhost:11434` fuera de
Docker) con los modelos de `OLLAMA_MODELS` (`src/config.py`) ya descargados
(`ollama pull <modelo>`); si no, `/synthesize` y `/consult` devuelven un mensaje de error
controlado en vez de lanzar una excepción.

`BACKEND_URL` (env var, por defecto `http://127.0.0.1:8000`) apunta el frontend al
backend.

### Herramientas de desarrollador

Con `DEV_MODE=true` (backend y frontend), el sidebar de Streamlit añade un panel para
alternar el SLM de síntesis, la metodología de prompting (Zero-Shot / Few-Shot / CoT /
CoT+FS) y sus hiperparámetros en tiempo de ejecución, sin tocar código. Desactivado por
defecto — en producción basta con no definir la variable.

## Estructura del proyecto

```
src/
  ner/        # Motor determinista: patrones regex + reglas CTCAE v5.0
  ollama/     # Motor generativo: LLMService, prompts por modelo/metodología
  api/        # FastAPI: rutas, esquemas Pydantic
  ui/         # Formateadores compartidos entre frontend y prompts
  test/       # Suites pytest + scripts de evaluación/benchmark del TFG
app.py        # Frontend Streamlit
```

## Tests y evaluación

```bash
pytest src/test/test_metrics_recomendadas.py
pytest src/test/test_prompting_strategies.py
```

Los demás scripts de `src/test/` (`evaluate_ner_regex.py`, `evaluate_slm.py`,
`benchmark_sintesis_slms.py`, `evaluate_benchmark_gemini.py`, `analyze_pdqi9.py`,
`graph_results.py`...) no son suites de pytest: son scripts de evaluación que se
ejecutan directamente (`python -m src.test.<script>`) y generan CSVs/JSON/PNGs en
`output/`, usados para las métricas y gráficas de la memoria del TFG.

Ver [`CLAUDE.md`](CLAUDE.md) para el detalle de la arquitectura interna.

## Notas

- Los corpus de analíticas reales (`analiticas/`) y `output/` están en `.gitignore`:
  contienen datos clínicos y no se versionan.
- `.env` contiene una clave de API (Gemini, usada como LLM-as-a-Judge en la evaluación)
  — nunca se debe commitear.
