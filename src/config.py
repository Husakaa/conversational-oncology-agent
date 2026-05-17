OLLAMA_MODELS = {
    "qwen2.5-7B": "qwen2.5-7B",
    "qwen3-0.6B": "qwen3-0.6B",
    "gemma4-nano-e2b": "gemma4-nano-e2b",
    "biomistral-7B": "biomistral-oncologo"
}

QWEN25_OPTIONS = {
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.9,
    "num_ctx": 2048,
    "num_predict": 512
}

QWEN3_OPTIONS = {
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.8,
    "min_p": 0,
    "num_ctx": 4096,
    "num_predict": 2024,
    "repeat_penalty": 1.2
}

GEMMA4_OPTIONS = {
    "temperature": 0.8,
    "top_k": 20,
    "top_p": 0.9,
    "num_ctx": 2048,
    "num_predict": 2024
}

BIOMISTRAL_OPTIONS = {
    "temperature": 0.4,
    "top_k": 30,
    "top_p": 0.85,
    "num_ctx": 2048,
    "num_predict": 512
}
EJEMPLO_FS_INPUT = """[Fecha no especificada]: Hb: 14.5 | Plaquetas: 147 | Neutrófilos: 3.41 | Creatinina: 0.50 | GGT: 297 | ALT: 83 | resto bien.

----------------------------------------------------------------------------------------------------

SECCIÓN C: ESTRATIFICACIÓN DE TOXICIDAD (CTCAE v5.0) ⚠️
    - GGT (297 U/L): Grado 2. Elevación GGT Moderada (214 - 425 U/L). (Ref: 5.0 - 40.0).
    - ALT (83 U/L): Grado 1. Elevación ALT (GPT) Leve (41 - 120 U/L). (Ref: 3.0 - 40.0).
    - FA (180 U/L): Grado 1. Elevación Fosfatasa Alc. Leve (117 - 291 U/L). (Ref: 35.0 - 104.0).
    - AST (51 U/L): Grado 1. Elevación AST (GOT) Leve (41 - 120 U/L). (Ref: 3.0 - 40.0).
    - LDH (274 U/L): Grado 0 (Normal). Nota: Resultado corregido estadísticamente por hemólisis. (Ref: 200.0 - 380.0)."""

EJEMPLO_FS_OUTPUT = """SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊

El evento limitante de dosis es la elevación de GGT Grado 2, que junto a la elevación leve de fosfatasa alcalina y transaminasas (G1), sugiere un perfil de colestasis o afectación biliar incipiente. Las series hematológicas (neutrófilos y plaquetas) y la función renal se encuentran preservadas dentro de rangos seguros. Se recomienda monitorizar el perfil hepático en el próximo ciclo y repetir la toma de LDH debido a la interferencia por hemólisis."""

MODEL_PROMPTS = {
    "qwen3-0.6B": {
        "Zero-Shot": """Basado en estos resultados, devuelve el análisis estrictamente en este formato JSON:
{{
  "hallazgo_limitante": "Nombre del biomarcador",
  "grado_hallazgo": "Número del grado",
  "otras_alteraciones": ["biomarcador 1", "biomarcador 2"],
  "plan_sugerido": "Acción breve de 5 palabras"
}}
RESULTADOS:
{context}""",
        "Few-Shot": """Eres un oncólogo experto. Analiza los resultados y devuelve el análisis estrictamente en este formato JSON:
{{
  "hallazgo_limitante": "Nombre del biomarcador",
  "grado_hallazgo": "Número del grado",
  "otras_alteraciones": ["biomarcador 1", "biomarcador 2"],
  "plan_sugerido": "Acción breve de 5 palabras"
}}
EJEMPLO INPUT:
""" + EJEMPLO_FS_INPUT + """
EJEMPLO OUTPUT:
{{
  "hallazgo_limitante": "GGT",
  "grado_hallazgo": "2",
  "otras_alteraciones": ["ALT", "FA", "AST"],
  "plan_sugerido": "Monitorizar en próximo ciclo"
}}
TAREA ACTUAL:
INPUT:
{context}
OUTPUT:""",
        "CoT": """Eres un oncólogo clínico experto. Analiza la fisiopatología conjunta de las toxicidades detectadas.
Basado exclusivamente en los resultados de laboratorio proporcionados, devuelve el análisis estrictamente en este formato JSON:
{{
  "hallazgo_limitante": "Nombre del biomarcador limitante",
  "grado_hallazgo": "Número del grado",
  "otras_alteraciones": ["biomarcador 1", "biomarcador 2"],
  "plan_sugerido": "Acción breve de 5 palabras"
}}
NO inventes datos.
DATOS DEL PACIENTE:
{context}""",
        "CoT+FS": """Eres un oncólogo clínico experto. Analiza la fisiopatología de las toxicidades detectadas.
Devuelve el análisis estrictamente en este formato JSON:
{{
  "hallazgo_limitante": "Nombre del biomarcador",
  "grado_hallazgo": "Número del grado",
  "otras_alteraciones": ["biomarcador 1", "biomarcador 2"],
  "plan_sugerido": "Acción breve de 5 palabras"
}}
EJEMPLO DE REFERENCIA:
INPUT:
""" + EJEMPLO_FS_INPUT + """
OUTPUT:
{{
  "hallazgo_limitante": "GGT",
  "grado_hallazgo": "2",
  "otras_alteraciones": ["ALT", "FA", "AST"],
  "plan_sugerido": "Monitorizar en próximo ciclo"
}}
TAREA ACTUAL:
INPUT:
{context}
OUTPUT:"""
    },
    "default": {
        "consult": """Eres un oncólogo experto. Tienes un alto dominio del ámbito médico.
Vas a recibir consultas y debes contestar de forma altamente explicativa dando detalles.
Si se proporcionan resultados de laboratorio, úsalos como referencia para contextualizar tu respuesta.{context_block}
Consulta: {context}""",
        "Zero-Shot": """Basado en estos resultados, redacta únicamente la 'SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊'.
RESULTADOS:
{context}
SECCIÓN D:""",
        "Few-Shot": """Eres un oncólogo experto. Sigue el estilo del ejemplo para redactar la 'SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊'.
EJEMPLO:
INPUT:
""" + EJEMPLO_FS_INPUT + """
OUTPUT SECCIÓN D:
""" + EJEMPLO_FS_OUTPUT + """
TAREA ACTUAL:
INPUT:
{context}
OUTPUT SECCIÓN D:""",
        "CoT": """Eres un oncólogo clínico experto redactando la evolución en una historia clínica.
Analiza la fisiopatología conjunta de las toxicidades detectadas y redacta una síntesis clínica profesional.

REGLAS ESTRICTAS DE FORMATO:
    1. NO inventes NINGÚN dato del paciente (ni edad, ni sexo, ni diagnóstico, ni tratamientos previos). No sabes quién es.
    2. Básate EXCLUSIVAMENTE en los resultados de laboratorio proporcionados.
    3. NO uses viñetas, ni palabras como "Paso 1", "Paso 2", "Análisis individual" o "Conclusión".
    4. Escribe un único texto narrativo (prosa médica) conectando los hallazgos de forma lógica.
    5. Tu respuesta DEBE empezar directamente por 'SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊'.

DATOS DEL PACIENTE:
{context}""",
        "CoT+FS": """Eres un oncólogo clínico experto. Tu tarea es analizar la fisiopatología conjunta de las toxicidades y redactar una síntesis profesional siguiendo estrictamente el estilo narrativo del ejemplo.

EJEMPLO DE REFERENCIA (ESTILO Y TONO):
INPUT:
""" + EJEMPLO_FS_INPUT + """
OUTPUT:
""" + EJEMPLO_FS_OUTPUT + """

REGLAS ESTRICTAS DE SEGURIDAD Y FORMATO:
1. NO inventes ningún dato (edad, sexo, diagnóstico o fármacos). Si no está en el INPUT, no existe.
2. NO incluyas el proceso de pensamiento ("Paso 1", "Paso 2") en la respuesta final.
3. Escribe un texto único en prosa médica, sin viñetas ni etiquetas de sección internas.
4. La respuesta DEBE empezar directamente con 'SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊'.
5. Usa un tono analítico, conectando cómo una alteración puede influir en otra.
6. INDEPENDENCIA DEL EJEMPLO: Usa el EJEMPLO DE REFERENCIA solo para aprender el tono y la estructura. 

TAREA ACTUAL:
INPUT:
{context}

INSTRUCCIÓN FINAL: Redacta ahora el informe. Empieza tu respuesta exactamente con la frase 'SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊' y continúa con la prosa médica."""
    }
}
