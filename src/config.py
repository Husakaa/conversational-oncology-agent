import os

# Activa las herramientas de desarrollador (selección de SLM, metodología de
# prompting e hiperparámetros en tiempo de ejecución) en el frontend y en la
# API. Desactivado (default) en producción: basta con no definir la variable
# de entorno o ponerla a "false".
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

OLLAMA_MODELS = {
    "qwen2.5-7B": "qwen2.5-7B",
    "qwen3-0.6B": "qwen3-0.6B",
    "gemma4-nano-e2b": "gemma4-nano-e2b",
    "qwen3.5-4B": "qwen3.5-4B",
    "biomistral-7B": "biomistral-7B"
}

# Metodologías de prompting soportadas por los prompts de src/config.py::MODEL_PROMPTS
METODOLOGIAS_PROMPT = ["Zero-Shot", "Few-Shot", "CoT", "CoT+FS"]

QWEN25_OPTIONS = {
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.9,
    "num_ctx": 2048,
    "num_predict": 512,
    "repeat_penalty": 1.2
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

QWEN35_OPTIONS = {
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.8,
    "num_ctx": 4096,
    "num_predict": 2024
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

# Opciones de inferencia por defecto de cada modelo, indexadas por model_key
# (la misma clave que OLLAMA_MODELS). Usado por LLMService para resolver los
# hiperparámetros de una petición y por el endpoint /dev/options para
# exponerlos al frontend como valores de partida editables.
MODEL_OPTIONS_MAP = {
    "qwen2.5-7B": QWEN25_OPTIONS,
    "qwen3-0.6B": QWEN3_OPTIONS,
    "gemma4-nano-e2b": GEMMA4_OPTIONS,
    "qwen3.5-4B": QWEN35_OPTIONS,
    "biomistral-7B": BIOMISTRAL_OPTIONS
}

# ── Ejemplo Few-Shot 1: Perfil hepatobiliar (colestasis) ──
EJEMPLO_FS_1_INPUT = """[Fecha no especificada]: Hb: 14.5 | Plaquetas: 147 | Neutrófilos: 3.41 | Creatinina: 0.50 | GGT: 297 | ALT: 83 | resto bien.

----------------------------------------------------------------------------------------------------

SECCIÓN C: ESTRATIFICACIÓN DE TOXICIDAD (CTCAE v5.0) ⚠️
    - GGT (297 U/L): Grado 2. Elevación GGT Moderada (214 - 425 U/L). (Ref: 5.0 - 40.0).
    - ALT (83 U/L): Grado 1. Elevación ALT (GPT) Leve (41 - 120 U/L). (Ref: 3.0 - 40.0).
    - FA (180 U/L): Grado 1. Elevación Fosfatasa Alc. Leve (117 - 291 U/L). (Ref: 35.0 - 104.0).
    - AST (51 U/L): Grado 1. Elevación AST (GOT) Leve (41 - 120 U/L). (Ref: 3.0 - 40.0).
    - LDH (274 U/L): Grado 0 (Normal). Nota: Resultado corregido estadísticamente por hemólisis. (Ref: 200.0 - 380.0)."""

EJEMPLO_FS_1_OUTPUT = """SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊

El perfil analítico revela una afectación hepatobiliar predominante, con elevación moderada de GGT (Grado 2) acompañada de incrementos leves en fosfatasa alcalina, ALT y AST (Grado 1), configurando un patrón compatible con colestasis incipiente. Las series hematológicas y la función renal permanecen conservadas. Se recomienda control estrecho del perfil hepático en el próximo ciclo y repetición de LDH por interferencia hemolítica en la muestra actual."""

# ── Ejemplo Few-Shot 2: Mielotoxicidad hematológica ──
EJEMPLO_FS_2_INPUT = """[Fecha no especificada]: Hb: 9.8 | Plaquetas: 62 | Neutrófilos: 0.89 | Creatinina: 0.72 | GGT: 38 | ALT: 25 | resto bien.

----------------------------------------------------------------------------------------------------

SECCIÓN C: ESTRATIFICACIÓN DE TOXICIDAD (CTCAE v5.0) ⚠️
    - Neutrófilos (0.89 ×10⁹/L): Grado 3. Neutropenia Grave (0.5 - 1.0 ×10⁹/L). (Ref: 1.5 - 8.0).
    - Plaquetas (62 ×10⁹/L): Grado 2. Trombocitopenia Moderada (50 - 75 ×10⁹/L). (Ref: 150 - 400).
    - Hemoglobina (9.8 g/dL): Grado 1. Anemia Leve (10.0 - LIN o 8.0 - 10.0 g/dL). (Ref: 12.0 - 16.0)."""

EJEMPLO_FS_2_OUTPUT = """SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊

Los resultados muestran un cuadro de mielotoxicidad significativa con neutropenia grave (Grado 3) como hallazgo de mayor relevancia clínica, asociada a trombocitopenia moderada (Grado 2) y anemia leve (Grado 1). La convergencia de estas tres citopenias sugiere supresión medular global, probablemente relacionada con el régimen quimioterápico. La función hepática y renal se mantienen dentro de parámetros normales. Se recomienda valorar el uso profiláctico de G-CSF, controlar el riesgo hemorrágico por la trombocitopenia y reevaluar el hemograma antes del siguiente ciclo para decidir sobre posible reducción de dosis."""

# ── Ejemplo Few-Shot 3: Nefrotoxicidad con hiponatremia ──
EJEMPLO_FS_3_INPUT = """[Fecha no especificada]: Hb: 12.1 | Plaquetas: 198 | Neutrófilos: 4.20 | Creatinina: 1.85 | Na: 128 | K: 5.3 | Urea: 68 | resto bien.

----------------------------------------------------------------------------------------------------

SECCIÓN C: ESTRATIFICACIÓN DE TOXICIDAD (CTCAE v5.0) ⚠️
    - Creatinina (1.85 mg/dL): Grado 2. Elevación de Creatinina Moderada (1.5 - 3.0× LSN). (Ref: 0.6 - 1.2).
    - Sodio (128 mEq/L): Grado 2. Hiponatremia Moderada (125 - 130 mEq/L). (Ref: 136 - 145).
    - Potasio (5.3 mEq/L): Grado 1. Hiperpotasemia Leve (LSN - 5.5 mEq/L). (Ref: 3.5 - 5.0).
    - Urea (68 mg/dL): Grado 0 (Normal). (Ref: 10.0 - 50.0). Nota: Elevada pero sin grado CTCAE asignado."""

EJEMPLO_FS_3_OUTPUT = """SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊

Se identifica un deterioro de la función renal con creatinina en Grado 2, acompañado de hiponatremia moderada (Grado 2) e hiperpotasemia leve (Grado 1), junto a elevación de urea. Este conjunto de alteraciones electrolíticas y de retención nitrogenada configura un patrón compatible con nefrotoxicidad inducida por tratamiento. Las series hematológicas y la función hepática no presentan alteraciones. Es prioritario asegurar una hidratación adecuada, monitorizar la evolución de la función renal y electrolitos en 48-72 horas, y considerar ajuste o suspensión temporal de fármacos nefrotóxicos."""

# ── Ejemplo compacto para qwen3-0.6B: prosa breve, sin encabezados, hallazgos ya priorizados ──
# (reutiliza el caso de mielotoxicidad de EJEMPLO_FS_2, pero con salida en 2 frases y sin "SECCIÓN D")
EJEMPLO_QWEN3_INPUT = EJEMPLO_FS_2_INPUT
EJEMPLO_QWEN3_OUTPUT = """Se observa neutropenia grave (Grado 3) como hallazgo predominante, junto a trombocitopenia moderada (Grado 2) y anemia leve (Grado 1), compatible con mielotoxicidad por quimioterapia. La función hepática y renal se mantienen sin alteraciones."""

MODEL_PROMPTS = {
    # qwen3-0.6B es un modelo de razonamiento (canal "thinking" nativo en Ollama) de solo 0.6B.
    # El grueso de las reglas (formato prosa, prohibido inventar datos, orden de gravedad) vive en
    # el SYSTEM del Modelfile (~/AI_Models/Qwen3_0.6B/Modelfile): el usuario aquí solo aporta la
    # tarea concreta, ya que sobrecargarlo con instrucciones repetidas empeora su cumplimiento.
    # El eje Zero-Shot/Few-Shot vs CoT/CoT+FS controla si se activa su "thinking" nativo
    # (ver THINK_BY_METHOD más abajo) en vez de instruirlo por texto a "razonar paso a paso".
    "qwen3-0.6B": {
        "Zero-Shot": """Redacta la síntesis de estos hallazgos (ya ordenados de mayor a menor gravedad):
{context}""",
        "Few-Shot": """Sigue el estilo del ejemplo (prosa breve, sin encabezados, sin viñetas).

EJEMPLO
HALLAZGOS:
""" + EJEMPLO_QWEN3_INPUT + """
SÍNTESIS:
""" + EJEMPLO_QWEN3_OUTPUT + """

TAREA ACTUAL
HALLAZGOS:
{context}
SÍNTESIS:""",
        "CoT": """Redacta la síntesis de estos hallazgos (ya ordenados de mayor a menor gravedad):
{context}""",
        "CoT+FS": """Sigue el estilo del ejemplo (prosa breve, sin encabezados, sin viñetas).

EJEMPLO
HALLAZGOS:
""" + EJEMPLO_QWEN3_INPUT + """
SÍNTESIS:
""" + EJEMPLO_QWEN3_OUTPUT + """

TAREA ACTUAL
HALLAZGOS:
{context}
SÍNTESIS:"""
    },
    "biomistral-7B": {
        "Zero-Shot": """Por favor, actúa de acuerdo a tu rol de Facultativo Especialista Senior para responder la consulta del usuario basándote EXCLUSIVAMENTE en los siguientes datos extraídos.{context_block}

CONSULTA DEL USUARIO:
{context}

INSTRUCCIÓN FINAL: Responde de forma directa, concisa y clínicamente precisa. NO proporciones tu cadena de razonamiento paso a paso, sólo la respuesta final.""",
        "Few-Shot": """Por favor, actúa de acuerdo a tu rol de Facultativo Especialista Senior para responder la consulta del usuario basándote EXCLUSIVAMENTE en los siguientes datos extraídos.{context_block}

A continuación se muestra un EJEMPLO del estilo de respuesta que se espera de ti (directa y sin cadena de razonamiento):

EJEMPLO DE CONSULTA:
"¿Qué significa una GGT de 297 U/L en un paciente oncológico?"

EJEMPLO DE RESPUESTA IDEAL:
Una GGT de 297 U/L representa una elevación significativa (aproximadamente 7× el límite superior normal), clasificable como toxicidad Grado 2-3 según CTCAE v5.0. En paciente oncológico, las causas más frecuentes incluyen hepatotoxicidad farmacológica y colestasis. Se recomienda correlacionar con otros marcadores hepáticos (FA, ALT, AST, bilirrubina) para determinar el patrón de daño (colestásico vs. hepatocelular) y valorar ajuste de tratamiento si procede.

CONSULTA DEL USUARIO:
{context}

INSTRUCCIÓN FINAL: Responde de forma directa, concisa y clínicamente precisa siguiendo el estilo del ejemplo. NO proporciones tu cadena de razonamiento paso a paso, sólo la respuesta final.""",
        "CoT": """Por favor, actúa de acuerdo a tu rol de Facultativo Especialista Senior para responder la consulta del usuario basándote EXCLUSIVAMENTE en los siguientes datos extraídos.{context_block}

INSTRUCCIONES DE RAZONAMIENTO:
Antes de responder, analiza paso a paso la fisiopatología relevante, los valores de referencia implicados y las posibles causas diferenciales en contexto oncológico. Después, proporciona una respuesta concisa y clínicamente precisa.

CONSULTA DEL USUARIO:
{context}

INSTRUCCIÓN FINAL: Ejecuta tu cadena de razonamiento paso a paso dentro de la etiqueta <razonamiento>, identificando valores de referencia, grado CTCAE si aplica y diagnóstico diferencial. Luego, proporciona tu respuesta clínica en la etiqueta <respuesta>.""",
        "CoT+FS": """Por favor, actúa de acuerdo a tu rol de Facultativo Especialista Senior para responder la consulta del usuario basándote EXCLUSIVAMENTE en los siguientes datos extraídos.{context_block}

INSTRUCCIONES DE RAZONAMIENTO:
Antes de responder, analiza paso a paso la fisiopatología relevante, los valores de referencia implicados y las posibles causas diferenciales en contexto oncológico. Después, proporciona una respuesta concisa y clínicamente precisa.

A continuación se muestra un EJEMPLO del estilo de respuesta que se espera de ti:

EJEMPLO DE CONSULTA:
"El paciente presenta plaquetas de 85.000/µL tras el tercer ciclo de quimioterapia. ¿Cuál es la gravedad y qué implica clínicamente?"

EJEMPLO DE RESPUESTA IDEAL:
<razonamiento>
El rango de referencia de plaquetas es 150.000-400.000/µL. Un valor de 85.000/µL representa trombocitopenia. Según CTCAE v5.0:
- Grado 1: 75.000 - <LIN (150.000)
- Grado 2: 50.000 - <75.000
- Grado 3: 25.000 - <50.000
Por tanto, 85.000/µL se clasifica como Grado 1. La causa más probable en este contexto es mielotoxicidad por quimioterapia. A este nivel, el riesgo hemorrágico espontáneo es bajo, pero debe monitorizarse la tendencia. Si las plaquetas descienden por debajo de 75.000 en ciclos sucesivos, podría requerirse ajuste de dosis o retraso del siguiente ciclo.
</razonamiento>
<respuesta>
Las plaquetas de 85.000/µL constituyen una trombocitopenia Grado 1 (CTCAE v5.0), probablemente secundaria a mielotoxicidad por quimioterapia. A este nivel, el riesgo hemorrágico espontáneo es bajo. Se recomienda monitorizar el hemograma previo al siguiente ciclo; si la cifra desciende por debajo de 75.000/µL (Grado 2), considerar retraso del ciclo o ajuste de dosis según protocolo.
</respuesta>

CONSULTA DEL USUARIO:
{context}

INSTRUCCIÓN FINAL: Ejecuta tu cadena de razonamiento paso a paso dentro de <razonamiento>, identificando valores de referencia, grado CTCAE si aplica y diagnóstico diferencial. Luego, proporciona tu respuesta clínica en <respuesta>."""
    },
    # gemma4-nano-e2b: modelo de razonamiento (canal "thinking" nativo). Con el prompt "default"
    # (qwen2.5-7B) alucinaba el sexo del paciente ("la paciente") pese a la regla explícita, y
    # confundía terminología (p. ej. "leucopenia" en vez de "neutropenia") — se refuerza aquí la
    # fidelidad a los datos y la terminología exacta, además de en el SYSTEM del Modelfile.
    "gemma4-nano-e2b": {
        "Zero-Shot": """Basado en estos resultados, redacta únicamente la 'SECCIÓN D: SÍNTESIS CLÍNICA 📊'.
EXTENSIÓN: Máximo un párrafo de 4-6 frases. Sé conciso y directo.
No menciones el sexo del paciente ("el/la paciente"): no lo conoces. Usa el nombre clínico exacto de cada hallazgo, tal cual aparece en los datos.
RESULTADOS:
{context}
SECCIÓN D:""",
        "Few-Shot": """Eres un oncólogo experto. Sigue el estilo del ejemplo para redactar la 'SECCIÓN D: SÍNTESIS CLÍNICA 📊'.
EXTENSIÓN OBLIGATORIA: Un solo párrafo de máximo 4-6 frases. No te extiendas más.
No menciones el sexo del paciente ("el/la paciente"): no lo conoces. Usa el nombre clínico exacto de cada hallazgo, tal cual aparece en los datos.
EJEMPLO:
INPUT:
""" + EJEMPLO_FS_1_INPUT + """
OUTPUT SECCIÓN D:
""" + EJEMPLO_FS_1_OUTPUT + """
TAREA ACTUAL:
INPUT:
{context}
OUTPUT SECCIÓN D:""",
        "CoT": """Eres un oncólogo clínico experto redactando la evolución en una historia clínica.
Analiza la fisiopatología conjunta de las toxicidades detectadas y redacta una síntesis clínica profesional.

REGLAS ESTRICTAS DE FORMATO:
    1. NO inventes NINGÚN dato del paciente (ni edad, ni sexo, ni diagnóstico, ni tratamientos previos). No sabes quién es: nunca escribas "el paciente" ni "la paciente", refiérete directamente a los hallazgos.
    2. Básate EXCLUSIVAMENTE en los resultados de laboratorio proporcionados. Usa el nombre clínico EXACTO de cada hallazgo tal como aparece en los datos (no sustituyas un término por otro parecido).
    3. NO uses viñetas, ni palabras como "Paso 1", "Paso 2", "Análisis individual" o "Conclusión".
    4. Escribe un único texto narrativo (prosa médica) conectando los hallazgos de forma lógica.
    5. EXTENSIÓN: Un solo párrafo conciso de máximo 4-6 frases. No te extiendas innecesariamente.

DATOS DEL PACIENTE:
{context}""",
        "CoT+FS": """Eres un oncólogo clínico experto. Tu tarea es analizar la fisiopatología conjunta de las toxicidades y redactar una síntesis profesional.

A continuación se muestran TRES EJEMPLOS DE REFERENCIA para que aprendas el tono y la estructura. Observa que cada ejemplo comienza de forma distinta:

--- EJEMPLO 1 ---
INPUT:
""" + EJEMPLO_FS_1_INPUT + """
OUTPUT:
""" + EJEMPLO_FS_1_OUTPUT + """

--- EJEMPLO 2 ---
INPUT:
""" + EJEMPLO_FS_2_INPUT + """
OUTPUT:
""" + EJEMPLO_FS_2_OUTPUT + """

--- EJEMPLO 3 ---
INPUT:
""" + EJEMPLO_FS_3_INPUT + """
OUTPUT:
""" + EJEMPLO_FS_3_OUTPUT + """

REGLAS ESTRICTAS DE SEGURIDAD Y FORMATO:
1. NO inventes ningún dato (edad, sexo, diagnóstico o fármacos). Si no está en el INPUT, no existe. Nunca escribas "el paciente" ni "la paciente".
2. Usa el nombre clínico EXACTO de cada hallazgo tal como aparece en el INPUT (no lo sustituyas por un término parecido).
3. NO incluyas el proceso de pensamiento ("Paso 1", "Paso 2") en la respuesta final.
4. Escribe un texto único en prosa médica, sin viñetas ni etiquetas de sección internas.
5. Usa un tono analítico, conectando cómo una alteración puede influir en otra.
6. INDEPENDENCIA DE LOS EJEMPLOS: Usa los ejemplos solo para aprender el tono y la estructura, NO para copiar contenido.
7. NO copies frases textuales de los ejemplos. Cada informe debe tener su propio inicio y redacción original adaptada a los datos reales del paciente.
8. EXTENSIÓN: Un solo párrafo conciso de máximo 4-6 frases. Observa la brevedad de los ejemplos.

TAREA ACTUAL:
INPUT:
{context}

INSTRUCCIÓN FINAL: Redacta ahora el informe en un solo párrafo breve, con un inicio original y adaptado a los datos proporcionados."""
    },
    "default": {
        "consult": """Eres un oncólogo experto. Tienes un alto dominio del ámbito médico.
Vas a recibir consultas y debes contestar de forma altamente explicativa dando detalles.
Si se proporcionan resultados de laboratorio, úsalos como referencia para contextualizar tu respuesta.{context_block}
Consulta: {context}""",
        "Zero-Shot": """Basado en estos resultados, redacta únicamente la 'SECCIÓN D: SÍNTESIS CLÍNICA 📊'.
EXTENSIÓN: Máximo un párrafo de 4-6 frases. Sé conciso y directo.
RESULTADOS:
{context}
SECCIÓN D:""",
        "Few-Shot": """Eres un oncólogo experto. Sigue el estilo del ejemplo para redactar la 'SECCIÓN D: SÍNTESIS CLÍNICA 📊'.
EXTENSIÓN OBLIGATORIA: Un solo párrafo de máximo 4-6 frases. No te extiendas más.
EJEMPLO:
INPUT:
""" + EJEMPLO_FS_1_INPUT + """
OUTPUT SECCIÓN D:
""" + EJEMPLO_FS_1_OUTPUT + """
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
    5. EXTENSIÓN: Un solo párrafo conciso de máximo 4-6 frases. No te extiendas innecesariamente.

DATOS DEL PACIENTE:
{context}""",
        "CoT+FS": """Eres un oncólogo clínico experto. Tu tarea es analizar la fisiopatología conjunta de las toxicidades y redactar una síntesis profesional.

A continuación se muestran TRES EJEMPLOS DE REFERENCIA para que aprendas el tono y la estructura. Observa que cada ejemplo comienza de forma distinta:

--- EJEMPLO 1 ---
INPUT:
""" + EJEMPLO_FS_1_INPUT + """
OUTPUT:
""" + EJEMPLO_FS_1_OUTPUT + """

--- EJEMPLO 2 ---
INPUT:
""" + EJEMPLO_FS_2_INPUT + """
OUTPUT:
""" + EJEMPLO_FS_2_OUTPUT + """

--- EJEMPLO 3 ---
INPUT:
""" + EJEMPLO_FS_3_INPUT + """
OUTPUT:
""" + EJEMPLO_FS_3_OUTPUT + """

REGLAS ESTRICTAS DE SEGURIDAD Y FORMATO:
1. NO inventes ningún dato (edad, sexo, diagnóstico o fármacos). Si no está en el INPUT, no existe.
2. NO incluyas el proceso de pensamiento ("Paso 1", "Paso 2") en la respuesta final.
3. Escribe un texto único en prosa médica, sin viñetas ni etiquetas de sección internas.
4. Usa un tono analítico, conectando cómo una alteración puede influir en otra.
5. INDEPENDENCIA DE LOS EJEMPLOS: Usa los ejemplos solo para aprender el tono y la estructura, NO para copiar contenido.
6. NO copies frases textuales de los ejemplos. Cada informe debe tener su propio inicio y redacción original adaptada a los datos reales del paciente.
7. EXTENSIÓN: Un solo párrafo conciso de máximo 4-6 frases. Observa la brevedad de los ejemplos.

TAREA ACTUAL:
INPUT:
{context}

INSTRUCCIÓN FINAL: Redacta ahora el informe en un solo párrafo breve, con un inicio original y adaptado a los datos proporcionados."""
    }
}

# ── Canal de razonamiento ("thinking") de Ollama ──
# qwen3-0.6B y gemma4-nano-e2b son modelos de razonamiento: Ollama devuelve su cadena de
# pensamiento en un campo `thinking` separado de `content`. Si no se controla, el modelo puede
# agotar todo `num_predict` pensando y devolver `content` vacío. THINKING_MODELS marca qué
# model_key necesita que LLMService pase explícitamente el parámetro `think` a ollama.chat().
THINKING_MODELS = {"qwen3-0.6B", "gemma4-nano-e2b", "qwen3.5-4B"}

# Para qwen3-0.6B, el eje Zero-Shot/Few-Shot vs CoT/CoT+FS se mapea a activar o no su
# "thinking" nativo (en vez de instruirlo por texto a "razonar paso a paso", que no es fiable a
# este tamaño): Zero-Shot/Few-Shot = respuesta directa (más rápida); CoT/CoT+FS = con
# razonamiento previo (más lenta, más precisa). Un model_key sin entrada aquí pero presente en
# THINKING_MODELS usa razonamiento activado siempre (gemma4-nano-e2b).
# qwen3.5-4B: con think=True tarda >200s Y AÚN ASÍ agota num_predict pensando sin dejar
# contenido (mismo fallo de truncación, mucho más severo). Se desactiva siempre.
THINK_BY_METHOD = {
    "qwen3-0.6B": {"Zero-Shot": False, "Few-Shot": False, "CoT": True, "CoT+FS": True},
    "qwen3.5-4B": {"Zero-Shot": False, "Few-Shot": False, "CoT": False, "CoT+FS": False}
}
