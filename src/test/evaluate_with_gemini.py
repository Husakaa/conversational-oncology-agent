import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash-lite"

PROMPT_QWEN = """Actúa como un médico especialista en Oncología que evalúa análíticas de sangre oncológicas con amplia experiencia en documentación clínica hospitalaria.
En tu tarea, debes comparar:
- Biomarcadores Oncológicos (BO)
- NOTA GENERADA (NG)
Evalúa la calidad clínica de la NG respecto a los BO y como documento clínico independiente.
CRITERIOS DE EVALUACIÓN
1. Precisa (datos NG datos BO)
Evalúa si todos los datos presentes en la NG aparecen en la BO y son clínicamente correctos. Penaliza alucinaciones o invención de datos.
2. Exhaustiva (datos BO datos NG)
Evalúa estrictamente la información y los datos clínicamente relevantes desde el punto de vista del manejo médico oncológico: diagnóstico, tratamiento, plan de actuación, estadificación, comorbilidades, etc.
3. Útil
Independientemente de los BO, evalúa si la NG contiene información clínicamente relevante para la toma de decisiones.
4. Organizada
Evalúa la estructura lógica y la organización del documento.
5. Comprensible
Evalúa la claridad del lenguaje y el uso adecuado de terminología médica.
6. Concisa
Evalúa si la NG evita redundancias innecesarias.
7. Sintetizada
Evalúa si la información clínica está integrada de forma coherente.
8. Consistencia interna
Evalúa si existen contradicciones clínicas dentro de la NG.
PREGUNTAS ADICIONALES
Existe algún error de seguridad grave: errores clínicos potencialmente peligrosos (dosis incorrectas, errores de estadiaje, procedimientos inseguros, etc.).
Respuesta: SI/NO
Existe alguna contradicción clínica insalvable:
contradicciones que hacen la nota clínicamente incoherente.
Respuesta: SI/NO
Esta nota podría haber sido generada por un humano:
Respuesta: SI/NO
REGLAS IMPORTANTES
- No penalices cambios de estilo o reordenación si la información clínica es fiel.
- Penaliza fuertemente alucinaciones clínicas.
- Penaliza omisiones relevantes.
- Si detectas alucinaciones importantes, "precisa" <= 2.
- Si detectas contradicciones internas importantes, "consistencia_interna" <= 2.
- Cuando la puntuación de un criterio sea 5 (Excelente), el campo "justification" debe ser una cadena vacía "". No escribas ningún texto en ese campo.

ESCALA DE EVALUACIÓN
Las puntuaciones deben asignarse utilizando estrictamente la siguiente escala:
5 (Excelente)
Cumplimiento total del criterio evaluado. No se detectan errores clínicos, inconsistencias ni problemas de interpretación.
justification: ""
4 (Adecuado)
Existen errores menores, estilísticos o de redacción que no afectan a la seguridad clínica ni a la correcta interpretación del caso.
3 (Regular)
Existen errores moderados o limitaciones que deberían corregirse, pero el significado clínico general se mantiene y la nota sigue siendo interpretable.
2 (Deficiente)
Existen errores significativos, omisiones relevantes o problemas de coherencia que requerirían una reescritura parcial de la nota para que sea clínicamente correcta.
1 (Inaceptable)
Existe un error crítico de seguridad, información clínica falsa peligrosa, contradicciones graves o incoherencia que invalidan la nota.

FORMATO DE RESPUESTA
Devuelve exclusivamente un JSON válido con esta estructura exacta:
{{
"precisa": {{"score": 0, "justification": ""}},
"exhaustiva": {{"score": 0, "justification": ""}},
"util": {{"score": 0, "justification": ""}},
"organizada": {{"score": 0, "justification": ""}},
"comprensible": {{"score": 0, "justification": ""}},
"concisa": {{"score": 0, "justification": ""}},
"sintetizada": {{"score": 0, "justification": ""}},
"consistencia_interna": {{"score": 0, "justification": ""}},
"error_seguridad_grave": false,
"contradicciones_clinicas_insalvables": false,
"human_generated": false
}}
No incluyas texto fuera del JSON.
Cuando score sea 5, justification debe ser "".

AHORA EVALÚA LO SIGUIENTE:
Biomarcadores Oncológicos (BO):
{bo}

NOTA GENERADA (NG):
{ng}
"""

PROMPT_BIOMISTRAL = """Actúa como un médico especialista en Oncología con amplia experiencia académica y clínica, encargado de evaluar respuestas generadas a consultas médicas o dudas clínicas de otros profesionales.
En tu tarea, debes comparar:
- CONSULTA DEL USUARIO (CU)
- RESPUESTA GENERADA (RG)
Evalúa la exactitud, calidad clínica y utilidad de la RG respecto a la CU planteada.

CRITERIOS DE EVALUACIÓN
1. Precisa (exactitud médica)
Evalúa si la información médica proporcionada en la RG es correcta, científicamente exacta y aplicable a la CU. Penaliza fuertemente las alucinaciones, la invención de datos o el uso de conceptos médicos erróneos.
2. Exhaustiva (completitud)
Evalúa si la RG aborda todos los aspectos clínicamente relevantes de la CU. Penaliza la omisión de información vital que un oncólogo necesitaría saber para esa duda (ej. olvidar pautas críticas, grados de toxicidad relevantes o diagnósticos diferenciales).
3. Útil
Evalúa si la RG resuelve la duda planteada aportando valor práctico y orientación clínica accionable.
4. Organizada
Evalúa la estructura lógica de la respuesta (ej. presentación clara del concepto seguida del desarrollo o plan sugerido).
5. Comprensible
Evalúa la claridad de la exposición y el uso riguroso y adecuado de la terminología médica.
6. Concisa
Evalúa si la RG va al grano, evitando redundancias, introducciones robóticas o texto de relleno innecesario.
7. Sintetizada
Evalúa si la RG logra resumir e integrar la evidencia médica de forma directa, ciñéndose estrictamente a lo preguntado sin desviarse hacia temas médicos irrelevantes para esa consulta.
8. Consistencia interna
Evalúa si existen contradicciones lógicas o clínicas dentro de la propia RG.

PREGUNTAS ADICIONALES
Existe algún error de seguridad grave: afirmaciones que de aplicarse resultarían en un daño al paciente (recomendaciones de dosis incorrectas, omisión de toxicidades letales, procedimientos inseguros, etc.).
Respuesta: SI/NO
Existe alguna contradicción clínica insalvable: contradicciones que hacen la respuesta clínicamente incoherente o inútil.
Respuesta: SI/NO
Esta respuesta podría haber sido generada por un humano (ej. un médico residente o adjunto):
Respuesta: SI/NO

REGLAS IMPORTANTES
- No penalices cambios de estilo formal si la información clínica es correcta y profesional.
- Penaliza fuertemente las alucinaciones clínicas o conceptos inventados.
- Penaliza omisiones de seguridad relevantes (ej. olvidar mencionar una contraindicación clave si se pregunta por un fármaco).
- Si detectas alucinaciones importantes o datos falsos, "precisa" <= 2.
- Si detectas contradicciones internas importantes, "consistencia_interna" <= 2.
- Cuando la puntuación de un criterio sea 5 (Excelente), el campo "justification" debe ser una cadena vacía "". No escribas ningún texto en ese campo.

ESCALA DE EVALUACIÓN
Las puntuaciones deben asignarse utilizando estrictamente la siguiente escala:
5 (Excelente)
Cumplimiento total del criterio evaluado. No se detectan errores clínicos, inconsistencias ni problemas de interpretación.
justification: ""
4 (Adecuado)
Existen errores menores, falta de fluidez o detalles omitidos que no afectan a la seguridad clínica ni a la validez general de la respuesta.
3 (Regular)
Existen errores moderados, ambigüedades o falta de profundidad que deberían corregirse para considerar la respuesta completamente fiable.
2 (Deficiente)
Existen errores conceptuales significativos, omisiones relevantes o problemas de coherencia que obligarían a rehacer la respuesta por completo.
1 (Inaceptable)
Existe un error crítico de seguridad, información clínica falsa y peligrosa, o incoherencias que invalidan totalmente la respuesta.

FORMATO DE RESPUESTA
Devuelve exclusivamente un JSON válido con esta estructura exacta:
{{
"precisa": {{"score": 0, "justification": ""}},
"exhaustiva": {{"score": 0, "justification": ""}},
"util": {{"score": 0, "justification": ""}},
"organizada": {{"score": 0, "justification": ""}},
"comprensible": {{"score": 0, "justification": ""}},
"concisa": {{"score": 0, "justification": ""}},
"sintetizada": {{"score": 0, "justification": ""}},
"consistencia_interna": {{"score": 0, "justification": ""}},
"error_seguridad_grave": false,
"contradicciones_clinicas_insalvables": false,
"human_generated": false
}}
No incluyas texto fuera del JSON.
Cuando score sea 5, justification debe ser "".

AHORA EVALÚA LO SIGUIENTE:
CONSULTA DEL USUARIO (CU):
{cu}

RESPUESTA GENERADA (RG):
{rg}
"""

def cargar_bo(id_analitica, base_dir="analiticas/lote1"):
    ruta = f"{base_dir}/{id_analitica}.txt"
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(ruta, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            return ""
    except FileNotFoundError:
        return ""

def procesar_qwen_lote2():
    print(">>> Evaluando Qwen (Lote 2) con Gemini LLM-as-a-judge <<<")
    try:
        df = pd.read_csv("output/qwen_prompting_lote2.csv")
    except FileNotFoundError:
        print("No se encontró output/qwen_prompting_lote2.csv")
        return
        
    resultados = []
    json_path = "output/evaluacion_qwen_lote2_gemini.json"
    
    # Intentar cargar resultados previos para resumir
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                resultados = json.load(f)
            print(f"Retomando desde {len(resultados)} evaluaciones previas...")
        except Exception:
            pass
            
    evaluados = {(r.get("id_analitica"), r.get("estrategia")) for r in resultados}
    
    for idx, row in df.iterrows():
        id_analitica = row["id_analitica"]
        sintesis = row["sintesis"]
        estrategia = row["estrategia"]
        
        if (id_analitica, estrategia) in evaluados:
            continue
            
        if pd.isna(sintesis) or "ERROR" in str(sintesis) or "Asegúrate de que Ollama" in str(sintesis):
            print(f"[{id_analitica}] {estrategia} - Saltando por error o ausencia en sintesis")
            continue
            
        bo_text = cargar_bo(id_analitica, "analiticas/lote2")
        prompt = PROMPT_QWEN.format(bo=bo_text, ng=sintesis)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[{id_analitica}] {estrategia} - Evaluando (intento {attempt+1})...", end="", flush=True)
                res = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                data = json.loads(res.text)
                data["id_analitica"] = id_analitica
                data["estrategia"] = estrategia
                resultados.append(data)
                
                # Guardado continuo
                with open(json_path, "w", encoding='utf-8') as f:
                    json.dump(resultados, f, indent=2, ensure_ascii=False)
                    
                print(" ✅ Hecho")
                time.sleep(4.1) # Respetar rate limits de Gemini free tier (15 RPM -> 4s por request)
                break
            except Exception as e:
                print(f" ❌ Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10) # Wait 10 seconds before retrying
            
    if resultados:
        with open("output/evaluacion_qwen_lote2_gemini.json", "w", encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print("Guardado en output/evaluacion_qwen_lote2_gemini.json")
            
def procesar_biomistral_lote2():
    print("\n>>> Evaluando Biomistral (Lote 2) con Gemini LLM-as-a-judge <<<")
    try:
        df = pd.read_csv("output/biomistral_prompting_lote2.csv")
    except FileNotFoundError:
        print("No se encontró output/biomistral_prompting_lote2.csv")
        return
        
    resultados = []
    json_path = "output/evaluacion_biomistral_lote2_gemini.json"
    
    # Intentar cargar resultados previos para resumir
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                resultados = json.load(f)
            print(f"Retomando desde {len(resultados)} evaluaciones previas...")
        except Exception:
            pass
            
    evaluados = {(r.get("pregunta_id"), r.get("estrategia")) for r in resultados}
    
    for idx, row in df.iterrows():
        pregunta = row["pregunta"]
        respuesta = row["respuesta"]
        estrategia = row["estrategia"]
        pregunta_id = row["pregunta_id"]
        
        if (pregunta_id, estrategia) in evaluados:
            continue
            
        if pd.isna(respuesta) or "ERROR" in str(respuesta) or "Asegúrate de que Ollama" in str(respuesta):
            print(f"[Pregunta {pregunta_id}] {estrategia} - Saltando por error o ausencia en respuesta")
            continue
            
        prompt = PROMPT_BIOMISTRAL.format(cu=pregunta, rg=respuesta)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[Pregunta {pregunta_id}] {estrategia} - Evaluando (intento {attempt+1})...", end="", flush=True)
                res = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                data = json.loads(res.text)
                data["pregunta_id"] = pregunta_id
                data["estrategia"] = estrategia
                resultados.append(data)
                
                # Guardado continuo
                with open(json_path, "w", encoding='utf-8') as f:
                    json.dump(resultados, f, indent=2, ensure_ascii=False)
                    
                print(" ✅ Hecho")
                time.sleep(4.1) # Respetar rate limits
                break
            except Exception as e:
                print(f" ❌ Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10) # Wait 10 seconds before retrying
            
    if resultados:
        with open("output/evaluacion_biomistral_lote2_gemini.json", "w", encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print("Guardado en output/evaluacion_biomistral_lote2_gemini.json")

if __name__ == "__main__":
    procesar_qwen_lote2()
    procesar_biomistral_lote2()
