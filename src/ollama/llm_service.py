import logging
from typing import Dict, Any, Tuple
import ollama

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class LLMService:
    """
    Servicio para la integración con Modelos de Lenguaje de Pequeña Escala a través de Ollama.
    Encapsula las estrategias de Prompting y los hiperparámetros de inferencia clínica.
    """

    def __init__(self, model_name: str = "qwen-oncologo"):
        self.model_name = model_name
        
        # Hiperparámetros ajustados para determinismo y rigor médico 
        self.options = {
            'temperature': 0.0,      # Bajo = más determinista 
            'top_k': 10,             # Limita el vocabulario a las K palabras más probables
            'top_p': 0.1,            # Nucleus sampling restrictivo
            'num_ctx': 8192,         # Tamaño de ventana de contexto
            'num_predict': 2048      # Límite de tokens en la respuesta
        }

        # Ejemplo "Gold Standard" para la metodología Few-Shot
        self.ejemplo_fs_input = """[Fecha no especificada]: Hb: 14.5 | Plaquetas: 147 | Neutrófilos: 3.41 | Creatinina: 0.50 | GGT: 297 | ALT: 83 | resto bien.

----------------------------------------------------------------------------------------------------

SECCIÓN C: ESTRATIFICACIÓN DE TOXICIDAD (CTCAE v5.0) ⚠️
    - GGT (297 U/L): Grado 2. Elevación GGT Moderada (214 - 425 U/L). (Ref: 5.0 - 40.0).
    - ALT (83 U/L): Grado 1. Elevación ALT (GPT) Leve (41 - 120 U/L). (Ref: 3.0 - 40.0).
    - FA (180 U/L): Grado 1. Elevación Fosfatasa Alc. Leve (117 - 291 U/L). (Ref: 35.0 - 104.0).
    - AST (51 U/L): Grado 1. Elevación AST (GOT) Leve (41 - 120 U/L). (Ref: 3.0 - 40.0).
    - LDH (274 U/L): Grado 0 (Normal). Nota: Resultado corregido estadísticamente por hemólisis. (Ref: 200.0 - 380.0)."""

        self.ejemplo_fs_output = """SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊

El evento limitante de dosis es la elevación de GGT Grado 2, que junto a la elevación leve de fosfatasa alcalina y transaminasas (G1), sugiere un perfil de colestasis o afectación biliar incipiente. Las series hematológicas (neutrófilos y plaquetas) y la función renal se encuentran preservadas dentro de rangos seguros. Se recomienda monitorizar el perfil hepático en el próximo ciclo y repetir la toma de LDH debido a la interferencia por hemólisis."""

    def _format_clinical_context(self, extracted_data: Dict[str, Any]) -> str:
        """
        Transforma el diccionario de datos extraídos con Regex en el texto plano
        de las Secciones A y C para alimentar al modelo de lenguaje.
        """
        # --- SECCIÓN A: Línea de Resumen Rápido ---
        fijos = ['Hb', 'Plaquetas', 'Neutrófilos', 'Creatinina']
        linea_a_parts = ["[Analítica Actual]:"]
        
        for f in fijos:
            val = extracted_data.get(f, {}).get("valor", "N/A")
            linea_a_parts.append(f"{f}: {val}")
            
        # Buscar hasta 2 valores críticos adicionales (Grado >= 1)
        criticos = []
        for k, v in extracted_data.items():
            if k not in fijos and v.get("ctcae", {}).get("grado", 0) >= 1:
                criticos.append((k, v))
                
        # Ordenar por gravedad
        criticos.sort(key=lambda x: x[1]["ctcae"]["grado"], reverse=True)
        
        for k, v in criticos[:2]:
            linea_a_parts.append(f"{k}: {v['valor']}")
            
        linea_a_parts.append("resto bien.")
        linea_a = " | ".join(linea_a_parts)

        # --- SECCIÓN C: Estratificación CTCAE ---
        seccion_c = "SECCIÓN C: ESTRATIFICACIÓN DE TOXICIDAD (CTCAE v5.0) ⚠️\n"
        hay_tox = False
        all_tox = [(k, v) for k, v in extracted_data.items() if v.get("ctcae", {}).get("grado", 0) >= 1]
        all_tox.sort(key=lambda x: x[1]["ctcae"]["grado"], reverse=True)
        
        for k, v in all_tox:
            hay_tox = True
            ctcae = v["ctcae"]
            ref = v.get("rango_referencia", {})
            ref_str = f"{ref.get('inf', '?')} - {ref.get('sup', '?')}"
            
            seccion_c += f"    - {k} ({v['valor']} {v['unidad']}): Grado {ctcae['grado']}. {ctcae['descripcion']}. (Ref Lab: {ref_str}).\n"
            
        if not hay_tox:
            seccion_c += "    - No se observan toxicidades significativas.\n"

        return f"{linea_a}\n\n----------------------------------------------------------------------------------------------------\n\n{seccion_c}"

    def _build_prompt(self, methodology: str, context: str) -> str:
        """Construye el prompt en base a la estrategia experimental seleccionada."""
        
        if methodology == "Zero-Shot":
            return f"""Basado en estos resultados, redacta únicamente la 'SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊'.
            
RESULTADOS:
{context}

SECCIÓN D:"""

        elif methodology == "Few-Shot":
            return f"""Eres un oncólogo experto. Sigue el estilo del ejemplo para redactar la 'SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊'.
            
EJEMPLO:
INPUT:
{self.ejemplo_fs_input}
OUTPUT SECCIÓN D:
{self.ejemplo_fs_output}

TAREA ACTUAL:
INPUT:
{context}
OUTPUT SECCIÓN D:"""

        elif methodology == "CoT":
            return f"""Eres un oncólogo experto. Realiza un análisis clínico exhaustivo siguiendo estos pasos:
PASO 1: Analiza cada valor alterado individualmente y su grado de toxicidad.
PASO 2: Explica la fisiopatología conjunta (qué está pasando en el cuerpo del paciente).
PASO 3: Redacta una conclusión formal para la 'SECCIÓN D: SÍNTESIS CLÍNICA Y EVOLUCIÓN 📊'.

IMPORTANTE: No seas breve. Justifica tus conclusiones basándote en los datos.

DATOS:
{context}"""
        
        else:
            raise ValueError(f"Metodología '{methodology}' no soportada.")

    def generate_synthesis(self, extracted_data: Dict[str, Any], method: str = "CoT") -> Tuple[str, str]:
        """
        Función principal. Toma el JSON de la capa determinista y devuelve
        tanto el contexto generado como la respuesta del modelo local.
        """
        # 1. Formatear datos a texto clínico
        clinical_context = self._format_clinical_context(extracted_data)
        
        # 2. Construir el prompt final
        prompt_final = self._build_prompt(method, clinical_context)
        
        logging.info(f"Lanzando inferencia a Ollama (Modelo: {self.model_name} | Método: {method})")
        
        try:
            # 3. Llamada a la API de Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt_final}],
                options=self.options
            )
            return clinical_context, response['message']['content']
            
        except Exception as e:
            logging.error(f"Error en la comunicación con Ollama: {str(e)}")
            return clinical_context, f"Error generando la síntesis clínica: Asegúrate de que Ollama está ejecutándose y el modelo '{self.model_name}' está descargado."


# ==========================================
# BLOQUE DE PRUEBAS LOCALES
# ==========================================
if __name__ == "__main__":
    # Simulación de un output de la clase MedicalExtractor
    mock_extractor_data = {
        "Hb": {"valor": 7.5, "unidad": "g/dL", "rango_referencia": {"inf": 12.0, "sup": 16.0}, "ctcae": {"grado": 3, "etiqueta": "Low", "descripcion": "Anemia: Severa (<8.0 g/dL)"}},
        "Plaquetas": {"valor": 30.0, "unidad": "x10^9/L", "rango_referencia": {"inf": 150.0, "sup": 400.0}, "ctcae": {"grado": 3, "etiqueta": "Low", "descripcion": "Trombocitopenia: Severa (25.0 - 50.0)"}},
        "Creatinina": {"valor": 1.1, "unidad": "mg/dL", "rango_referencia": {"inf": 0.6, "sup": 1.1}, "ctcae": {"grado": 1, "etiqueta": "High", "descripcion": "Creatinina elevada: Leve (1.03 - 1.53)"}},
        "Neutrófilos": {"valor": 2.5, "unidad": "x10^9/L", "rango_referencia": {"inf": 1.5, "sup": 7.5}, "ctcae": {"grado": 0, "etiqueta": "Normal", "descripcion": "Normal"}}
    }
    
    
    llm = LLMService(model_name="qwen2.5:latest") # Usa tu modelo local por defecto para la prueba
    context, synthesis = llm.generate_synthesis(mock_extractor_data, method="Chain-of-Thought")
    
    print("\n=== CONTEXTO ENVIADO (SECCIÓN A y C) ===")
    print(context)
    print("\n=== RESPUESTA DEL SLM (SECCIÓN D) ===")
    print(synthesis)