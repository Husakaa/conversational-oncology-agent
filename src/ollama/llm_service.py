import logging
from typing import Dict, Any, Tuple
import ollama

from src.ui.formatters import clinical_context

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class LLMService:
    def __init__(self):
        self.models = {
            "qwen-oncologo": "qwen-oncologo",
            "biomistral-oncologo": "biomistral-oncologo"
        }
        self.options_qwen = {
            'temperature': 0.0,
            'top_k': 10,
            'top_p': 0.1,
            'num_ctx': 8192,
            'num_predict': 2048
        }
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


    def _build_prompt(self, methodology: str, context: str, consult: bool = False) -> str:
        if consult:
            return f"""Eres un oncólogo experto. Tienes un alto dominio del ámbito médico.
             Vas a recibir consultas y debes contestar de forma altamente explicativa dando detalles.
             
             Consulta: {context}"""
        else:
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
        context = clinical_context(extracted_data)
        prompt_final = self._build_prompt(method, context)
        logging.info(f"Lanzando inferencia a Ollama (Modelo: {self.models['qwen-oncologo']} | Método: {method})")
        try:
            response = ollama.chat(
                model=self.models["qwen-oncologo"],
                messages=[{'role': 'user', 'content': prompt_final}],
                options=self.options_qwen
            )
            return context, response['message']['content']
        except Exception as e:
            logging.error(f"Error en la comunicación con Ollama: {str(e)}")
            return context, f"Error generando la síntesis clínica: Asegúrate de que Ollama está ejecutándose y el modelo '{self.models['qwen-oncologo']}' está descargado."

    def generate_response(self, consult: str, method: str = "Zero-Shot") -> str:
        prompt = self._build_prompt(methodology=method, context=consult, consult=True)
        logging.info(f"Lanzando consulta a Ollama (Modelo: {self.models['biomistral-oncologo']})")
        try:
            response = ollama.chat(
                model=self.models["biomistral-oncologo"],
                messages=[{'role': 'user', 'content': prompt}],
                options=self.options_qwen
            )
            return response['message']['content']
        except Exception as e:
            logging.error(f"Error en la comunicación con Ollama: {str(e)}")
            return f"Error generando la respuesta: Asegúrate de que Ollama está ejecutándose y el modelo '{self.models['biomistral-oncologo']}' está descargado."