import logging
from typing import Dict, Any, Tuple
import ollama

from src.ui.formatters import clinical_context, biomarkers_to_markdown
from src.config import OLLAMA_MODELS, QWEN25_OPTIONS, QWEN3_OPTIONS, QWEN35_OPTIONS, GEMMA4_OPTIONS, BIOMISTRAL_OPTIONS, MODEL_PROMPTS

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class LLMService:
    def __init__(self):
        self.models = OLLAMA_MODELS
        self.options_qwen25 = QWEN25_OPTIONS
        self.options_qwen3 = QWEN3_OPTIONS
        self.options_qwen35 = QWEN35_OPTIONS
        self.options_gemma4 = GEMMA4_OPTIONS
        self.options_biomistral = BIOMISTRAL_OPTIONS
        self.last_biomarkers = None


    def _build_prompt(self, model_id: str, methodology: str, context: str, consult: bool = False) -> str:
        model_prompts = MODEL_PROMPTS.get(model_id, MODEL_PROMPTS["default"])
        if consult:
            prompt_template = model_prompts.get("consult", MODEL_PROMPTS["default"]["consult"])
            context_block = ""
            if self.last_biomarkers:
                context_block = f"\n\nRESULTADOS DE LABORATORIO DEL PACIENTE ACTUAL:\n{self.last_biomarkers}\n"
            return prompt_template.format(context_block=context_block, context=context)
        else:
            prompt_template = model_prompts.get(methodology)
            if not prompt_template:
                raise ValueError(f"Metodología '{methodology}' no soportada para el modelo '{model_id}'.")
            return prompt_template.format(context=context)

    def generate_synthesis(self, extracted_data: Dict[str, Any], method: str = "CoT+FS", model_key: str = "qwen2.5-7B") -> Tuple[str, str]:
        context = clinical_context(extracted_data)
        model_id = self.models.get(model_key, model_key)
        prompt_final = self._build_prompt(model_id, method, context)
        logging.info(f"Lanzando inferencia a Ollama (Modelo: {model_id} | Método: {method})")
        
        options = self.options_qwen35
        if model_key == "qwen2.5-7B": options = self.options_qwen25
        elif model_key == "gemma4-nano-e2b": options = self.options_gemma4
        elif model_key == "biomistral-7B": options = self.options_biomistral

        try:
            response = ollama.chat(
                model=model_id,
                messages=[{'role': 'user', 'content': prompt_final}],
                options=options
            )
            synthesis = response['message']['content']
            self.last_biomarkers = biomarkers_to_markdown(extracted_data)
            logging.info("Tabla de biomarcadores guardada como contexto para BioMistral.")
            return context, synthesis
        except Exception as e:
            logging.error(f"Error en la comunicación con Ollama: {str(e)}")
            return context, f"Error generando la síntesis clínica: Asegúrate de que Ollama está ejecutándose y el modelo '{model_id}' está descargado."

    def generate_response(self, consult: str, method: str = "CoT+FS", model_key: str = "biomistral-7B") -> str:
        model_id = self.models.get(model_key, "biomistral-7B")
        prompt = self._build_prompt(model_id, methodology=method, context=consult, consult=True)
        logging.info(f"Lanzando consulta a Ollama (Modelo: {model_id})")
        
        options = self.options_biomistral
        if model_key == "qwen3-0.6B": options = self.options_qwen3
        elif model_key == "qwen2.5-7B": options = self.options_qwen25
        elif model_key == "gemma4-nano-e2b": options = self.options_gemma4

        try:
            response = ollama.chat(
                model=model_id,
                messages=[{'role': 'user', 'content': prompt}],
                options=options
            )
            return response['message']['content']
        except Exception as e:
            logging.error(f"Error en la comunicación con Ollama: {str(e)}")
            return f"Error generando la respuesta: Asegúrate de que Ollama está ejecutándose y el modelo '{model_id}' está descargado."